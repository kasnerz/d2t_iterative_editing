#!/usr/bin/env python3

import numpy as np
import os
import logging
import re
import shutil
import json
import argparse
import pytorch_lightning as pl
import torch
import pickle

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from data import get_dataset_class

from collections import defaultdict
from datasets import load_dataset, dataset_dict, Dataset

import phrase_vocabulary_optimization
import tagging
import tagging_converter
from utils import lt_utils as utils

from torch.nn.utils.rnn import pad_sequence

from collections import OrderedDict
from transformers import (
    AdamW,
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)


class FLAGS(defaultdict):
    """
    Class for confuguration of the LT model maintaining backward compatibility
    with Abseil (https://abseil.io) used in the original LT implementation
    """
    def __init__(self):
        # yields a defaultdict with attributes acting as keys
        super(FLAGS, self).__init__(None)
        self.__dict__ = self


class FuseModel:
    def fuse(self, first, second):
        raise NotImplementedError()


class LTDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # disable the "huggingface/tokenizers: The current process just got forked" warning
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name,
                                                       use_fast=True)

        if self.args.train_only:
            exp_output_dir = os.path.join(self.args.output_dir,
                                          self.args.experiment,
                                          str(self.args.vocab_size))
            os.makedirs(exp_output_dir, exist_ok=True)

            label_map_file = os.path.join(exp_output_dir, "label_map.txt")
            self.label_map = utils.read_label_map(label_map_file)

    def prepare_data(self):
        """
        The training pipeline for LT:
        1. extracting phrases used for the vocabulary
        2. converting text to tags
        3. training the model
        """
        dataset_name = get_dataset_class(self.args.dataset).name
        self.dataset_dir = os.path.join("data", dataset_name, self.args.mode)

        exp_output_dir = os.path.join(self.args.output_dir,
                                      self.args.experiment,
                                      str(self.args.vocab_size))
        os.makedirs(exp_output_dir, exist_ok=True)

        if not self.args.train_only:
            logger.info(
                "Building vocabulary and computing labels (this step can be skipped using the --train_only flag)"
            )
            self._phrase_vocabulary_optimization(
                dataset_dir=self.dataset_dir,
                vocab_size=self.args.vocab_size,
                max_input_examples=self.args.max_input_examples,
                exp_output_dir=exp_output_dir,
                experiment_name=self.args.experiment,
                enable_swap_tag=self.args.enable_swap_tag)

            self._build_examples(
                split="train",
                dataset_dir=self.dataset_dir,
                exp_output_dir=exp_output_dir,
                max_input_examples=self.args.max_input_examples,
                output_arbitrary_targets_for_infeasible_examples=False)

            self._build_examples(
                split="dev",
                dataset_dir=self.dataset_dir,
                exp_output_dir=exp_output_dir,
                max_input_examples=None,
                output_arbitrary_targets_for_infeasible_examples=True)

    def setup(self, stage):
        exp_output_dir = os.path.join(
            self.args.output_dir,
            self.args.experiment,
            str(self.args.vocab_size),
        )

        with open(os.path.join(exp_output_dir, "dev.bin"), 'rb') as f_in:
            dev = pickle.load(f_in)

        with open(os.path.join(exp_output_dir, "train.bin"), 'rb') as f_in:
            train = pickle.load(f_in)

        self.dataset = dataset_dict.DatasetDict({
            "dev":
            Dataset.from_dict(dev),
            "train":
            Dataset.from_dict(train)
        })

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self._convert_to_features,
                batched=True,
                remove_columns=['labels'],
            )
            self.dataset[split].set_format(
                type="torch",
                columns=[
                    "attention_mask", "token_type_ids", "input_ids", "labels"
                ])

    def _pad_sequence(self, batch):
        batch_collated = {}

        for key in ["labels", "input_ids", "attention_mask", "token_type_ids"]:
            elems = [x[key] for x in batch]
            elems_pad = pad_sequence(elems, batch_first=True, padding_value=0)
            batch_collated[key] = elems_pad

        return batch_collated

    def train_dataloader(self):
        return DataLoader(self.dataset['train'],
                          batch_size=self.args.batch_size,
                          num_workers=self.args.max_threads,
                          collate_fn=self._pad_sequence)

    def val_dataloader(self):
        return DataLoader(self.dataset['dev'],
                          batch_size=self.args.batch_size,
                          num_workers=self.args.max_threads,
                          collate_fn=self._pad_sequence)

    def test_dataloader(self):
        return DataLoader(self.dataset['test'],
                          batch_size=self.args.batch_size,
                          num_workers=self.args.max_threads,
                          collate_fn=self._pad_sequence)

    def _convert_to_features(self, example_batch, indices=None):
        tokens_batch = [sent.split() for sent in example_batch["text"]]

        # encode the pre-tokenized text
        features = self.tokenizer.batch_encode_plus(
            tokens_batch,
            add_special_tokens=True,
            max_length=self.args.max_length,
            truncation=True,
            is_split_into_words=True)
        features['labels'] = self._align_labels_with_tokens(
            features, example_batch['labels'])

        return features

    def _align_labels_with_tokens(self, features, labels):
        aligned_labels_batch = []

        for b in range(len(labels)):
            default_label = self.label_map["KEEP"]
            aligned_labels = list(
                map(lambda l: default_label if l is None else labels[b][l],
                    features.words(b)))
            aligned_labels_batch.append(aligned_labels)

        return aligned_labels_batch

    def _phrase_vocabulary_optimization(self, dataset_dir, vocab_size,
                                        max_input_examples, exp_output_dir,
                                        experiment_name, enable_swap_tag):
        flags = FLAGS()

        flags.input_file = os.path.join(dataset_dir, "train")
        flags.input_format = "fuse"
        flags.vocabulary_size = vocab_size
        flags.max_input_examples = max_input_examples
        flags.output_file = os.path.join(exp_output_dir, "label_map.txt")
        flags.enable_swap_tag = enable_swap_tag
        flags.num_extra_statistics = 100

        logger.info("Beginning phrase vocabulary optimization...")
        phrase_vocabulary_optimization.main(flags)
        logger.info("Phrase vocabulary optimization finished.")

    def _build_examples(self, split, dataset_dir, exp_output_dir,
                        max_input_examples,
                        output_arbitrary_targets_for_infeasible_examples):
        label_map_file = os.path.join(exp_output_dir, "label_map.txt")
        input_file = os.path.join(dataset_dir, split)
        output_file = os.path.join(exp_output_dir, f"{split}.bin")

        self.label_map = utils.read_label_map(label_map_file)
        self.converter = tagging_converter.TaggingConverter(
            tagging_converter.get_phrase_vocabulary_from_label_map(
                self.label_map))

        examples = {"text": [], "labels": []}

        logger.info(f"Processing {split} dataset")

        with open(f"{input_file}.in") as f_in, open(
                f"{input_file}.ref") as f_ref:
            for i, (source, target) in enumerate(zip(f_in, f_ref)):
                source = source.rstrip('\n')
                target = target.rstrip('\n')

                labels = self._convert_to_labels(
                    source, target,
                    output_arbitrary_targets_for_infeasible_examples)

                if i % 10000 == 0:
                    logger.info(f"{i} examples processed")

                if not labels:
                    continue

                examples["text"].append(source)
                examples["labels"].append(labels)

                if max_input_examples is not None and i > max_input_examples:
                    break

        with open(output_file, 'wb') as f_out:
            pickle.dump(examples, f_out, protocol=4)

    def _convert_to_labels(self, source, target,
                           output_arbitrary_targets_for_infeasible_examples):
        task = tagging.EditingTask([source])
        if target is not None:
            tags = self.converter.compute_tags(task, target)
            if not tags:
                if output_arbitrary_targets_for_infeasible_examples:
                    # Create a tag sequence [KEEP, DELETE, KEEP, DELETE, ...] which is
                    # unlikely to be predicted by chance.
                    tags = [
                        tagging.Tag('KEEP') if i %
                        2 == 0 else tagging.Tag('DELETE')
                        for i, _ in enumerate(task.source_tokens)
                    ]
                else:
                    return None
        else:
            # If target is not provided, we set all target labels to KEEP.
            tags = [tagging.Tag('KEEP') for _ in task.source_tokens]

        labels = [self.label_map[str(tag)] for tag in tags]

        return labels


class TransformerDecoderLayer(nn.Module):
    """
    https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerDecoderLayer
    """
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.proj_layer = nn.Linear(2*d_model, d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.gelu

    def forward(self,
                tgt,
                memory,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None):

        # directly consuming encoder outputs instead of full multihead attention
        # follows implementation in https://github.com/google-research/lasertagger/blob/master/transformer_decoder.py
        tgt = torch.cat([tgt, memory], dim=-1)
        tgt = self.proj_layer(tgt)

        tgt2 = self.self_attn(tgt, tgt,tgt,
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class LaserTagger(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters()

        self.hidden_size = 768
        # KEEP / DELETE for each token + standalone
        self.num_labels = self.args.vocab_size * 2 + 2

        if self.args.enable_swap_tag:
            self.num_labels += 1

        self.encoder = AutoModel.from_pretrained(args.model_name,
                                              return_dict=True)
        decoder_layer = TransformerDecoderLayer(d_model=self.hidden_size,
                                                nhead=4,
                                                dim_feedforward=3072,
                                                dropout=0.1)
        decoder_norm = nn.LayerNorm(self.hidden_size)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer,
                                                         num_layers=1,
                                                         norm=decoder_norm)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, **inputs):
        encoder_output = self.encoder(
            input_ids=inputs["input_ids"],
            token_type_ids=inputs["token_type_ids"],
            attention_mask=inputs["attention_mask"])["last_hidden_state"]

        embedding_layer = self.encoder.embeddings
        decoder_input = embedding_layer(inputs["input_ids"]).permute(1, 0, 2)
        encoder_output = encoder_output.permute(1, 0, 2)

        mask = self.generate_square_subsequent_mask(encoder_output.size(0))
        decoder_output = self.transformer_decoder(tgt=decoder_input,
                                                  memory=encoder_output)
        decoder_output = decoder_output.permute(1, 0, 2)

        logits = self.classifier(decoder_output)
        attention_mask = inputs["attention_mask"]

        if "labels" in inputs:
            labels = inputs["labels"]

            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1),
                    torch.tensor(self.loss_fn.ignore_index).type_as(labels))
                loss = self.loss_fn(active_logits, active_labels)
            else:
                loss = self.loss_fn(logits.view(-1, self.num_labels),
                                    labels.view(-1))
        else:
            loss = None

        return {"loss": loss, "logits": logits}

    def training_step(self, batch, batch_idx):
        labels = batch["labels"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        outputs = self(input_ids=input_ids,
                       token_type_ids=token_type_ids,
                       attention_mask=attention_mask,
                       labels=labels)

        loss = outputs["loss"]
        self.log('loss/train', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch["labels"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        outputs = self(input_ids=input_ids,
                       token_type_ids=token_type_ids,
                       attention_mask=attention_mask,
                       labels=labels)

        loss = outputs["loss"]

        self.log('loss/val', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.encoder.parameters(),
                          lr=self.args.learning_rate,
                          eps=self.args.adam_epsilon,
                          betas=(self.args.adam_beta1, self.args.adam_beta2))

        total_steps = self.args.max_steps if self.args.max_steps else len(
            self.train_dataloader()) * self.args.max_epochs
        warmup_steps = total_steps * self.args.warmup_proportion

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps)

        logger.info(f"Learning rate: {self.args.learning_rate}")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Warmup steps: {warmup_steps}")

        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser],
                                         add_help=False)
        parser.add_argument("--learning_rate", default=2e-5, type=float)
        parser.add_argument("--adam_epsilon", default=1e-9, type=float)
        parser.add_argument("--adam_beta1", default=0.9, type=float)
        parser.add_argument("--adam_beta2", default=0.997, type=float)
        parser.add_argument("--warmup_proportion", default=0.1, type=float)
        parser.add_argument("--label_smoothing", default=0.1, type=float)

        return parser

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz,
                                      device=self.device)) == 1).transpose(
                                          0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
            mask == 1, float(0.0))
        return mask


class LTFuseModel(FuseModel):
    def __init__(self, args, model_path, model_name):
        exp_output_dir = os.path.join(args.exp_dir, args.experiment,
                                      str(args.vocab_size))
        label_map_file = os.path.join(exp_output_dir, "label_map.txt")
        label_map = utils.read_label_map(label_map_file)

        self._id_2_tag = {tag_id: tag for tag, tag_id in label_map.items()}

        model_path = os.path.join(model_path, "last.ckpt")
        self.model = LaserTagger.load_from_checkpoint(model_path)
        logger.info(f"Loaded model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                       use_fast=True)

    def beam_search_decoder(self, logits, k):
        batch_size, seq_length, vocab_size = logits.shape
        log_prob, indices = logits[:, 0, :].topk(k, sorted=True)
        indices = indices.unsqueeze(-1)

        for i in range(1, seq_length):
            log_prob = log_prob.unsqueeze(-1) + logits[:, i, :].unsqueeze(
                1).repeat(1, k, 1)
            log_prob, index = log_prob.view(batch_size, -1).topk(k,
                                                                 sorted=True)
            indices = torch.cat(
                [indices, index.unsqueeze(-1) % vocab_size], dim=-1)

        return indices

    def fuse(self, first, second, beam_size=10):
        sentence = first + " " + second
        inputs = self.tokenizer.encode_plus(sentence,
                                            return_tensors="pt",
                                            return_offsets_mapping=True,
                                            add_special_tokens=True)
        logits = self.model(input_ids=inputs["input_ids"],
                            token_type_ids=inputs["token_type_ids"],
                            attention_mask=inputs["attention_mask"])["logits"]
        task = tagging.EditingTask([sentence])

        if beam_size > 1:
            predictions = self.beam_search_decoder(logits, 10)
            predictions = predictions[0].detach().numpy()
        else:
            predictions = np.argmax(logits.detach().numpy(), axis=2)

        outputs = []
        for labels in predictions:
            offset_mapping = inputs["offset_mapping"][0]
            labels_for_tokens = [
                label for offset, label in zip(offset_mapping, labels)
                if self._is_at_word_boundary(sentence, offset)
            ]
            tags_for_tokens = [
                tagging.Tag(self._id_2_tag[label])
                for label in labels_for_tokens
            ]
            try:
                out = task.realize_output(tags_for_tokens)
            except:
                logger.error("Fusing unsuccessful")
                out = sentence
            outputs.append(out)

        return outputs

    def _is_at_word_boundary(self, sentence, offset):
        # first token (which is not a special token) or a token for which previous character is a space
        return (offset[0] == 0
                and offset[1] != 0) or sentence[offset[0] - 1] == " "
