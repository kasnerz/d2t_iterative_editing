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

from torch.utils.data import DataLoader
from data import get_dataset_class

from collections import defaultdict
from datasets import load_dataset, dataset_dict, Dataset

from lasertagger import phrase_vocabulary_optimization
from lasertagger import preprocess_main
from lasertagger import tagging
from lasertagger import tagging_converter
from lasertagger import utils

from torch.nn.utils.rnn import pad_sequence

from collections import OrderedDict
from transformers import (
    AdamW,
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    BertLMHeadModel,
    BertForTokenClassification,
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
            # TODO mask?
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


class LaserTagger(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters()
        self.model = BertForTokenClassification.from_pretrained(
            args.model_name,
            return_dict=True,
            num_labels=self.args.vocab_size * 2 +
            2)  # KEEP / DELETE for each token + standalone

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        labels = batch["labels"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        outputs = self.model(input_ids=input_ids,
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

        outputs = self.model(input_ids=input_ids,
                             token_type_ids=token_type_ids,
                             attention_mask=attention_mask,
                             labels=labels)

        val_loss, logits = outputs["loss"], outputs["logits"]
        # preds = torch.argmax(logits, axis=1)
        self.log('loss/val', val_loss, prog_bar=True)

        # return {'loss/val': val_loss, "preds": preds, "labels": labels}

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(),
                          lr=self.args.learning_rate,
                          eps=self.args.adam_epsilon,
                          betas=(self.args.adam_beta1, self.args.adam_beta2))

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.max_steps * self.args.warmup_proportion,
            num_training_steps=self.args.max_steps)
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


class LTFuseModel(FuseModel):
    def __init__(self, args, model_path, model_name):

        exp_output_dir = os.path.join(args.exp_dir, args.experiment,
                                      str(args.vocab_size))
        label_map_file = os.path.join(exp_output_dir, "label_map.txt")
        label_map = utils.read_label_map(label_map_file)

        self._id_2_tag = {tag_id: tag for tag, tag_id in label_map.items()}

        self.model = LaserTagger.load_from_checkpoint(
            os.path.join(model_path, "model.ckpt"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                       use_fast=True)

    def fuse(self, first, second):
        sentence = first #TODO debug
        # sentence = first + " " + second
        inputs = self.tokenizer.encode_plus(sentence,
                                            return_tensors="pt",
                                            return_offsets_mapping=True)
        predictions = self.model(input_ids=inputs["input_ids"],
                                 token_type_ids=inputs["token_type_ids"],
                                 attention_mask=inputs["attention_mask"])[0]

        labels = np.argmax(predictions.detach().numpy(), axis=2)[0]
        task = tagging.EditingTask([sentence])

        offset_mapping = inputs["offset_mapping"][0]
        labels_for_tokens = [
            label for offset, label in zip(offset_mapping, labels)
            if self._is_at_word_boundary(sentence, offset)
        ]
        tags_for_tokens = [
            tagging.Tag(self._id_2_tag[label]) for label in labels_for_tokens
        ]

        try:
            out = task.realize_output(tags_for_tokens)
        except:
            logger.error("Fusing unsuccessful")
            out = sentence

        return out

        # beam = self.model.model.generate(**inputs, max_length=50, num_beams=10, early_stopping=True, num_return_sequences=10)

    def _is_at_word_boundary(self, sentence, offset):
        # first token (which is not a special token) or a token for which previous character is a space
        return (offset[0] == 0
                and offset[1] != 0) or sentence[offset[0] - 1] == " "
