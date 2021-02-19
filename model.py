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

from torch.utils.data import DataLoader
from data import get_dataset_class

from collections import defaultdict
from datasets import load_dataset

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
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, use_fast=True)

        # disable the "huggingface/tokenizers: The current process just got forked" warning
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        
    def prepare_data(self):
        """
        The training pipeline for LT:
        1. extracting phrases used for the vocabulary
        2. converting text to tags
        3. training the model
        """
        dataset_name = get_dataset_class(self.args.dataset).name
        self.dataset_dir = os.path.join("data",
                               dataset_name,
                               self.args.mode)

        exp_output_dir = os.path.join(self.args.output_dir,
            self.args.experiment,
            str(self.args.vocab_size)
        )
        os.makedirs(exp_output_dir, exist_ok=True)

        if not (self.args.train_only or self.args.export_only):
            # self._phrase_vocabulary_optimization(
            #     dataset_dir=self.dataset_dir,
            #     vocab_size=self.args.vocab_size,
            #     max_input_examples=self.args.max_input_examples,
            #     exp_output_dir=exp_output_dir,
            #     experiment_name=self.args.experiment
            # )

            self._build_examples(
                split="dev",
                dataset_dir=self.dataset_dir,
                exp_output_dir=exp_output_dir,
                max_input_examples=None,
                output_arbitrary_targets_for_infeasible_examples=True
            )

            self._build_examples(
                split="train",
                dataset_dir=self.dataset_dir,
                exp_output_dir=exp_output_dir,
                max_input_examples=self.args.max_input_examples,
                output_arbitrary_targets_for_infeasible_examples=False
            )

    def setup(self, stage):
        exp_output_dir = os.path.join(self.args.output_dir,
            self.args.experiment,
            str(self.args.vocab_size),
        )

        self.dataset = load_dataset('json', 
            data_files={'dev': os.path.join(exp_output_dir, "dev.json"), 
                        'train': os.path.join(exp_output_dir, "train.json")},
            field='data')


        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self._convert_to_features,
                batched=True,
                remove_columns=['labels'],
            )
            self.dataset[split].set_format(type="torch",
                columns=["attention_mask", "token_type_ids", "input_ids", "labels"])


    def _pad_sequence(self, batch):
        batch_collated = {}

        for key in ["labels", "input_ids", "attention_mask", "token_type_ids"]:
            elems = [x[key] for x in batch]
            # y_lens = [len(y) for y in yy]

            elems_pad = pad_sequence(elems, batch_first=True, padding_value=0)
            batch_collated[key] = elems_pad
            # batch_collated[key]

        return batch_collated

    def train_dataloader(self):
        return DataLoader(self.dataset['train'], batch_size=self.args.batch_size, num_workers=self.args.max_threads, collate_fn=self._pad_sequence)
    

    def val_dataloader(self):
        return DataLoader(self.dataset['dev'], batch_size=self.args.batch_size, num_workers=self.args.max_threads, collate_fn=self._pad_sequence)

    def test_dataloader(self):
        return DataLoader(self.dataset['test'], batch_size=self.args.batch_size, num_workers=self.args.max_threads, collate_fn=self._pad_sequence)



    def _convert_to_features(self, example_batch, indices=None):
        tokens_batch = [sent.split() for sent in example_batch["text"]]

        # encode the pre-tokenized text
        features = self.tokenizer.batch_encode_plus(
            tokens_batch,
            add_special_tokens=True,
            max_length=self.args.max_length,
            # padding='longest',
            truncation=True,
            # return_tensors='pt',
            is_split_into_words=True
        )
        features['labels'] = self._align_labels_with_tokens(features, example_batch['labels'])

        return features


    def _align_labels_with_tokens(self, features, labels):
        aligned_labels_batch = []

        for b in range(len(labels)):
            # TODO mask?
            default_label = self.label_map["KEEP"]

            aligned_labels = list(map(lambda l: default_label if l is None else labels[b][l], 
                features.words(b)))
            aligned_labels_batch.append(aligned_labels)

        return aligned_labels_batch


    def _phrase_vocabulary_optimization(self, dataset_dir, vocab_size, max_input_examples, exp_output_dir, experiment_name):
        flags = FLAGS()

        flags.input_file = os.path.join(dataset_dir, "train")
        flags.input_format = "fuse"
        flags.vocabulary_size = vocab_size
        flags.max_input_examples = max_input_examples
        flags.output_file = os.path.join(exp_output_dir, "label_map.txt")
        flags.enable_swap_tag = False
        flags.num_extra_statistics = 100

        logger.info("Beginning phrase vocabulary optimization...")
        phrase_vocabulary_optimization.main(flags)
        logger.info("Phrase vocabulary optimization finished.")


    def _build_examples(self, split, dataset_dir, exp_output_dir, max_input_examples, output_arbitrary_targets_for_infeasible_examples):
        label_map_file = os.path.join(exp_output_dir, "label_map.txt")
        input_file = os.path.join(dataset_dir, split)
        output_file = os.path.join(exp_output_dir, f"{split}.json")

        self.label_map = utils.read_label_map(label_map_file)
        self.converter = tagging_converter.TaggingConverter(
        tagging_converter.get_phrase_vocabulary_from_label_map(self.label_map))

        examples = []

        with open(f"{input_file}.in") as f_in, open(f"{input_file}.ref") as f_ref:
            for i, (source, target) in enumerate(zip(f_in, f_ref)):
                source = source.rstrip('\n')
                target = target.rstrip('\n')

                labels = self._convert_to_labels(source, target, 
                    output_arbitrary_targets_for_infeasible_examples)

                if not labels:
                    continue

                # tokens = self._truncate_list(tokens)
                # labels = self._truncate_list(labels)

                # tokens = self.tokenizer.encode_pls
                # # TODO no manual prepend of flags
                # input_tokens = ['[CLS]'] + tokens + ['[SEP]']
                # labels_mask = [0] + [1] * len(labels) + [0]
                # labels = [0] + labels + [0]
                # input_mask = [1] * len(input_tokens)
                # segment_ids = [0] * len(input_tokens)

                # example = {
                #     "input_tokens" : input_tokens,
                #     "labels_mask" : labels_mask,
                #     "labels" : labels,
                #     "input_mask" : input_mask,
                #     "segment_ids" : segment_ids
                # }

                example = {
                    "text" : source,
                    "labels" : labels
                }
                examples.append(example)

                if max_input_examples is not None and i > max_input_examples:
                    break

        with open(output_file, 'w') as f_out:
            json.dump({"data": examples}, f_out)


    def _convert_to_labels(self, source, target, output_arbitrary_targets_for_infeasible_examples):
        task = tagging.EditingTask([source])
        if target is not None:
          tags = self.converter.compute_tags(task, target)
          if not tags:
            if output_arbitrary_targets_for_infeasible_examples:
              # Create a tag sequence [KEEP, DELETE, KEEP, DELETE, ...] which is
              # unlikely to be predicted by chance.
              tags = [tagging.Tag('KEEP') if i % 2 == 0 else tagging.Tag('DELETE')
                      for i, _ in enumerate(task.source_tokens)]
            else:
              return None
        else:
          # If target is not provided, we set all target labels to KEEP.
          tags = [tagging.Tag('KEEP') for _ in task.source_tokens]

        labels = [self.label_map[str(tag)] for tag in tags]
        # labels = [str(tag) for tag in tags]

        return labels

    # def _truncate_list(self, x):
    #   """Returns truncated version of x according to the self._max_seq_length."""
    #   # Save two slots for the first [CLS] token and the last [SEP] token.
    #   return x[:self._max_seq_length - 2]



    

class LaserTagger(pl.LightningModule, FuseModel):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = BertForTokenClassification.from_pretrained('bert-base-uncased', 
            return_dict=True, 
            num_labels=self.args.vocab_size*2 + 2) # KEEP / DELETE for each token + standalone


    def forward(self, **inputs):
        return self.model(**inputs)


    def training_step(self, batch, batch_idx):
        labels = batch["labels"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        outputs = self.model(
                input_ids,
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

        outputs = self(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        val_loss, logits = outputs["loss"], outputs["logits"]
        preds = torch.argmax(logits, axis=1)

        return {'loss': val_loss, "preds": preds, "labels": labels}


    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), 
            lr=self.args.learning_rate, 
            eps=self.args.adam_epsilon,
            betas=(self.args.adam_beta1, self.args.adam_beta2))

        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=self.args.num_train_steps * self.args.warmup_proportion,
            num_training_steps=self.args.num_train_steps
        )
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", default=2e-5, type=float)
        parser.add_argument("--adam_epsilon", default=1e-9, type=float)
        parser.add_argument("--adam_beta1", default=0.9, type=float)
        parser.add_argument("--adam_beta2", default=0.997, type=float)
        parser.add_argument("--warmup_proportion", default=0.1, type=float)
        parser.add_argument("--label_smoothing", default=0.1, type=float)

        return parser
