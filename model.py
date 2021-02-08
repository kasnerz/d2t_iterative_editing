#!/usr/bin/env python3

import numpy as np
import os
import logging
import re
import shutil
import json

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from collections import defaultdict
from datasets import load_dataset

from lasertagger import phrase_vocabulary_optimization
from lasertagger import preprocess_main

from lasertagger import tagging

from lasertagger import tagging_converter
from lasertagger import utils

from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
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

    def __init__(self, args, dataset_dir):
        super().__init__()
        self.args = args
        self.dataset_dir = dataset_dir
        
        # TODO can be used directly from args
        self.num_labels = args.vocab_size
        self.train_batch_size = args.batch_size
        self.eval_batch_size = args.batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def prepare_data(self):
        """
        The training pipeline for LT:
        1. extracting phrases used for the vocabulary
        2. converting text to tags
        3. training the model
        """

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
 
            self.dataset[split].set_format(type="torch")


    def train_dataloader(self):
        return DataLoader(self.dataset['train'], batch_size=self.train_batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.dataset['validation'], batch_size=self.eval_batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset['test'], batch_size=self.eval_batch_size)


    def _convert_to_features(self, example_batch, indices=None):
        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            example_batch["text"],
            # max_length=self.args.max_seq_length,
            # pad_to_max_length=True
        )

        # Rename label to labels to make it easier to pass to model forward
        features['labels'] = example_batch['labels']

        return features


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

        label_map = utils.read_label_map(label_map_file)
        self.converter = tagging_converter.TaggingConverter(
        tagging_converter.get_phrase_vocabulary_from_label_map(label_map))

        examples = []

        with open(f"{input_file}.in") as f_in, open(f"{input_file}.ref") as f_ref:
            for i, (source, target) in enumerate(zip(f_in, f_ref)):
                source = source.rstrip('\n')
                target = target.rstrip('\n')

                labels = self._convert_to_labels(source, target, label_map, 
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
            json.dump({"data": examples}, f_out, indent=4)


        

    def _convert_to_labels(self, source, target, label_map, output_arbitrary_targets_for_infeasible_examples):
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

        # labels = [label_map[str(tag)] for tag in tags]
        labels = [str(tag) for tag in tags]

        return labels

    # def _truncate_list(self, x):
    #   """Returns truncated version of x according to the self._max_seq_length."""
    #   # Save two slots for the first [CLS] token and the last [SEP] token.
    #   return x[:self._max_seq_length - 2]



    

class LaserTagger(pl.LightningModule, FuseModel):
    def __init__(self):
        super().__init__()
        # self.bert = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
        # self.W = nn.Linear(bert.config.hidden_size, 3)
        # self.num_classes = 3


    def forward(self, **inputs):
        import pdb; pdb.set_trace()  # breakpoint b7cda139 //
        output = self.model(**inputs)
        return output



    def training_step(self, batch, batch_idx):
        import pdb; pdb.set_trace()  # breakpoint e459cb81 //

        labels = batch["label"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        loss, _ = self.model(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels
                )

        tqdm_dict = {"train_loss": loss}
        output = OrderedDict({
            "loss": loss,
            "progress_bar": tqdm_dict,
            "log": tqdm_dict
            })

        return output


    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, 
            lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]