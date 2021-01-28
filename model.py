#!/usr/bin/env python3

import numpy as np
import os
import logging
import re
import shutil

from collections import defaultdict

from lasertagger import phrase_vocabulary_optimization
from lasertagger import preprocess_main

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


# class LTBert(LightningModule):
#     def __init__(self):
#         super().__init__()
#         self.bert = BertModel.from_pretrained('bert-base-cased', output_attentions=True)
#         self.W = nn.Linear(bert.config.hidden_size, 3)
#         self.num_classes = 3


#     def forward(self, input_ids, attention_mask, token_type_ids):
#         h, _, attn = self.bert(input_ids=input_ids,
#                          attention_mask=attention_mask,
#                          token_type_ids=token_type_ids)
#         h_cls = h[:, 0]
#         logits = self.W(h_cls)
#         return logits, attn



class LaserTagger(FuseModel):
    def __init__(self):
        super().__init__()

    def train(self, train_args, dataset_dir):
        """
        The training pipeline for LT:
        1. phrase vocabulary optimization (extracting phrases used as a vocabulary)
        2. converting text to tags (model is trained directly on KEEP, ADD and DELETE tags)
        3. training
        """

        exp_output_dir = os.path.join(train_args.output_dir,
            train_args.experiment,
            str(train_args.vocab_size)
        )
        os.makedirs(exp_output_dir, exist_ok=True)

        if not (train_args.train_only or train_args.export_only):
            self._phrase_vocabulary_optimization(
                dataset_dir=dataset_dir,
                vocab_size=train_args.vocab_size,
                max_input_examples=train_args.max_input_examples,
                exp_output_dir=exp_output_dir,
                experiment_name=train_args.experiment
            )

            self._convert_text_to_tags(
                dataset_dir=dataset_dir,
                exp_output_dir=exp_output_dir,
                bert_base_dir=train_args.bert_base_dir,
                max_input_examples=train_args.max_input_examples
            )


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


    def _convert_text_to_tags(self, dataset_dir, exp_output_dir, bert_base_dir, max_input_examples):
        # ---- preprocess dev set ----
        flags = FLAGS()

        flags.input_file = os.path.join(dataset_dir, "dev")
        flags.input_format = "fuse"
        flags.output_tfrecord = os.path.join(exp_output_dir, "dev.tf_record")
        flags.label_map_file = os.path.join(exp_output_dir, "label_map.txt")
        flags.vocab_file = os.path.join(bert_base_dir, "vocab.txt")
        flags.max_input_examples = max_input_examples
        # Set this to True when preprocessing the development set.
        flags.output_arbitrary_targets_for_infeasible_examples = True
        # Whether to enable the SWAP tag.
        flags.enable_swap_tag = False
         # Maximum sequence length.
        flags.max_seq_length = 128
        # Whether to lower case the input text. Should be True for uncased models and False for cased models.
        flags.do_lower_case = False
        preprocess_main.main(flags)


        # ---- preprocess train set (reuse the flags) ----
        flags.input_file = os.path.join(dataset_dir, "train")
        flags.output_tfrecord = os.path.join(exp_output_dir, "train.tf_record")
        flags.output_arbitrary_targets_for_infeasible_examples = False
        preprocess_main.main(flags)