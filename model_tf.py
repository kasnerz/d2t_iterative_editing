#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import os
import logging
import re
import shutil

from collections import defaultdict
from model import FuseModel

from lasertagger_tf import bert_example
from lasertagger_tf import predict_utils
from lasertagger_tf import tagging_converter
from lasertagger_tf import utils
from lasertagger_tf import phrase_vocabulary_optimization
from lasertagger_tf import preprocess_main
from lasertagger_tf import run_lasertagger


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


class LaserTaggerTF(FuseModel):
    """
    A wrapper for the original LT implementation. Supplements the individual scripts
    used in the LT training pipeline.
    """
    def __init__(self):
        super().__init__()


    def fuse(self, first, second):
        sentence = " ".join([first, second])
        return self.predictor.predict([sentence])


    def train(self, train_args):
        """
        The training pipeline for LT:
        1. phrase vocabulary optimization (extracting phrases used as a vocabulary)
        2. converting text to tags (model is trained directly on KEEP, ADD and DELETE tags)
        3. training
        """
        dataset_dir = os.path.join("data",
                                   train_args.dataset_name,
                                   train_args.mode)

        exp_output_dir = os.path.join(train_args.output_dir,
            train_args.experiment_name,
            str(train_args.vocab_size)
        )
        os.makedirs(exp_output_dir, exist_ok=True)

        if not (train_args.train_only or train_args.export_only) :
            self._phrase_vocabulary_optimization(
                dataset_dir=dataset_dir,
                vocab_size=train_args.vocab_size,
                max_input_examples=train_args.max_input_examples,
                exp_output_dir=exp_output_dir,
                experiment_name=train_args.experiment_name
            )
            self._convert_text_to_tags(
                  dataset_dir=dataset_dir,
                  exp_output_dir=exp_output_dir,
                  bert_base_dir=train_args.bert_base_dir,
                  max_input_examples=train_args.max_input_examples
            )
        else:
            logger.info("Skipping phrase vocabulary optimization and converting text to tags...")

        self._train(
            exp_output_dir=exp_output_dir,
            bert_base_dir=train_args.bert_base_dir,
            learning_rate=train_args.learning_rate,
            batch_size=train_args.batch_size,
            num_train_steps=train_args.num_train_steps,
            train_only=train_args.train_only,
            export_only=train_args.export_only
        )


    def predict(self):
        hyperparams = f"b{args.batch_size}l{args.learning_rate}"

        self.label_map_file = os.path.join(args.exp_dir, args.experiment, args.vocab_size, "label_map.txt")
        self.vocab_file = os.path.join(args.bert_dir, "vocab.txt")
        self.model_path = os.path.join(args.exp_dir, args.experiment, args.vocab_size, hyperparams, "models", "export", args.model)

        self.label_map = utils.read_label_map(self.label_map_file)

        self.converter = tagging_converter.TaggingConverter(
          tagging_converter.get_phrase_vocabulary_from_label_map(self.label_map),
          args.enable_swap_tag)

        self.builder = bert_example.BertExampleBuilder(self.label_map, self.vocab_file,
                                                args.max_seq_length,
                                                args.is_uncased, self.converter)

        self.predictor = predict_utils.LaserTaggerPredictor(
          tf.contrib.predictor.from_saved_model(self.model_path), self.builder,
          self.label_map)


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


    def _train(self, exp_output_dir, bert_base_dir, learning_rate, batch_size, num_train_steps, train_only, export_only):
        # ---- run training ----
        flags = FLAGS()

        flags.training_file = os.path.join(exp_output_dir, "train.tf_record")
        flags.label_map_file = os.path.join(exp_output_dir, "label_map.txt")
        flags.model_config_file = os.path.join("lasertagger_tf", "configs", "lasertagger_config.json")
        flags.output_dir = os.path.join(exp_output_dir, "models")
        flags.do_train = True
        flags.do_eval = False
        flags.do_export = False
        flags.init_checkpoint = os.path.join(bert_base_dir, "bert_model.ckpt")
        flags.max_seq_length = 128
        flags.learning_rate = learning_rate
        flags.train_batch_size = batch_size
        flags.predict_batch_size = 8
        flags.eval_batch_size = 8
        flags.save_checkpoints_steps = 5000
        # The maximum amount of time (in seconds) for eval worker to wait between checkpoints.
        flags.eval_timeout = 600
        flags.eval_batch_size = 8
        # Proportion of training to perform linear learning rate warmup for.
        flags.warmup_proportion = 0.1
        flags.num_train_steps = num_train_steps

        # copy default (unused) settings for TPUs
        flags.use_tpu = False
        flags.tpu_name = None
        flags.tpu_zone = None
        flags.gcp_project = None
        flags.iterations_per_loop = 1000
        flags.master = None
        flags.export_path = None

        with open(os.path.join(exp_output_dir, "train.tf_record.num_examples.txt"), "r") as f:
            flags.num_train_examples = int(f.read().strip())

        with open(os.path.join(exp_output_dir, "dev.tf_record.num_examples.txt"), "r") as f:
            flags.num_eval_examples = int(f.read().strip())

        if not export_only:
            run_lasertagger.main(flags)


        if not train_only:
            # ---- export the model (reuse the flags) ----
            flags.do_train = False
            flags.do_export = True
            flags.export_path = os.path.join(flags.output_dir, "export")

            with open(os.path.join(flags.output_dir, "checkpoint"), "r") as f:
                t = f.read()
                export_checkpoint = re.findall(r"model_checkpoint_path: \"([^\s]*)\"", t)[0]

            flags.init_checkpoint = os.path.join(flags.output_dir, export_checkpoint)
            run_lasertagger.main(flags)

            # exported model directory gets timestamped - move the file to the parent directory for consistency
            all_subdirs = [os.path.join(flags.export_path, d) for d in os.listdir(flags.export_path)]

            if len(all_subdirs) > 1:
                latest_subdir = max(all_subdirs, key=os.path.getmtime)
                logger.info(f"Selecting the latest exported checkpoint: {latest_subdir}")
            else:
                latest_subdir = all_subdirs[0]

            shutil.move(os.path.join(latest_subdir, "saved_model.pb"), flags.export_path)
            shutil.move(os.path.join(latest_subdir, "variables"), flags.export_path)
            os.rmdir(latest_subdir)