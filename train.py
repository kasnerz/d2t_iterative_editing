#!/usr/bin/env python3

from lasertagger_tf import LaserTaggerTF

import logging
import argparse

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True,
        help="Dataset name (webnlg / e2e / ...).")
    parser.add_argument("--mode", type=str, required=True,
        help="Training mode ('best', 'best_tgt', 'full')")
    parser.add_argument("--vocab_size", type=int, required=True,
        help="Phrase vocabulary size")
    parser.add_argument("--batch_size", type=int, default=32,
        help="Batch size for finetuning the model")
    parser.add_argument("--learning_rate", type=int, default=2e-5,
        help="Learning rate for finetuning the model")
    parser.add_argument("--output_dir", type=str, default="experiments",
        help="Output directory")
    parser.add_argument("--experiment_name", type=str, default=None,
        help="Experiment name (created from the dataset_name and mode if not specified)")
    parser.add_argument("--max_input_examples", type=int, default=1000000,
        help="Limit for the number of training examples")
    parser.add_argument("--bert_base_dir", type=str, default="lasertagger/bert/cased_L-12_H-768_A-12",
        help="Base directory with the BERT pretrained model")
    parser.add_argument("--skip_phrase_vocab_opt", action="store_true",
        help="Skip phrase vocabulary optimization (use if the vocabulary is already optimized)")
    parser.add_argument("--skip_convert_text_to_tags", action="store_true",
        help="Skip converting text to tags (use if the text is already converted)")
    args = parser.parse_args()

    logger.info("Initializing LaserTagger")
    model = LaserTaggerTF()

    if args.experiment_name is None:
        args.experiment_name = f"{args.dataset_name}_{args.mode}"

    # train_args = {
    #     "dataset_name" : args.dataset_name,
    #     "mode" : args.mode,
    #     "vocab_size" : args.vocab_size,
    #     "max_input_examples" : args.max_input_examples,
    #     "output_dir" : args.output_dir,
    #     "experiment_name" : args.experiment_name,
    #     "bert_base_dir" : args.bert_base_dir,
    #     "skip_phrase_vocab_opt" : args.skip_phrase_vocab_opt,
    #     "skip_convert_text_to_tags" : args.skip_convert_text_to_tags
    # }
    model.train(args)