#!/usr/bin/env python3

# from model_tf import LaserTaggerTF
from model import LaserTagger

import logging
import argparse
import os

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
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
    parser.add_argument("--experiment", type=str, required=True,
        help="Experiment name used for naming the experiment directory")
    parser.add_argument("--max_input_examples", type=int, default=1000000,
        help="Maximum number of training examples to preprocess")
    parser.add_argument("--num_train_steps", type=int, default=10000,
        help="Number of training steps (set e.g. to 100 for testing)")
    parser.add_argument("--bert_base_dir", type=str, default="lasertagger_tf/bert/cased_L-12_H-768_A-12",
        help="Base directory with the BERT pretrained model")
    parser.add_argument("--train_only", action="store_true",
        help="Skip phrase vocabulary optimization, converting text to tags and exporting the model")
    parser.add_argument("--export_only", action="store_true",
        help="Skip phrase vocabulary optimization, converting text to tags and training the model")
    args = parser.parse_args()


    try:
        dataset_mod = __import__("datasets", fromlist=[args.dataset])
        dataset_cls = getattr(dataset_mod, args.dataset)
        dataset = dataset_cls()
    except AttributeError:
        logger.error(f"Unknown dataset: '{args.dataset}'. Please create a class '{args.dataset}' in 'datasets.py'.")
        exit()


    dataset_dir = os.path.join("data",
                               dataset.name,
                               args.mode)


    logger.info("Initializing LaserTagger")
    # model = LaserTaggerTF()
    model = LaserTagger()

    model.train(args, dataset_dir)