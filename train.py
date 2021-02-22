#!/usr/bin/env python3

# from model_tf import LaserTaggerTF
from model import LaserTagger, LTDataModule

import logging
import argparse
import os
import warnings

import pytorch_lightning as pl

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LTDataModule.add_argparse_args(parser)
    parser = LaserTagger.add_model_specific_args(parser)

    parser.add_argument("--model_name", type=str, default="bert-base-cased",
        help="Name of the model from the Huggingface Transformers library.")
    parser.add_argument("--dataset", type=str, required=True,
        help="Dataset name (webnlg / e2e / ...).")
    parser.add_argument("--mode", type=str, required=True,
        help="Training mode ('best', 'best_tgt', 'full')")
    parser.add_argument("--vocab_size", type=int, required=True,
        help="Phrase vocabulary size")
    parser.add_argument("--batch_size", type=int, default=32,
        help="Batch size for finetuning the model")
    parser.add_argument("--output_dir", type=str, default="experiments",
        help="Output directory")
    parser.add_argument("--experiment", type=str, required=True,
        help="Experiment name used for naming the experiment directory")
    parser.add_argument("--max_input_examples", type=int, default=1000000,
        help="Maximum number of training examples to preprocess")
    parser.add_argument("--num_train_steps", type=int, default=10000,
        help="Number of training steps (set e.g. to 100 for testing)")
    parser.add_argument("--max_length", type=int, default=256,
        help="Maximum number of tokens per example")
    parser.add_argument("--train_only", action="store_true",
        help="Skip phrase vocabulary optimization, converting text to tags and exporting the model")
    parser.add_argument("--export_only", action="store_true",
        help="Skip phrase vocabulary optimization, converting text to tags and training the model")
    parser.add_argument("--seed", default=42, type=int,
        help="Random seed.")
    parser.add_argument("--max_threads", default=8, type=int,
        help="Maximum number of CPU threads.")
    parser.add_argument("--accumulate_grad_batches", default=1, type=int,
            help="Number of batches to accumulate before running the backward pass (efficiently multiplies the batch size).")
    
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()
    logger.info("Initializing LaserTagger")

    pl.seed_everything(args.seed)
    dm = LTDataModule(args)
    dm.prepare_data()
    dm.setup('fit')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        model = LaserTagger(args)

        ckpt_output_dir = os.path.join(args.output_dir,
            args.experiment,
            str(args.vocab_size),
            "model.ckpt"
        )

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filepath=ckpt_output_dir,
            save_top_k=1,
            verbose=True,
            monitor='val_loss',
            mode='min'
        )
        trainer = pl.Trainer.from_argparse_args(args)
        trainer.fit(model, dm)
