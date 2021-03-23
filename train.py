#!/usr/bin/env python3

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
    parser.add_argument("--max_length", type=int, default=128,
        help="Maximum number of tokens per example")
    parser.add_argument("--train_only", action="store_true",
        help="Skip phrase vocabulary optimization, converting text to tags and exporting the model")
    parser.add_argument("--enable_swap_tag", action="store_true",
        help="Enable LaserTagger SWAP tag for swapping sentences")
    parser.add_argument("--seed", default=42, type=int,
        help="Random seed.")
    parser.add_argument("--max_threads", default=8, type=int,
        help="Maximum number of CPU threads.")
    
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()
    logger.info("Initializing LaserTagger")

    pl.seed_everything(args.seed)
    dm = LTDataModule(args)
    dm.prepare_data()
    dm.setup('fit')

    # disable " UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors."
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        model = LaserTagger(args)

        ckpt_output_dir = os.path.join(args.output_dir,
            args.experiment,
            str(args.vocab_size)
        )
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_output_dir,
            filename='model',
            save_top_k=0,
            save_last=True,
            verbose=True,
            monitor='loss/val',
            mode='min'
        )
        trainer = pl.Trainer.from_argparse_args(args, 
            callbacks=[checkpoint_callback])
        trainer.fit(model, dm)
