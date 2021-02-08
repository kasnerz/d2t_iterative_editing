#!/usr/bin/env python

import argparse
import logging
import numpy as np
import os
import re
import tensorflow as tf
import torch

from utils.sentence_scorer import SentenceScorer
from utils.tokenizer import Tokenizer
from model_tf import LaserTaggerTF


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

class IncrementalDecoder:
    def __init__(self, fuse_model, lms_device, reduce_mode, export_file_handle=None, use_e2e_double_templates=True):
        self.fuse_model = fuse_model
        self.sentence_scorer = SentenceScorer(device=lms_device, reduce_mode=reduce_mode)
        self.export_file_handle = export_file_handle
        self.tokenizer = Tokenizer()
        self.use_e2e_double_templates = use_e2e_double_templates


    def _fill_template(self, template, triple):
        template = template.replace("<subject>", self.tokenizer.tokenize(triple.subj)) \
                            .replace("<predicate>", self.tokenizer.tokenize(triple.pred)) \
                            .replace("<object>", self.tokenizer.tokenize(triple.obj))
        return template


    def _triple_to_template(self, dataset, triple):
        templates = dataset.get_templates(triple)

        sentences = []

        for template in templates:
            sentence = self._fill_template(template, triple)
            sentences.append(sentence)

        return self.sentence_scorer.select_best(sentences)


    def fill_template_double(self, template, sorted_triples):
        subj = sorted_triples[0].subj
        obj1 = sorted_triples[0].obj
        obj2 = sorted_triples[1].obj


        template = template.replace("<subject>", self.tokenizer.tokenize(subj)) \
                            .replace("<object1>", self.tokenizer.tokenize(obj1)) \
                            .replace("<object2>", self.tokenizer.tokenize(obj2))
        return template


    def _triple_to_template_double(self, dataset, triples):
        # key for double templates is sorted alphabetically
        triples = sorted(triples, key=lambda t:t.pred)

        res = dataset.select_triples_for_double_template(triples)

        if not res:
            logger.info("Cannot find double template, using a single fallback")
            return None, [], triples

        templates, idx1, idx2 = res

        # triples for which a double template will be used
        double_triples = [triples[idx1], triples[idx2]]

        # remaining triples
        additional_triples = [triple for i, triple in enumerate(triples) if i != idx1 and i != idx2]
        sentences = []

        for template in templates:
            sentence = self.fill_template_double(template, double_triples)
            sentences.append(sentence)

        return self.sentence_scorer.select_best(sentences), double_triples, additional_triples



    def _filter_beam(self, beam, triples, dataset):
        beam_ok = []

        for sent in beam:
            if dataset.check_facts(sent, triples, self.tokenizer):
                beam_ok.append(sent)

        return beam_ok


    def _decode_entry(self, dataset, entry):
        triples = entry.triples

        facts = []
        logger.info(triples)
        logger.info(f"Step #0")

        current_text = None

        if dataset.name == "e2e" and self.use_e2e_double_templates and len(triples) > 1:
            current_text, used_triples, additional_triples = self._triple_to_template_double(dataset, triples)

        if not current_text:
            current_text = self._triple_to_template(dataset, triples[0])
            used_triples = [triples[0]]
            additional_triples = triples[1:]

        logger.info(f"{current_text}")
        facts += used_triples

        for step, triple in enumerate(additional_triples):
            logger.info(f"Step #{step+1}")

            template = self._triple_to_template(dataset, triple)
            facts.append(triple)

            beam = self.fuse_model.fuse(current_text, template)
            beam = self._filter_beam(beam, facts, dataset)

            if beam:
                logger.info(f"{len(beam)} sentences left in beam")
                current_text = self.sentence_scorer.select_best(beam)
            else:
                logger.info("Beam empty, using fallback")
                current_text = " ".join([current_text, template])

            logger.info(f"{current_text}")

        if self.export_file_handle:
            self.export_file_handle.write(self.tokenizer.detokenize(current_text) + "\n")

        logger.info("=========================")


    def decode(self, dataset, split):
        for i, entry in enumerate(dataset.data[split]):
            logger.info(f"Example {i}")
            logger.info(f"-------------"
            self._decode_entry(dataset, entry)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", default="experiments", type=str,
        help="Base directory of the experiment.")
    parser.add_argument("--dataset", type=str, required=True,
        help="Dataset class.")
    parser.add_argument("--dataset_dir", type=str, required=True,
        help="Dataset directory.")
    parser.add_argument("--experiment", type=str, required=True,
        help="Experiment name.")
    parser.add_argument('--splits', type=str, nargs='+', default=["dev"],
        help='Splits to load and decode (e.g. dev test)')
    parser.add_argument("--bert_base_dir", type=str, default="lasertagger_tf/bert/cased_L-12_H-768_A-12",
        help="Base directory with the BERT pretrained model")
    parser.add_argument("--max_seq_length", default=128, type=int,
        help="Maximum sequence length.")
    parser.add_argument("--is_uncased", default=False, action='store_true',
        help="Whether to lower case the input text.")
    parser.add_argument("--lms_device", default="cpu", type=str, required=True,
        help="Device for the sentence scorer ('cpu' / 'cuda').")
    parser.add_argument("--vocab_size", type=str, required=True,
        help="Phrase vocabulary size.")
    parser.add_argument("--reduce_mode", type=str, default="gmean",
        help="Reduce mode for the LMScorer.")
    parser.add_argument("--seed", default=42, type=int,
        help="Random seed.")
    parser.add_argument("--max_threads", default=8, type=int,
        help="Maximum number of threads.")
    parser.add_argument("--no_export", action='store_true',
        help="Do not export the output (print only).")
    parser.add_argument("--use_e2e_double_templates", default=False, action='store_true',
        help="For E2E templates, start with templates for two properties simultaneously extracted from the data.")
    args = parser.parse_args()

    logger.info(args)

    torch.manual_seed(args.seed)
    tf.random.set_random_seed(args.seed)
    np.random.seed(args.seed)

    lms_device = 'cuda' if args.lms_device == 'gpu' else 'cpu'

    # Load dataset class
    try:
        dataset_mod = __import__("data", fromlist=[args.dataset])
        dataset_cls = getattr(dataset_mod, args.dataset)
        dataset = dataset_cls()
    except AttributeError:
        logger.error(f"Unknown dataset: '{args.dataset}'. Please create a class '{args.dataset}' in 'data.py'.")
        exit()

    template_dir = os.path.join("data", dataset.name)
    dataset.load_templates(template_dir)

    dataset.load_from_dir(args.dataset_dir, args.splits)

    tf.config.threading.set_inter_op_parallelism_threads(args.max_threads)
    torch.set_num_threads(args.max_threads)

    for split in args.splits:
        if args.no_export:
            export_file_handle = None
        else:
            os.makedirs("out", exist_ok=True)
            out_filename = f"{args.experiment}_{args.vocab_size}_{split}.hyp"
            export_file_handle = open(os.path.join("out", out_filename), "w")

        fuse_model = LaserTaggerTF()

        label_map_file = os.path.join(args.exp_dir, args.experiment, args.vocab_size, "label_map.txt")
        vocab_file = os.path.join(args.bert_base_dir, "vocab.txt")
        model_path = os.path.join(args.exp_dir, args.experiment, args.vocab_size, "models", "export")

        fuse_model.predict(label_map_file=label_map_file, vocab_file=vocab_file, model_path=model_path,
                            is_uncased=args.is_uncased, max_seq_length=args.max_seq_length)


        decoder = IncrementalDecoder(fuse_model, lms_device=lms_device,
            reduce_mode=args.reduce_mode,
            export_file_handle=export_file_handle,
            use_e2e_double_templates=args.use_e2e_double_templates)

        logger.info(f"Processing {args.dataset} {split} dataset...")

        decoder.decode(dataset, split)

        if export_file_handle is not None:
            export_file_handle.close()

    logger.info("Decoding finished.")