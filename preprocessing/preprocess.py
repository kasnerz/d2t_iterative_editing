#!/usr/bin/env python3

import os
import argparse
import logging

from utils.sentence_scorer import SentenceScorer
from utils.tokenizer import Tokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


class Preprocessor:
    def __init__(self, dataset, mode, delexicalize=False, ss_device="cuda", reduce_mode="gmean"):
        self.dataset = dataset
        self.mode = mode
        self.tokenizer = Tokenizer()

        if self.mode != "full":
            self.sentence_scorer = SentenceScorer(device=ss_device, reduce_mode=reduce_mode)

    def extract_incremental(self, splits, output_path):
        """
        Extracts incremental examples from the dataset as training data
        for the sentence fusion model.
        """
        logger.info(f"Extracting incremental examples (mode={self.mode}).")

        for split in splits:
            entry_list = self.dataset.data[split]
            lengths, idxs = self.dataset.sort_by_lengths(entry_list)
            entries_out = []
            prev_n = -1
            idxs_pos = 1

            for entry in entry_list:
                triples = entry.triples
                n = len(triples)

                if n > prev_n:
                    # next size
                    logger.info(f"Processing {n}-tuples")
                    beg = idxs[idxs_pos] if idxs_pos < len(idxs) else None
                    end = idxs[idxs_pos+1] if idxs_pos+1 < len(idxs) else None
                    prev_n = n
                    idxs_pos += 1

                # corrupted items
                if self.dataset.name == "e2e" and n < 2:
                    continue

                # extract all incremental examples for the current entry
                entries_out += self._extract(entry_list, entry, n, lengths, beg, end)

            self._write(output_path, split, entries_out)


    def _extract(self, entry_list, entry, n, lengths, beg, end):
        """
        Extracts incremental examples for a single entry
        """
        if n+1 not in lengths:
            return []

        entry_out_all = []
        triples = entry.triples

        for entry_p1 in entry_list[beg:end]:
            triples_p1 = entry_p1.triples

            assert len(triples) + 1 == len(triples_p1), \
                f"l1: {len(triples)}, l2: {len(triples_p1)}"

            if not self._is_incremental(triples, triples_p1):
                continue

            triple = [x for x in triples_p1 if x not in triples]
            text_list = [lex["target_txt"] for lex in entry.lexs]
            ref_list = [lex_p1["target_txt"] for lex_p1 in entry_p1.lexs]

            entry_out = {
                "text_list" : text_list,
                "data" : triple,
                "ref_list" : ref_list,
            }
            entry_out_all.append(entry_out)

        return entry_out_all


    def _is_incremental(self, triples, triples_p1):
        """Checks if `triples_p1` (length n+1) contains all the triples from `triples` (length n)"""
        return all(x in triples_p1 for x in triples)


    def _write(self, out_dir, split, entryset_out):
        out_path = os.path.join(out_dir, self.mode)
        os.makedirs(out_path, exist_ok=True)

        f_in = open(os.path.join(out_path, f"{split}.in"), "w")
        f_ref = open(os.path.join(out_path, f"{split}.ref"), "w")

        samples_processed = 0

        for entry in entryset_out:
            pairs = self._get_lex_pair(entry)

            for inp, ref in pairs:
                f_in.write(inp + "\n")
                f_ref.write(ref + "\n")

                samples_processed += 1
                if samples_processed % 1000 == 0:
                    logger.info(f"{samples_processed} samples processed")

        logger.info(f"{samples_processed} samples extracted.")


    def _fill_template(self, template, triple):
        template = template.replace("<subject>", triple.subj) \
                            .replace("<predicate>", triple.pred) \
                            .replace("<object>", triple.obj)
        return template


    def _get_lex_pair(self, entry):
        inp_sents = []

        assert len(entry["data"]) == 1
        triple = entry["data"][0]

        text_list = [self.tokenizer.tokenize(el) for el in entry["text_list"]]
        ref_list = [self.tokenizer.tokenize(el) for el in entry["ref_list"]]

        templates = self.dataset.get_templates(triple)
        templates = [self._fill_template(template, triple) for template in templates]
        templates = [self.tokenizer.tokenize(template) for template in templates]

        pairs = []

        if self.mode == "best":
            text = self.sentence_scorer.select_best(text_list)
            template = self.sentence_scorer.select_best(templates)
            ref = self.sentence_scorer.select_best(ref_list)

            inp = " ".join([text, template])
            pairs.append((inp, ref))

        elif self.mode == "best_tgt":
            ref = self.sentence_scorer.select_best(ref_list)

            for text in text_list:
                for template in templates:
                    inp = " ".join([text, template])
                    pairs.append((inp, ref))

        elif self.mode == "full":
            for text in text_list:
                for template in templates:
                    inp = " ".join([text, template])

                    for ref in ref_list:
                        pairs.append((inp, ref))
        else:
            raise ValueError("Unknown mode (available: 'best', 'best_tgt', 'full')")

        return pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
        help="Dataset name.")
    parser.add_argument("--input", type=str, required=True,
        help="Path to the dataset")
    parser.add_argument("--mode", type=str, required=True,
        help="Preprocess mode ('best', 'best_tgt', 'full')")
    parser.add_argument("--templates", type=str, required=False,
        help="Path to a JSON file with templates for the dataset")
    parser.add_argument("--output_path", type=str, required=True,
        help="Path where to store the incremental examples")
    parser.add_argument("--device", type=str, default="cuda",
        help="Device to run the sentence scorer on ('cpu' or 'cuda').")
    parser.add_argument('--splits', type=str, nargs='+', default=["train", "dev", "test"],
                    help='Dataset splits (e.g. train dev test)')
    args = parser.parse_args()

    # Load dataset class
    try:
        dataset_mod = __import__("datasets.dataset", fromlist=[args.dataset])
        dataset_cls = getattr(dataset_mod, args.dataset)
        dataset = dataset_cls()
    except AttributeError:
        logger.error(f"Unknown dataset: '{args.dataset}'. Please create class '{args.dataset}' in 'datasets/dataset.py'.")
        exit()

    # Load data
    logger.info(f"Loading dataset {args.dataset}")
    try:
        dataset.load_from_dir(path=args.input, splits=args.splits)
    except FileNotFoundError:
        logger.error(f"Dataset not found in {args.input}")
        exit()

    # Create output directory
    try:
        out_dirname = os.path.join(args.output_path, dataset.name)
        os.makedirs(out_dirname, exist_ok=True)
    except OSError:
        logger.error(f"Output directory {out_dirname} can not be created")
        exit()

    # Load or extract templates
    if args.templates:
        dataset.load_templates(args.templates)
        logger.info(f"Loaded templates from {args.templates}")
    else:
        templates_filename = os.path.join(out_dirname, "templates.json")

        if os.path.isfile(templates_filename):
            dataset.load_templates(templates_filename)
            logger.info(f"Loaded existing templates from {templates_filename}")
        else:
            logger.info("JSON file with templates not specified, templates will be extracted from the training data")
            dataset.extract_templates(templates_filename)


    # Extract incremental examples
    logger.info(f"Extracting incremental examples for the dataset {args.dataset}")

    preprocessor = Preprocessor(dataset=dataset, mode=args.mode)
    preprocessor.extract_incremental(splits=args.splits, output_path=out_dirname)
