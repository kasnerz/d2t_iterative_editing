#!/usr/bin/env python

import os
import argparse
import contextlib
import logging

import sys
sys.path.insert(0, 'e2e_metrics')
import measure_scores

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def evaluate(args):
    data_src, data_ref, data_sys = measure_scores.load_data(args.ref_file, args.hyp_file, None)

    if args.lowercase:
        data_src = [sent.lower() for sent in data_src]
        data_ref = [[sent.lower() for sent in sent_list] for sent_list in data_ref]
        data_sys = [sent.lower() for sent in data_sys]

    measure_scores.evaluate(data_src, data_ref, data_sys, print_as_table=True, print_table_header=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_file", type=str, required=True,
        help="Dataset to compare the results with.")
    parser.add_argument("--hyp_file", type=str, default=None,
        help="File with output from the model.")
    parser.add_argument("--lowercase", action="store_true", default=False,
        help="Evaluate on lower-cased files.")
    args = parser.parse_args()

    evaluate(args)
