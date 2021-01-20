#!/bin/bash

# # WebNLG, E2E
# DATASET="WebNLG"
# DATA_PATH="$PWD/datasets/webnlg/data/v1.4/en/"

# E2E
DATASET="E2E"
DATA_PATH="$PWD/datasets/e2e-cleaning/cleaned-data"

OUTPUT_PATH="$PWD/data"

# "full": all lexicalizations
# "best": best lexicalizations according to LMScorer (source, target)
# "best_tgt": all lexicalizations (source) and best lexicalizations according to LMScorer (target)
MODE=best_tgt

# size of the LaserTagger vocabulary
VOCAB_SIZE=100

# device for the LMScorer: cpu / cuda
# note that an extra GPU is needed in case the sentence fusion model also runs on GPU (which is recommended)
# if mode=full, preprocessing does not use LMScorer and thus the parameter LMS_DEVICE_PREPROCESSING is unused
LMS_DEVICE_PREPROCESSING=cuda
LMS_DEVICE_DECODING=$LMS_DEVICE_PREPROCESSING

mkdir -p "$OUTPUT_PATH"


# TODO LT beam size
python3 preprocess.py \
    --dataset "$DATASET" \
    --input "$DATA_PATH" \
    --mode "$MODE" \
    --output_path "$OUTPUT_PATH" \
    --splits "train" "test" "dev" \
    --lms_device "$LMS_DEVICE_PREPROCESSING"

# python3 train.py \
#     --dataset "$DATASET" \
#     --mode "$MODE" \
#     --vocab_size "$VOCAB_SIZE" \
#     --num_train_steps 100

# python3 decode.py \
#     --dataset "$DATASET" \
#     --experiment webnlg_full \
#     --dataset_dir /lnet/spec/work/people/kasner/NLG/datasets/webnlg/data/v1.4/en/ \
#     --vocab_size 100 \
#     --lms_device $LMS_DEVICE_DECODING \
#     --split dev \
#     --max_threads 8