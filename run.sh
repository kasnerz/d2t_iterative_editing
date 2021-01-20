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
MODE=full

# size of the LaserTagger vocabulary
VOCAB_SIZE=100

# device for the LMScorer: cpu / gpu
# note that an extra GPU is needed in case the sentence fusion model also runs on GPU (which is recommended)
# if mode=full, preprocessing does not use LMScorer and thus the parameter LMS_DEVICE_PREPROCESSING is unused
LMS_DEVICE_PREPROCESSING=gpu
LMS_DEVICE_DECODING=$USE_LMS_GPU_PREPROCESSING

# experiment name (used for naming the experiment directory), e.g. webnlg_full
EXPERIMENT_NAME="${DATASET,,}_${MODE}"

# TODO increase
NUM_TRAIN_STEPS=100

mkdir -p "$OUTPUT_PATH"


python3 preprocess.py \
    --dataset "$DATASET" \
    --input "$DATA_PATH" \
    --mode "$MODE" \
    --output_path "$OUTPUT_PATH" \
    --splits "train" "test" "dev" \
    --lms_device "$LMS_DEVICE_PREPROCESSING"

python3 train.py \
    --dataset "$DATASET" \
    --mode "$MODE" \
    --vocab_size "$VOCAB_SIZE" \
    --experiment "$EXPERIMENT_NAME" \
    --num_train_steps "$NUM_TRAIN_STEPS"

python3 decode.py \
    --dataset "$DATASET" \
    --experiment "$EXPERIMENT_NAME" \
    --dataset_dir "$DATA_PATH" \
    --vocab_size "$VOCAB_SIZE" \
    --lms_device "$LMS_DEVICE_DECODING" \
    --split dev