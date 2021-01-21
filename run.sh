#!/bin/bash

# # WebNLG
# DATASET="WebNLG"
# DATASET_PATH="./datasets/webnlg/data/v1.4/en/"

# # E2E
# DATASET="E2E"
# DATASET_PATH="./datasets/e2e-cleaning/cleaned-data"

# DiscoFuse
DATASET="DiscoFuse"
DATASET_PATH="./datasets/discofuse/discofuse_v1/wikipedia"

# "full": all lexicalizations
# "best": best lexicalizations according to LMScorer (source & target)
# "best_tgt": all lexicalizations (source) and best lexicalizations according to LMScorer (target)
MODE=best

# size of the LaserTagger vocabulary
VOCAB_SIZE=100

# device for the LMScorer: cpu / gpu
# note that an extra GPU is needed in case the sentence fusion model also runs on GPU (which is recommended)
# if mode=full, preprocessing does not use LMScorer and thus the parameter LMS_DEVICE_PREPROCESSING is unused
LMS_DEVICE_PREPROCESSING=gpu
LMS_DEVICE_DECODING=cpu

# experiment name (used for naming the experiment directory), e.g. webnlg_full
EXPERIMENT_NAME="${DATASET,,}_${MODE}"

# number of LT finetuning steps
NUM_TRAIN_STEPS=10000

python3 preprocess.py \
    --dataset "$DATASET" \
    --input "$DATASET_PATH" \
    --mode "$MODE" \
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
    --dataset_dir "$DATASET_PATH" \
    --vocab_size "$VOCAB_SIZE" \
    --lms_device "$LMS_DEVICE_DECODING" \
    --split dev \
    --use_e2e_double_templates