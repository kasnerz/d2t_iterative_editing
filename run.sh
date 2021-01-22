#!/bin/bash


WEBNLG_PATH="./datasets/webnlg/data/v1.4/en/"
E2E_PATH="./datasets/e2e-cleaning/cleaned-data"
DF_PATH="./datasets/discofuse/discofuse_v1/wikipedia"

case $1 in
    "webnlg")
        DATASET_TRAIN="WebNLG"
        DATASET_EVAL="$DATASET_TRAIN"
        DATASET_PATH_TRAIN="$WEBNLG_PATH"
        DATASET_PATH_EVAL="$WEBNLG_PATH"
    ;;
    "e2e")
        DATASET_TRAIN="E2E"
        DATASET_EVAL="$DATASET_TRAIN"
        DATASET_PATH_TRAIN="$E2E_PATH"
        DATASET_PATH_EVAL="$E2E_PATH"
    ;;
    "df-webnlg")
        DATASET_TRAIN="DiscoFuse"
        DATASET_EVAL="WebNLG"
        DATASET_PATH_TRAIN="$DF_PATH"
        DATASET_PATH_EVAL="$WEBNLG_PATH"
    ;;
    "df-e2e")
        DATASET_TRAIN="DiscoFuse"
        DATASET_EVAL="E2E"
        DATASET_PATH_TRAIN="$DF_PATH"
        DATASET_PATH_EVAL="$E2E_PATH"
    ;;
    *)
        echo "Error: Unknown experiment $1"
        exit
    ;;
esac

# "full": all lexicalizations
# "best": best lexicalizations according to LMScorer (source & target)
# "best_tgt": all lexicalizations (source) and best lexicalizations according to LMScorer (target)
MODE=full

# size of the LaserTagger vocabulary
VOCAB_SIZE=100

# device for the LMScorer: cpu / gpu
# note that an extra GPU is needed in case the sentence fusion model also runs on GPU (which is recommended)
# if mode=full, preprocessing does not use LMScorer and thus the parameter LMS_DEVICE_PREPROCESSING is unused
LMS_DEVICE_PREPROCESSING=gpu
LMS_DEVICE_DECODING=cpu

# experiment name (used for naming the experiment directory), e.g. webnlg_full
EXPERIMENT_NAME="${1,,}_${MODE}"

# number of LT finetuning steps
NUM_TRAIN_STEPS=100

# which data split to decode and evaluate (dev / test)
DECODE_AND_EVAL_SPLIT=test



python3 preprocess.py \
    --dataset "$DATASET_TRAIN" \
    --input "$DATASET_PATH_TRAIN" \
    --mode "$MODE" \
    --splits "train" "test" "dev" \
    --lms_device "$LMS_DEVICE_PREPROCESSING"

if [ $DATASET_TRAIN != $DATASET_EVAL ]; then
    python3 preprocess.py \
        --dataset "$DATASET_EVAL" \
        --input "$DATASET_PATH_EVAL" \
        --mode "$MODE" \
        --splits "test" "dev" \
        --lms_device "$LMS_DEVICE_PREPROCESSING"
fi

python3 train.py \
    --dataset "$DATASET_TRAIN" \
    --mode "$MODE" \
    --vocab_size "$VOCAB_SIZE" \
    --experiment "$EXPERIMENT_NAME" \
    --num_train_steps "$NUM_TRAIN_STEPS"

python3 decode.py \
    --dataset "$DATASET_EVAL" \
    --experiment "$EXPERIMENT_NAME" \
    --dataset_dir "$DATASET_PATH_EVAL" \
    --vocab_size "$VOCAB_SIZE" \
    --lms_device "$LMS_DEVICE_DECODING" \
    --split "$DECODE_AND_EVAL_SPLIT" \
    --use_e2e_double_templates

python3 evaluate.py \
    --ref_file "data/${DATASET_EVAL,,}/ref/$DECODE_AND_EVAL_SPLIT.ref" \
    --hyp_file "out/$EXPERIMENT_NAME_$VOCAB_SIZE_$DECODE_AND_EVAL_SPLIT.out" \
    --lowercase