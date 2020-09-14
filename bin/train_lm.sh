#!/usr/bin/env bash

# usage
# bash bin/train_lm.sh bert_with_amounts /notebook/morozov/data/age

CONFIG_NAME=$1
DATA_DIR=$2
DISCRETIZER_NAME=${3:-"discretizer_100_quantile"}
DATASET_NAME=$(basename ${DATA_DIR})
DATE=$(date +%H%M%S-%d%m)
EXP_NAME=${DATE}-${CONFIG_NAME}

DISCRETIZER_PATH=./presets/${DATASET_NAME}/${DISCRETIZER_NAME} \
    LM_TRAIN_DATA_PATH=${DATA_DIR}/lm/train.jsonl \
    LM_VALID_DATA_PATH=${DATA_DIR}/lm/valid.jsonl \
    VOCAB_PATH=./presets/${DATASET_NAME}/${DISCRETIZER_NAME}_vocabulary \
    allennlp train ./configs/language_models/${CONFIG_NAME}.jsonnet \
    --serialization-dir ./logs/${DATASET_NAME}/lm/${EXP_NAME} \
    --include-package advsber
