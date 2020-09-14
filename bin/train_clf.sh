#!/usr/bin/env bash

# usage
# bash bin/train_clf.sh gru_with_amounts /notebook/morozov/data/age substitute

CONFIG_NAME=$1
DATA_DIR=$2
CLF_TYPE=${3:-"substitute"}
DISCRETIZER_NAME=${4:-"discretizer_100_quantile"}
DATASET_NAME=$(basename ${DATA_DIR})
DATE=$(date +%H%M%S-%d%m)
EXP_NAME=${DATE}-${CONFIG_NAME}

DISCRETIZER_PATH=./presets/${DATASET_NAME}/${DISCRETIZER_NAME} \
    CLF_TRAIN_DATA_PATH=${DATA_DIR}/${CLF_TYPE}_clf/train.jsonl \
    CLF_VALID_DATA_PATH=${DATA_DIR}/${CLF_TYPE}_clf/valid.jsonl \
    VOCAB_PATH=./presets/${DATASET_NAME}/${DISCRETIZER_NAME}_vocabulary \
    allennlp train ./configs/classifiers/${CONFIG_NAME}.jsonnet \
    --serialization-dir ./logs/${DATASET_NAME}/${CLF_TYPE}_clf/${EXP_NAME} \
    --include-package advsber \
    -o '{"trainer.cuda_device": -1}'
