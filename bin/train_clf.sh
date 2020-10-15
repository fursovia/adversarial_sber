
#!/usr/bin/env bash

# usage
# bash bin/train_clf.sh gru_with_amounts /notebook/morozov/data/age substitute

CONFIG_NAME=$1
DATA_DIR=$2
CLF_TYPE=${3:-"substitute"}
DISCRETIZER_NAME=${4:-"100_quantile"}

DATASET_NAME=$(basename ${DATA_DIR})
DATE=$(date +%H%M%S-%d%m)
EXP_NAME=${DATE}-${CONFIG_NAME}

CLF_TRAIN_DATA_PATH=${DATA_DIR}/${CLF_TYPE}_clf/train.jsonl \
    CLF_VALID_DATA_PATH=${DATA_DIR}/${CLF_TYPE}_clf/valid.jsonl \
    DISCRETIZER_PATH=./presets/${DATASET_NAME}/discretizers/${DISCRETIZER_NAME} \
    VOCAB_PATH=./presets/${DATASET_NAME}/vocabs/${DISCRETIZER_NAME} \
    allennlp train ./configs/classifiers/${CONFIG_NAME}.jsonnet \
        --serialization-dir ./logs/${DATASET_NAME}/${CLF_TYPE}_clf/${EXP_NAME} \
        --include-package advsber