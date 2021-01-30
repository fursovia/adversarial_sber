#!/usr/bin/env bash

CONFIG_NAME=${1:-"bert_with_amounts"}
DISCRETIZER_NAME=${2:-"100_quantile"}
DATASET_NAME=${3:-"age"}

rm -rf ../experiments/trained_models/${DATASET_NAME}/lm/${CONFIG_NAME}

LM_TRAIN_DATA_PATH=../data/${DATASET_NAME}/lm/train.jsonl \
    LM_VALID_DATA_PATH=../data/${DATASET_NAME}/lm/valid.jsonl \
    DISCRETIZER_PATH=./presets/${DATASET_NAME}/discretizers/${DISCRETIZER_NAME} \
    VOCAB_PATH=./presets/${DATASET_NAME}/vocabs/${DISCRETIZER_NAME} \
    allennlp train ./configs/language_models/${CONFIG_NAME}.jsonnet \
        --serialization-dir ../experiments/trained_models/${DATASET_NAME}/lm/${CONFIG_NAME} \
        --include-package advsber