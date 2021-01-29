#!/usr/bin/env bash

CONFIG_NAME=${1:-"gru_with_amounts"}
CLF_TYPE=${2:-"substitute"}
DISCRETIZER_NAME=${3:-"100_quantile"}
DATASET_NAME=${4:-"age"}

CLF_TRAIN_DATA_PATH=../data/${DATASET_NAME}/${CLF_TYPE}_clf/train.jsonl \
CLF_VALID_DATA_PATH=../data/${DATASET_NAME}/${CLF_TYPE}_clf/valid.jsonl \
CLF_TEST_DATA_PATH=../data/${DATASET_NAME}/test.jsonl \
DISCRETIZER_PATH=./presets/${DATASET_NAME}/discretizers/${DISCRETIZER_NAME} \
VOCAB_PATH=./presets/${DATASET_NAME}/vocabs/${DISCRETIZER_NAME} \
RANDOM_SEED=0 \
allennlp train ./configs/classifiers/${CONFIG_NAME}.jsonnet \
--serialization-dir ../experiments/trained_models/${DATASET_NAME}/${CLF_TYPE}_clf/${CONFIG_NAME} \
--include-package advsber