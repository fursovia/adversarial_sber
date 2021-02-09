#!/usr/bin/env bash


SUBSTITUTE_MODEL_DIR=${1:-"../Test/adv_train/gender/models/substitute_clf/gru_with_amounts.tar.gz"}
MASKED_LM_DIR=${2:-"../Test/adv_train/gender/models/lm/bert_with_amounts.tar.gz"}
DATA_DIR=${3:-"../../data/gender"}
NUM_SAMPLES=${4:-'3'}
OUTPUT_DIR=${5:-"../Test/adv_train/exps"}
DATASET_TYPE='gender'
TARGET_CONFIG_NAME='lstm_with_amounts'


#create file with sampling fool examples

DATA_PATH=${DATA_DIR}/substitute_clf/train.jsonl \
OUTPUT_PATH=${OUTPUT_DIR}/results_sf.json \
MASKED_LM_PATH=${MASKED_LM_DIR} \
CLF_PATH=${SUBSTITUTE_MODEL_DIR} \
PYTHONPATH=. python advsber/commands/attack.py configs/attackers/sampling_fool.jsonnet \
--samples ${NUM_SAMPLES}

#create file with concat sampling fool examples
DATA_PATH=${DATA_DIR}/substitute_clf/train.jsonl \
OUTPUT_PATH=${OUTPUT_DIR}/results_csf.json \
MASKED_LM_PATH=${MASKED_LM_DIR} \
CLF_PATH=${SUBSTITUTE_MODEL_DIR} \
PYTHONPATH=. python advsber/commands/attack.py configs/attackers/concat_sampling_fool.jsonnet \
--samples ${NUM_SAMPLES}

#create file with fgsm examples
DATA_PATH=${DATA_DIR}/substitute_clf/train.jsonl \
OUTPUT_PATH=${OUTPUT_DIR}/results_fgsm.json \
CLF_PATH=${SUBSTITUTE_MODEL_DIR} \
PYTHONPATH=. python advsber/commands/attack.py configs/attackers/fgsm.jsonnet \
--samples ${NUM_SAMPLES}

#create file with concat fgsm examples
DATA_PATH=${DATA_DIR}/substitute_clf/train.jsonl \
OUTPUT_PATH=${OUTPUT_DIR}/results_cfgsm.json \
CLF_PATH=${SUBSTITUTE_MODEL_DIR} \
PYTHONPATH=. python advsber/commands/attack.py configs/attackers/fgsm.jsonnet \
--samples ${NUM_SAMPLES}

#add sampling fool examples to the train set of target clf
PYTHONPATH=. python advsber/commands/create_dataset.py ${DATA_DIR}/target_clf/train.jsonl ${OUTPUT_DIR}/train_sf.jsonl  ${OUTPUT_DIR}/results_sf.json ${NUM_SAMPLES}

#add concat sampling fool examples to the train set of target clf
PYTHONPATH=. python advsber/commands/create_dataset.py ${OUTPUT_DIR}/train_sf.jsonl ${OUTPUT_DIR}/train_csf.jsonl  ${OUTPUT_DIR}/results_csf.json ${NUM_SAMPLES}

rm -rf ${OUTPUT_DIR}/train_sf.jsonl

#add fgsm examples to the train set of target clf
PYTHONPATH=. python advsber/commands/create_dataset.py ${OUTPUT_DIR}/train_csf.jsonl ${OUTPUT_DIR}/train_fgsm.jsonl  ${OUTPUT_DIR}/results_fgsm.json ${NUM_SAMPLES}

rm -rf ${OUTPUT_DIR}/train_csf.jsonl

#add concat fgsm examples to the train set of target clf
PYTHONPATH=. python advsber/commands/create_dataset.py ${OUTPUT_DIR}/train_fgsm.jsonl ${OUTPUT_DIR}/train_adv.jsonl  ${OUTPUT_DIR}/results_cfgsm.json ${NUM_SAMPLES}

rm -rf ${OUTPUT_DIR}/train_fgsm.jsonl


CLF_TRAIN_DATA_PATH=${OUTPUT_DIR}/train_adv.jsonl    \
CLF_VALID_DATA_PATH=${DATA_DIR}/target_clf/valid.jsonl \
DISCRETIZER_PATH=./presets/${DATASET_TYPE}/discretizers/100_quantile  \
VOCAB_PATH=./presets/${DATASET_TYPE}/vocabs/100_quantile \
allennlp train ./configs/classifiers/${TARGET_CONFIG_NAME}.jsonnet   \
--serialization-dir ${OUTPUT_DIR}/logs \
--include-package advsber