#!/usr/bin/env bash

SUBSTITUTE_CONFIG_NAME="gru_with_amounts"
TARGET_CONFIG_NAME="lstm_with_amounts"

NUM_SAMPLES=${1:-"500"}
DATA_DIR=${2:-"/notebook/fursov/adversarial_sber/datasets"}

CONFIG_DIR="configs/attackers"
PRESETS_DIR="./presets"
RESULTS_DIR="./results"
DATE=$(date +%H%M%S-%d%m)


for dataset_name in "age" "gender"; do
    CURR_DATA_PATH=${DATA_DIR}/${dataset_name}/test.jsonl

    for config_path in ${CONFIG_DIR}/*.jsonnet; do
        attack_name=$(basename ${config_path})
        attack_name="${attack_name%.*}"

        RESULTS_PATH=${RESULTS_DIR}/${DATE}/${dataset_name}/${TARGET_CONFIG_NAME}_via_${SUBSTITUTE_CONFIG_NAME}/${attack_name}
        mkdir -p ${RESULTS_PATH}

        DATA_PATH=${CURR_DATA_PATH} \
            OUTPUT_PATH=${RESULTS_PATH}/output.json \
            MASKED_LM_PATH=${PRESETS_DIR}/${dataset_name}/models/lm/bert_with_amounts.tar.gz \
            CLF_PATH=${PRESETS_DIR}/${dataset_name}/models/substitute_clf/${SUBSTITUTE_CONFIG_NAME}.tar.gz \
            PYTHONPATH=. python advsber/commands/attack.py ${config_path} --samples ${NUM_SAMPLES}

        PYTHONPATH=. python advsber/commands/evaluate.py ${RESULTS_PATH}/output.json \
            --save-to=${RESULTS_PATH}/metrics.json \
            --target-clf-path=${PRESETS_DIR}/${dataset_name}/models/target_clf/${TARGET_CONFIG_NAME}.tar.gz
    done
done
