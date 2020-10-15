#!/usr/bin/env bash

NUM_SAMPLES=${1:-"100"}
DATA_DIR=${2:-"/notebook/fursov/adversarial_sber/datasets"}
CONFIG_DIR="configs/attackers"

PRESETS_DIR="./presets"
RESULTS_DIR="./results"
DATE=$(date +%H%M%S-%d%m)

for dataset_name in "age" "gender"; do
    CURR_DATA_PATH=${DATA_DIR}/${dataset_name}/test.jsonl

    # "substitute" "target"
    for clf_type in "substitute"; do
        # "gru_with_amounts" "lstm_with_amounts"
        for clf_config_name in "gru_with_amounts"; do
            for config_path in ${CONFIG_DIR}/*.jsonnet; do
                attack_name=$(basename ${config_path})
                attack_name="${attack_name%.*}"

                RESULTS_PATH=${RESULTS_DIR}/${dataset_name}/${clf_type}/${clf_config_name}/${attack_name}/${DATE}
                mkdir -p ${RESULTS_PATH}

                DATA_PATH=${CURR_DATA_PATH} \
                    OUTPUT_PATH=${RESULTS_PATH}/output.json \
                    MASKED_LM_PATH=${PRESETS_DIR}/${dataset_name}/models/lm/bert_with_amounts.tar.gz \
                    CLF_PATH=${PRESETS_DIR}/${dataset_name}/models/${clf_type}_clf/${clf_config_name}.tar.gz \
                    PYTHONPATH=. python advsber/commands/attack.py ${config_path} --samples ${NUM_SAMPLES}
            done
        done
    done
done
