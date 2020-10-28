#!/usr/bin/env bash

# datasets
# target, substitute
# architectures?

DATA_DIR=${1:-"/notebook/fursov/adversarial_sber/datasets"}
DISCRETIZER_NAME="100_quantile"
CLF_TYPE="lstm_with_amounts_via_gru_with_amounts"
CONFIG_NAME="gru_with_amounts"

DATE=$(date +%H%M%S-%d%m)

for dataset_name in "age" "gender"; do

    for dir in ${DATA_DIR}/${dataset_name}/"adv_training"/*/${CLF_TYPE}; do
        attack_type=$(dirname ${dir})
        attack_type=$(basename ${attack_type})

        CLF_TRAIN_DATA_PATH=${dir}/train.jsonl \
            CLF_VALID_DATA_PATH=${dir}/valid.jsonl \
            DISCRETIZER_PATH=./presets/${dataset_name}/discretizers/${DISCRETIZER_NAME} \
            VOCAB_PATH=./presets/${dataset_name}/vocabs/${DISCRETIZER_NAME} \
            allennlp train ./configs/classifiers/${CONFIG_NAME}.jsonnet \
                --serialization-dir ./logs/${DATE}/${dataset_name}/${attack_type} \
                --include-package advsber
    done
done

