#!/usr/bin/env bash


DATA_DIR=${1:-"/notebook/fursov/adversarial_sber/datasets"}
RESULTS_DIR="./results_models"
suffix=".tar.gz"

for dataset_name in "age" "gender"; do
    for clf_type in "substitute_clf" "target_clf"; do
        for model_name in ${DATA_DIR}/${dataset_name}/models/${clf_type}/*; do
            name=$(basename ${model_name})
            name="${model_name%"$suffix"}"
            RESULTS_PATH=${RESULTS_DIR}/${dataset_name}/${clf_type}/${name}
            mkdir -p ${RESULTS_PATH}
            allennlp evaluate ${model_name} presets/data/${dataset_name}/test.jsonl --output-file ${RESULTS_PATH}/output_models.json --include-package advsber
        done
        python advsber/commands/aggregate_results_models.py ${RESULTS_DIR}/${dataset_name}/${clf_type}
    done
    python advsber/commands/aggregate_results_models.py ${RESULTS_DIR}/${dataset_name}
done

python advsber/commands/aggregate_results_models.py ${RESULTS_DIR}
