#!/usr/bin/env bash


RESULTS_DIR=./results/sf_exp

for i in $(seq 0 99); do
    config_path=sampling_fool_${i}.json
    mkdir -p ${RESULTS_DIR}/${i}

    CUDA_VISIBLE_DEVICES="1" \
        PYTHONPATH=. python advsber/commands/attack.py configs/attackers/sf_exp/${config_path} \
        --samples 500

    PYTHONPATH=. python advsber/commands/evaluate.py ${RESULTS_DIR}/${i}/output.json \
        --save-to=${RESULTS_DIR}/${i}/metrics.json \
        --target-clf-path="./presets/age/models/target_clf/lstm_with_amounts.tar.gz"

done