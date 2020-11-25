#!/usr/bin/env bash


RESULTS_DIR=./results/wer_exp
mkdir -p $RESULTS_DIR


for i in $(seq 1 10); do
    config_path=concat_fgsm_${i}.jsonnet
    filename="${config_path%.*}"
    CUDA_VISIBLE_DEVICES="1" \
        DATA_PATH="./datasets/age/test.jsonl" \
        OUTPUT_PATH=${RESULTS_DIR}/${filename}.json \
        MASKED_LM_PATH="./presets/age/models/lm/bert_with_amounts.tar.gz" \
        CLF_PATH="./presets/age/models/substitute_clf/gru_with_amounts.tar.gz" \
        PYTHONPATH=. python advsber/commands/attack.py configs/attackers/wer_exp/${config_path} \
        --samples 500


    config_path=concat_sampling_fool${i}.jsonnet
    filename="${config_path%.*}"
    CUDA_VISIBLE_DEVICES="1" \
        DATA_PATH="./datasets/age/test.jsonl" \
        OUTPUT_PATH=${RESULTS_DIR}/${filename}.json \
        MASKED_LM_PATH="./presets/age/models/lm/bert_with_amounts.tar.gz" \
        CLF_PATH="./presets/age/models/substitute_clf/gru_with_amounts.tar.gz" \
        PYTHONPATH=. python advsber/commands/attack.py configs/attackers/wer_exp/${config_path} \
        --samples 500


    config_path=greedy_concat_sampling_fool${i}.jsonnet
    filename="${config_path%.*}"
    CUDA_VISIBLE_DEVICES="1" \
        DATA_PATH="./datasets/age/test.jsonl" \
        OUTPUT_PATH=${RESULTS_DIR}/${filename}.json \
        MASKED_LM_PATH="./presets/age/models/lm/bert_with_amounts.tar.gz" \
        CLF_PATH="./presets/age/models/substitute_clf/gru_with_amounts.tar.gz" \
        PYTHONPATH=. python advsber/commands/attack.py configs/attackers/wer_exp/${config_path} \
        --samples 500
done