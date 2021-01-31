#!/usr/bin/env bash

for dataset_name in "age" "gender"; do
    bash scripts/local/train_lm.sh "bert_with_amounts" "100_quantile" ${dataset_name}
    rm -rf ./presets/${dataset_name}/models/lm/bert_with_amounts.tar.gz
    cp -r ../experiments/trained_models/${dataset_name}/lm/bert_with_amounts/model.tar.gz ./presets/${dataset_name}/models/lm/bert_with_amounts.tar.gz
done

