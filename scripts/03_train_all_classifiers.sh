#!/usr/bin/env bash

mkdir ../experiments
mkdir ../experiments/trained_models/

for dataset_name in "age" "age_short" "age_tinkoff" "rosbank"; do
    mkdir ../experiments/trained_models/${dataset_name}/
    for clf_type in "target" "substitute"; do
        mkdir ../experiments/trained_models/${dataset_name}/${clf_type}_clf
        for config_name in "gru_with_amounts" "lstm_with_amounts" "cnn_with_amounts"; do
            bash scripts/local/train_clf.sh ${config_name} ${clf_type} "100_quantile" ${dataset_name}
            rm -rf ./presets/${dataset_name}/models/${clf_type}_clf/${config_name}.tar.gz
            cp -r ../experiments/trained_models/${dataset_name}/${clf_type}_clf/${config_name}/model.tar.gz ./presets/${dataset_name}/models/${clf_type}_clf/${config_name}.tar.gz
        done
    done
done