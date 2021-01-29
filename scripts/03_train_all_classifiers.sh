#!/usr/bin/env bash

mkdir ../experiments

rm -rf ../experiments/trained_models
mkdir ../experiments/trained_models/


for dataset_name in "gender" "age"; do
    mkdir ../experiments/trained_models/${dataset_name}/
    for clf_type in "target" "substitute"; do
        mkdir ../experiments/trained_models/${dataset_name}/${clf_type}_clf
        for config_name in "gru_with_amounts" "lstm_with_amounts" "cnn_with_amounts"; do
            bash scripts/local/train_clf.sh ${config_name} ${clf_type} "100_quantile" ${dataset_name}
        done
    done
done