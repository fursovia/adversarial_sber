#!/usr/bin/env bash

for dataset_name in "gender" "age"; do
    bash scripts/local/train_lm.sh "bert_with_amounts" "100_quantile" ${dataset_name}
done

