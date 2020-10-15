#!/usr/bin/env bash

# datasets
# target, substitute
# architectures?

DATA_DIR=${1:-"/notebook/fursov/adversarial_sber/datasets"}
CONFIG_DIR="configs/classifiers"
DISCRETIZER_NAME="100_quantile"

LM_CONFIG="bert_with_amounts"


for dataset_name in "age" "gender"; do

    bash bin/train_lm.sh ${LM_CONFIG} ${DATA_DIR}/${dataset_name} ${DISCRETIZER_NAME}

    for clf_type in "substitute" "target"; do
        for config_name in "gru_with_amounts" "lstm_with_amounts"; do
            bash bin/train_clf.sh ${config_name} ${DATA_DIR}/${dataset_name} ${clf_type} ${DISCRETIZER_NAME}
        done
    done
done
