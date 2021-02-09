#!/usr/bin/env bash

rm -rf ../experiments/attacks
mkdir ../experiments/attacks/


for dataset_name in "age" "gender"; do
    mkdir ../experiments/attacks/${dataset_name}/
    for targ_clf in "gru_with_amounts" "lstm_with_amounts" "cnn_with_amounts"; do
        mkdir ../experiments/attacks/${dataset_name}/targ_${targ_clf}
        for subst_clf in "gru_with_amounts" "lstm_with_amounts" "cnn_with_amounts"; do
            mkdir ../experiments/attacks/${dataset_name}/targ_${targ_clf}/subst_${subst_clf}
            bash scripts/local/attack.sh ${subst_clf} ${targ_clf} 500 ${dataset_name}
        done
    done
done
