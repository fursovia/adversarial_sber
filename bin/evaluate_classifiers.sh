#!/usr/bin/env bash

# usage
# bash bin/evaluate_classifiers.sh age substitute white 35100


DATASET_NAME=${1:-"gender"}
MODEL_TYPE=${2:-"gru"}
ATTACK_TYPE=${3:-"white"}
SUBST_TARGET=${4:-"35100"}

if [ ! -d ./results ]
then
  mkdir ./results
fi
if [ ! -d ./results/${ATTACK_TYPE} ]
then
  mkdir ./results/${ATTACK_TYPE}
fi
if [ ! -d ./results/${ATTACK_TYPE}/${DATASET_NAME} ]
then
  mkdir ./results/${ATTACK_TYPE}/${DATASET_NAME}
fi
if [ $ATTACK_TYPE == "black" ]
then
  CUDA_VISIBLE_DEVICES=1 \
      DATA_PATH=./presets/${DATASET_NAME}/test.jsonl \
      OUTPUT_PATH=./results/${ATTACK_TYPE}/${DATASET_NAME}/${MODEL_TYPE}_sf_output.json \
      MASKED_LM_PATH=./presets/${DATASET_NAME}/lm.model.tar.gz \
      CLF_TARGET_PATH=./presets/${DATASET_NAME}/models/target_clf/${MODEL_TYPE}_with_amounts${SUBST_TARGET}.tar.gz \
      CLF_SUBST_PATH=./presets/${DATASET_NAME}/models/substitute_clf/${MODEL_TYPE}_with_amounts${SUBST_TARGET}.tar.gz \
      PYTHONPATH=. python advsber/commands/attack.py configs/attackers/sampling_fool.jsonnet \
      --samples 500
  CUDA_VISIBLE_DEVICES=1 \
      DATA_PATH=./presets/${DATASET_NAME}/test.jsonl \
      OUTPUT_PATH=./results/${ATTACK_TYPE}/${DATASET_NAME}/${MODEL_TYPE}_con_sf_output.json \
      MASKED_LM_PATH=./presets/${DATASET_NAME}/lm.model.tar.gz \
      CLF_TARGET_PATH=./presets/${DATASET_NAME}/models/target_clf/${MODEL_TYPE}_with_amounts${SUBST_TARGET}.tar.gz \
      CLF_SUBST_PATH=./presets/${DATASET_NAME}/models/substitute_clf/${MODEL_TYPE}_with_amounts${SUBST_TARGET}.tar.gz \
      PYTHONPATH=. python advsber/commands/attack.py configs/attackers/concat_sampling_fool.jsonnet \
      --samples 500
  CUDA_VISIBLE_DEVICES=2 \
      DATA_PATH=./presets/${DATASET_NAME}/test.jsonl \
      OUTPUT_PATH=./results/${ATTACK_TYPE}/${DATASET_NAME}/${MODEL_TYPE}_gr_con_sf_output.json \
      MASKED_LM_PATH=./presets/${DATASET_NAME}/lm.model.tar.gz \
      CLF_TARGET_PATH=./presets/${DATASET_NAME}/models/target_clf/${MODEL_TYPE}_with_amounts${SUBST_TARGET}.tar.gz \
      CLF_SUBST_PATH=./presets/${DATASET_NAME}/models/substitute_clf/${MODEL_TYPE}_with_amounts${SUBST_TARGET}.tar.gz \
      PYTHONPATH=. python advsber/commands/attack.py configs/attackers/greedy_concat_sampling_fool.jsonnet \
      --samples 500
  CUDA_VISIBLE_DEVICES=2 \
      DATA_PATH=./presets/${DATASET_NAME}/test.jsonl \
      OUTPUT_PATH=./results/${ATTACK_TYPE}/${DATASET_NAME}/${MODEL_TYPE}_fgsm_output.json \
      MASKED_LM_PATH=./presets/${DATASET_NAME}/lm.model.tar.gz \
      CLF_TARGET_PATH=./presets/${DATASET_NAME}/models/target_clf/${MODEL_TYPE}_with_amounts${SUBST_TARGET}.tar.gz \
      CLF_SUBST_PATH=./presets/${DATASET_NAME}/models/substitute_clf/${MODEL_TYPE}_with_amounts${SUBST_TARGET}.tar.gz \
      PYTHONPATH=. python advsber/commands/attack.py configs/attackers/fgsm.jsonnet \
      --samples 500

else
    CUDA_VISIBLE_DEVICES=1 \
      DATA_PATH=./presets/${DATASET_NAME}/test.jsonl \
      OUTPUT_PATH=./results/${ATTACK_TYPE}/${DATASET_NAME}/${MODEL_TYPE}_sf_output.json \
      MASKED_LM_PATH=./presets/${DATASET_NAME}/lm.model.tar.gz \
      CLF_TARGET_PATH=./presets/${DATASET_NAME}/models/target_clf/${MODEL_TYPE}_with_amounts${SUBST_TARGET}.tar.gz \
      CLF_SUBST_PATH=./presets/${DATASET_NAME}/models/target_clf/${MODEL_TYPE}_with_amounts${SUBST_TARGET}.tar.gz \
      PYTHONPATH=. python advsber/commands/attack.py configs/attackers/sampling_fool.jsonnet \
      --samples 500
  CUDA_VISIBLE_DEVICES=1 \
      DATA_PATH=./presets/${DATASET_NAME}/test.jsonl \
      OUTPUT_PATH=./results/${ATTACK_TYPE}/${DATASET_NAME}/${MODEL_TYPE}_con_sf_output.json \
      MASKED_LM_PATH=./presets/${DATASET_NAME}/lm.model.tar.gz \
      CLF_TARGET_PATH=./presets/${DATASET_NAME}/models/target_clf/${MODEL_TYPE}_with_amounts${SUBST_TARGET}.tar.gz \
      CLF_SUBST_PATH=./presets/${DATASET_NAME}/models/target_clf/${MODEL_TYPE}_with_amounts${SUBST_TARGET}.tar.gz \
      PYTHONPATH=. python advsber/commands/attack.py configs/attackers/concat_sampling_fool.jsonnet \
      --samples 500
  CUDA_VISIBLE_DEVICES=2 \
      DATA_PATH=./presets/${DATASET_NAME}/test.jsonl \
      OUTPUT_PATH=./results/${ATTACK_TYPE}/${DATASET_NAME}/${MODEL_TYPE}_gr_con_sf_output.json \
      MASKED_LM_PATH=./presets/${DATASET_NAME}/lm.model.tar.gz \
      CLF_TARGET_PATH=./presets/${DATASET_NAME}/models/target_clf/${MODEL_TYPE}_with_amounts${SUBST_TARGET}.tar.gz \
      CLF_SUBST_PATH=./presets/${DATASET_NAME}/models/target_clf/${MODEL_TYPE}_with_amounts${SUBST_TARGET}.tar.gz \
      PYTHONPATH=. python advsber/commands/attack.py configs/attackers/greedy_concat_sampling_fool.jsonnet \
      --samples 500
  CUDA_VISIBLE_DEVICES=2 \
      DATA_PATH=./presets/${DATASET_NAME}/test.jsonl \
      OUTPUT_PATH=./results/${ATTACK_TYPE}/${DATASET_NAME}/${MODEL_TYPE}_fgsm_output.json \
      MASKED_LM_PATH=./presets/${DATASET_NAME}/lm.model.tar.gz \
      CLF_TARGET_PATH=./presets/${DATASET_NAME}/models/target_clf/${MODEL_TYPE}_with_amounts${SUBST_TARGET}.tar.gz \
      CLF_SUBST_PATH=./presets/${DATASET_NAME}/models/target_clf/${MODEL_TYPE}_with_amounts${SUBST_TARGET}.tar.gz \
      PYTHONPATH=. python advsber/commands/attack.py configs/attackers/fgsm.jsonnet \
      --samples 500

fi
echo "Sampling Fool:" >> ./results/${ATTACK_TYPE}/${DATASET_NAME}/${MODEL_TYPE}_output.json
PYTHONPATH=. python advsber/commands/evaluate.py ./results/${ATTACK_TYPE}/${DATASET_NAME}/${MODEL_TYPE}_sf_output.json >> ./results/${ATTACK_TYPE}/${DATASET_NAME}/${MODEL_TYPE}_output.json
echo "Concat Sampling Fool:" >> ./results/${ATTACK_TYPE}/${DATASET_NAME}/${MODEL_TYPE}_output.json
PYTHONPATH=. python advsber/commands/evaluate.py ./results/${ATTACK_TYPE}/${DATASET_NAME}/${MODEL_TYPE}_con_sf_output.json >> ./results/${ATTACK_TYPE}/${DATASET_NAME}/${MODEL_TYPE}_output.json
echo "Greedy Concat Sampling Fool:" >> ./results/${ATTACK_TYPE}/${DATASET_NAME}/${MODEL_TYPE}_output.json
PYTHONPATH=. python advsber/commands/evaluate.py ./results/${ATTACK_TYPE}/${DATASET_NAME}/${MODEL_TYPE}_gr_con_sf_output.json >> ./results/${ATTACK_TYPE}/${DATASET_NAME}/${MODEL_TYPE}_output.json
echo "FGSM:" >> ./results/${ATTACK_TYPE}/${DATASET_NAME}/${MODEL_TYPE}_output.json
PYTHONPATH=. python advsber/commands/evaluate.py ./results/${ATTACK_TYPE}/${DATASET_NAME}/${MODEL_TYPE}_fgsm_output.json >> ./results/${ATTACK_TYPE}/${DATASET_NAME}/${MODEL_TYPE}_output.json