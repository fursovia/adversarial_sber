SUBST_CONFIG_NAME=${1:-"cnn_with_amounts"}
TARG_CONFIG_NAME=${2:-"lstm_with_amounts"}
NUM_SAMPLES=${3:-"500"}
DATASET_NAME=${4:-"age"}

CONFIG_DIR="configs/attackers"
PRESETS_DIR="./presets"
TEST_DATA_PATH=../data/${DATASET_NAME}/test.jsonl

for config_path in ${CONFIG_DIR}/*.jsonnet; do
    attack_name=$(basename ${config_path})
    attack_name="${attack_name%.*}"
    RESULTS_PATH=../experiments/attacks/${dataset_name}/targ_${targ_clf}/subst_${subst_clf}/${attack_name}
    mkdir -p ${RESULTS_PATH}

    DATA_PATH=${TEST_DATA_PATH} \
        OUTPUT_PATH=${RESULTS_PATH}/adversarial.json \
        MASKED_LM_PATH=../experiments/trained_models/${DATASET_NAME}/lm/bert_with_amounts \
        CLF_PATH=../experiments/trained_models/${DATASET_NAME}/substitute_clf/${SUBST_CONFIG_NAME}/model.tar.gz \
        PYTHONPATH=. python advsber/commands/attack.py ${config_path} --samples ${NUM_SAMPLES}

     PYTHONPATH=. python advsber/commands/evaluate.py ${RESULTS_PATH}/adversarial.json \
            --save-to=${RESULTS_PATH}/metrics.json \
            --target-clf-path=../experiments/trained_models/${DATASET_NAME}/target_clf/${TARG_CONFIG_NAME}/model.tar.gz
done

python advsber/commands/aggregate.py ../experiments/attacks/${dataset_name}
