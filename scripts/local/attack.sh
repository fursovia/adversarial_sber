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
    RESULTS_PATH=../experiments/attacks/${DATASET_NAME}/targ_${TARG_CONFIG_NAME}/subst_${SUBST_CONFIG_NAME}/${attack_name}
    
    mkdir -p ${RESULTS_PATH}

    DATA_PATH=${TEST_DATA_PATH} \
    OUTPUT_PATH=${RESULTS_PATH}/adversarial.json \
    MASKED_LM_PATH=./presets/${DATASET_NAME}/models/lm/bert_with_amounts.tar.gz \
    CLF_PATH=./presets/${DATASET_NAME}/models/substitute_clf/${SUBST_CONFIG_NAME}.tar.gz \
    PYTHONPATH=. python advsber/commands/attack.py ${config_path} --samples ${NUM_SAMPLES}

     PYTHONPATH=. python advsber/commands/evaluate.py ${RESULTS_PATH}/adversarial.json \
            --save-to=${RESULTS_PATH}/metrics.json \
            --target-clf-path=./presets/${DATASET_NAME}/models/target_clf/${TARG_CONFIG_NAME}.tar.gz
done

python scripts/python_scripts/aggregate.py ../experiments/attacks/${DATASET_NAME}