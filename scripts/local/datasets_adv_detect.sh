
SUBST_CONFIG_NAME=${1:-"cnn_with_amounts"}
TARG_CONFIG_NAME=${2:-"lstm_with_amounts"}
DATASET_NAME=${3:-"age"}
ATTACK_NAME=${4:-"fgsm"}

CONFIG_DIR="configs/attackers"
PRESETS_DIR="./presets"
TEST_DATA_PATH=../data/${DATASET_NAME}/test.jsonl

for config_path in ${CONFIG_DIR}/${ATTACK_NAME}.jsonnet; do
    attack_name=$(basename ${config_path})
    attack_name="${attack_name%.*}"
  
    RESULTS_PATH=../experiments/attacks/${DATASET_NAME}/targ_${TARG_CONFIG_NAME}/subst_${SUBST_CONFIG_NAME}/${attack_name}
    PYTHONPATH=. python advsber/commands/create_adv_detection_dataset.py ${RESULTS_PATH}

done

python scripts/python_scripts/aggregate.py ../experiments/attacks/${DATASET_NAME}
