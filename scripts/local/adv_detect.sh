CONFIG_NAME=${1:-"gru_with_amounts"}
CLF_TYPE=${2:-"adv_detection"}
DISCRETIZER_NAME=${3:-"100_quantile"}
DATASET_NAME=${4:-"age"}
ATTACK_NAME=${5:-"fgsm"}

mkdir -p ../experiments/trained_models/${DATASET_NAME}/${CLF_TYPE}/${CONFIG_NAME}
CLF_TRAIN_DATA_PATH=../experiments/attacks/${DATASET_NAME}/targ_gru_with_amounts/subst_gru_with_amounts/${ATTACK_NAME}/train_adv_detection_dataset.jsonl \
CLF_VALID_DATA_PATH=../experiments/attacks/${DATASET_NAME}/targ_gru_with_amounts/subst_gru_with_amounts/${ATTACK_NAME}/valid_adv_detection_dataset.jsonl \
CLF_TEST_DATA_PATH=../experiments/attacks/${DATASET_NAME}/targ_gru_with_amounts/subst_gru_with_amounts/${ATTACK_NAME}/test_adv_detection_dataset.jsonl \
DISCRETIZER_PATH=./presets/${DATASET_NAME}/discretizers/${DISCRETIZER_NAME} \
VOCAB_PATH=./presets/${DATASET_NAME}/vocabs/${DISCRETIZER_NAME} \
RANDOM_SEED=0 \
allennlp train ./configs/classifiers/${CONFIG_NAME}.jsonnet \
 --serialization-dir ../experiments/trained_models/${DATASET_NAME}/${CLF_TYPE}/${CONFIG_NAME} \
 --include-package advsber
    #rm -r ../experiments/trained_models

