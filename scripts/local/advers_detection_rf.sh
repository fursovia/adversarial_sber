DATASET_NAME=${1:-"age"}
ATTACK_NAME=${2:-"sampling_fool"}
SUBST_NAME=${3:-"gru_with_amounts"}
TARG_NAME=${4:-"gru_with_amounts"}

PYTHONPATH=. python advsber/commands/detect_advers_rf.py ${DATASET_NAME} ${ATTACK_NAME} ${SUBST_NAME} ${TARG_NAME}
