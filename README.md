# Adversarial Sber


## Usage

Install dependencies

```bash
poetry install
poetry shell
```

### Training 

Train BERT

```bash
DISCRETIZER_PATH=./presets/age/discretizer_100_quantile \
    LM_TRAIN_DATA_PATH=./presets/age/sample.jsonl \
    LM_VALID_DATA_PATH=./presets/age/sample.jsonl \
    allennlp train ./configs/language_models/bert_with_amounts.jsonnet \
    --serialization-dir ./logs/age/lm \
    --include-package advsber \
    -o '{"trainer.cuda_device": -1}'
```

Train classifier

```bash
DISCRETIZER_PATH=./presets/age/discretizer_100_quantile \
    CLF_TRAIN_DATA_PATH=./presets/age/sample.jsonl \
    CLF_VALID_DATA_PATH=./presets/age/sample.jsonl \
    allennlp train ./configs/classifiers/gru_with_amounts.jsonnet \
    --serialization-dir ./logs/age/clf \
    --include-package advsber \
    -o '{"trainer.cuda_device": -1}'
```


### Attacking

Attack example

```bash
# data to attack
export DATA_PATH="./data/test.json"
# where to save results
export OUTPUT_PATH="./results/output.json"
export MASKED_LM_PATH="./presets/age/lm.model.tar.gz"
export CLF_PATH="./presets/age/target_clf.model.tar.gz"

PYTHONPATH=. python advsber/commands/attack.py configs/attackers/sampling_fool.jsonnet --samples 100
```