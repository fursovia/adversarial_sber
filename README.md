# Adversarial Sber


## Usage

Install dependencies

```bash
poetry install
poetry shell
```

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