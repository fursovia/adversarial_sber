# Adversarial Sber


## Usage

Install dependencies

```bash
poetry install
poetry shell
```

## Data

```text
.
├── lm
│   ├── train.jsonl
│   └── valid.jsonl
├── substitute_clf
│   ├── train.jsonl
│   └── valid.jsonl
├── target_clf
│   ├── train.jsonl
│   └── valid.jsonl
└── test.jsonl
```

### Training 

Train vanilla BERT (w/o amounts)

```bash
CUDA_VISIBLE_DEVICES="0" bash bin/train_lm.sh bert ./presets/age
```

Train BERT (w/ amounts)

```bash
CUDA_VISIBLE_DEVICES="0" bash bin/train_lm.sh bert_with_amounts ./presets/age
```

Train substitute classifier

```bash
CUDA_VISIBLE_DEVICES="0" bash bin/train_clf.sh gru_with_amounts ./presets/age substitute
```

Train target classifier

```bash
CUDA_VISIBLE_DEVICES="0" bash bin/train_clf.sh gru_with_amounts ./presets/age target
```

### Attacking

Attack example

```bash
CUDA_VISIBLE_DEVICES="1" \
    DATA_PATH="/notebook/morozov/data/age/test.jsonl" \
    OUTPUT_PATH="./results/output.json" \
    MASKED_LM_PATH="./presets/age/lm.model.tar.gz" \
    CLF_PATH="./presets/age/gru_age_subsitute_clf.tar.gz" \
    PYTHONPATH=. python advsber/commands/attack.py configs/attackers/sampling_fool.jsonnet \
    --samples 500
```


Evaluate attack

```bash
PYTHONPATH=. python advsber/commands/evaluate.py ./results/output.json
```