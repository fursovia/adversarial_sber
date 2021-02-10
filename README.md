# Adversarial Sber


## Usage
### Step 0. Install dependencies

```bash
poetry install
poetry shell
```

## Reproducibility

To reproduce all our experiments, please, run all bash scripts from `./scripts` in numerical order:

```
01_build_datasets.sh
02_build_vocabs_discretizers.sh
03_train_all_classifiers.sh
04_train_all_lm.sh
05_attack_all_classifiers.sh
```

### Step 1. Building datasets

We are working with following transactional datasets: `Age` and `Gender`. They are available on the following link: 
1. `Age`: https://drive.google.com/drive/u/0/folders/1oTkPI5Z091JbXHmOR0N7D-KKpN_9Qiyp
2. `Gender`: https://drive.google.com/drive/u/0/folders/1FJYWM5P9wUzieC8uPkSx7JKHhcZJboXg

To get the processed datasets, you need to run

`bash scripts/01_build_datasets.sh`

As a result, in the directory `../data` you will get the data for the next experiments in following directories:

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
Each  row  in `.jsonl` -- is a dictionary with keys `transactions`, `amounts`, `label`, `client_id`. 

### Step 2. Building vocabs and discretizers.
To build vocabulary and train discretizer run:

`bash scripts/02_build_vocabs_discretizers.sh`

Traind discretizers will be stored in `./presets/${DATASET_NAME}/discretizers/100_quantile`, and vocabs in `./presets/${DATASET_NAME}/vocabs/100_quantile`.

## Experiments

All results will be at `../experiments`:

1. Trained models: `../experiments/trained_models`
2. Result of attacks: `../experiments/attacks`

### Step 3. Training all classifiers.

To train all classifiers (LSTM, CNN, GRU) run:

`bash scripts/03_train_all_classifiers.sh`

As a result, all trained models will be stored in `../experiments/trained_models`.


If you want to train a certain model, use:

`bash scripts/local/train_clf.sh ${config_name} ${clf_type} "100_quantile" ${dataset_name}`,

where `clf_type` is "substitute" or "target" and `config_name` is "gru_with_amounts"/"lstm_with_amounts"/"cnn_with_amounts".

### Step 4. Training language models.

To train all lanuage models run:

`bash scripts/03_train_all_lm.sh`

As a result, all trained language models will be stored in `../experiments/trained_models`.

### Step 5. Attacking all models

To attack all models run:

`bash scripts/05_attack_all_classifiers.sh`

The results will be stored in `../experiments/trained_models/attacks`. There metrics of resulted attacks will be available at `.metrics.json` and adversarial data in `adversarial.json`.

If you want to attack a certain model for fixed dataset, you can use:

`bash scripts/local/attack.sh ${subst_clf} ${targ_clf} ${number of samples to attack} ${dataset_name}`,

where `subst_clf` and `targ_clf` are "gru_with_amounts"/"lstm_with_amounts"/"cnn_with_amounts".
