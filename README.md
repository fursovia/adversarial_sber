# Adversarial Sber


## Usage
### Step 0. Install dependencies

```bash
poetry install
poetry shell
```

## Reproducibility

To reproduce all our experiments, please, run all bash scripts from `./scripts` in numerical order.

### Step 1. Building datasets

We are working with following transactional datasets: `Age` and `Gender`. They are available on the following link: 
1. `Age`: https://drive.google.com/drive/u/0/folders/1oTkPI5Z091JbXHmOR0N7D-KKpN_9Qiyp
2. `Gender`: https://drive.google.com/drive/u/0/folders/1FJYWM5P9wUzieC8uPkSx7JKHhcZJboXg

To get the processed datasets, you need to run

`bash scripts/01_build_datasets.sh`

As a result, in the directory `../data` you will get data for the next experiments.


### Step 2. Building vocabs and discretizers.

`bash scripts/02_build_vocabs_discretizers.sh`

## Experiments

All results will be at `../experiments`

### Step 3. Training all classifiers.

`bash scripts/03_train_all_classifiers.sh`



## Data

Данные должны лежать в таком формате. в папке `lm` -- данные для обучения лингвистической модели.
`substitute_clf` -- данные для обучения substitute classifier. 
`target_clf` -- для обучения target classifier.

Каждая строка в `.jsonl` -- это словарь с ключами `transactions`, `amounts`, `label`, `client_id`. 
(пример см. в папке `presets`).

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

Чтобы обучить все модели (классификаторы и лингвистические), запускаем следующую команду.

```bash
CUDA_VISIBLE_DEVICES="0" bash bin/train_all_models.sh
```

Затем, архивы всех моделей `model.tar.gz` кладем в папку `presets`. 

### Attacking

Чтобы проатаковать модели всеми доступными атаками, запускаем

```bash
CUDA_VISIBLE_DEVICES="0" bash bin/attack_all_models.sh 5000 DATA_PATH
```

где `5000` -- это количество примеров из `test.jsonl`, которые мы будем атаковать.
`DATA_PATH` -- путь до папки с данными.

На выходе вы получите `.csv` файл со всеми метриками.


Если хотите проатаковать конкретную модель, конкретным методом, запускайте

```bash
CUDA_VISIBLE_DEVICES="1" \
    DATA_PATH="/notebook/morozov/data/age/test.jsonl" \
    OUTPUT_PATH="./results/output.json" \
    MASKED_LM_PATH="./presets/age/lm.model.tar.gz" \
    CLF_PATH="./presets/age/gru_age_subsitute_clf.tar.gz" \
    PYTHONPATH=. python advsber/commands/attack.py configs/attackers/sampling_fool.jsonnet \
    --samples 500
```

Подсчет метрик атак

```bash
PYTHONPATH=. python advsber/commands/evaluate.py ./results/output.json
```

## Adversarial detection

Для запуска эксперимента по детектированию атак

```bash
CUDA_VISIBLE_DEVICES="0" bash bin/adversarial_detection.sh


```

## Adversarial train
Для запуска эксперимента по адверсальному обучению

```bash
CUDA_VISIBLE_DEVICES="0" bash bin/adversarial_train.sh \
SUBSTITUTE_MODEL_DIR MASKED_LM_DIR DATA_DIR NUM_SAMPLES OUTPUT_DIR \
DATASET_TYPE TARGET_CONFIG_NAME

```
где 
1. `SUBSTITUTE_MODEL_DIR` -- путь до суррогатного классификатора.
2. `MASKED_LM_DIR` -- путь до лингвистической модели.
3. `DATA_DIR` -- путь до папки с обучающими выборками `target_clf` и `substitute_clf`.
4. `NUM_SAMPLES` -- число адверсальных примеров, которые будут добавлены в обучающую выборку после каждой из 4-х атак: sampling fool, concat sampling fool, fgsm, concat fgsm.
5. `OUTPUT_DIR` -- путь до папки, куда будут сохранены новая обучающая выборка, результаты атак, обученная на новой выборке модель и логи обучения.
6. `DATASET_TYPE` -- название датасета. Например, gender или age.
7. `TARGET_CONFIG_NAME` -- название конфига целевого классификатора. Например, lstm_with_amounts.

Примеры аргументов можно найти в самом bash скрипте: `bin/adversarial_train.sh`
