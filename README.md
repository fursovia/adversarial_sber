# Adversarial Sber


## Usage

Install dependencies

```bash
poetry install
poetry shell
```

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