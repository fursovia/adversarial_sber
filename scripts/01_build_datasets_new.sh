#!/usr/bin/env bash

mkdir ../data/gender_short
mkdir ../data/gender_short/lm
mkdir ../data/gender_short/target_clf
mkdir ../data/gender_short/substitute_clf

PYTHONPATH=. python scripts/python_scripts/create_gender_short.py

mkdir ../data/age_short
mkdir ../data/age_short/lm
mkdir ../data/age_short/target_clf
mkdir ../data/age_short/substitute_clf

PYTHONPATH=. python scripts/python_scripts/create_age_short.py


mkdir ../data/age_tinkoff
mkdir ../data/age_tinkoff/original
mkdir ../data/age_tinkoff/lm
mkdir ../data/age_tinkoff/target_clf
mkdir ../data/age_tinkoff/substitute_clf

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1XeB1bGtlyObeBw0tC3nTrIV4up2GUdDB' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1XeB1bGtlyObeBw0tC3nTrIV4up2GUdDB" -O '../data/age_tinkoff/original/transactions.csv' && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=19e03oucRNd6stRjEm0zs0rmceL-s7QlJ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=19e03oucRNd6stRjEm0zs0rmceL-s7QlJ" -O '../data/age_tinkoff/original/customer_train.csv' && rm -rf /tmp/cookies.txt


PYTHONPATH=. python scripts/python_scripts/create_age_tinkoff.py


mkdir ../data/gender_tinkoff
mkdir ../data/gender_tinkoff/lm
mkdir ../data/gender_tinkoff/target_clf
mkdir ../data/gender_tinkoff/substitute_clf

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1XeB1bGtlyObeBw0tC3nTrIV4up2GUdDB' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1XeB1bGtlyObeBw0tC3nTrIV4up2GUdDB" -O '../data/gender_tinkoff/original/transactions.csv' && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=19e03oucRNd6stRjEm0zs0rmceL-s7QlJ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=19e03oucRNd6stRjEm0zs0rmceL-s7QlJ" -O '../data/gender_tinkoff/original/customer_train.csv' && rm -rf /tmp/cookies.txt


PYTHONPATH=. python scripts/python_scripts/create_gender_tinkoff.py

mkdir ../data/rosbank
mkdir ../data/rosbank/original
mkdir ../data/rosbank/lm
mkdir ../data/rosbank/target_clf
mkdir ../data/rosbank/substitute_clf

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1gVprY6E6jK_VZHFxkOXSuLjPqDTYog7t' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1gVprY6E6jK_VZHFxkOXSuLjPqDTYog7t" -O '../data/rosbank/original/train.csv' && rm -rf /tmp/cookies.txt

PYTHONPATH=. python scripts/python_scripts/create_rosbank.py










