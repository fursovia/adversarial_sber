#!/usr/bin/env bash

rm -rf ../data
mkdir ../data

mkdir ../data/age
mkdir ../data/age/original
mkdir ../data/age/lm
mkdir ../data/age/target_clf
mkdir ../data/age/substitute_clf

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1roUn50LtDDPcl3UsI_1phZh-EryVstZi' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1roUn50LtDDPcl3UsI_1phZh-EryVstZi" -O '../data/age/original/transactions_train.csv' && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1QiNIOfoJhz0ILHJ3Xk_gunE7yPVMSI-8' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1QiNIOfoJhz0ILHJ3Xk_gunE7yPVMSI-8" -O '../data/age/original/train_target.csv' && rm -rf /tmp/cookies.txt


PYTHONPATH=. python scripts/python_scripts/create_age.py



mkdir ../data/gender
mkdir ../data/gender/original
mkdir ../data/gender/lm
mkdir ../data/gender/target_clf
mkdir ../data/gender/substitute_clf

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1bxGlxX9-T1vEc-x-AU1OWB4H6po2jWrm' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1bxGlxX9-T1vEc-x-AU1OWB4H6po2jWrm" -O '../data/gender/original/transactions.csv' && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1IlObdLeQkr5mZP0iVv2WcJTsxz4Nz62h' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1IlObdLeQkr5mZP0iVv2WcJTsxz4Nz62h" -O '../data/gender/original/gender_train.csv' && rm -rf /tmp/cookies.txt

PYTHONPATH=. python scripts/python_scripts/create_gender.py

