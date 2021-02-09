#!/usr/bin/env bash


rm -rf ./presets/gender/vocabs/100_quantile
rm -rf ./presets/age/vocabs/100_quantile

rm -rf ./presets/gender/discretizers/100_quantile
rm -rf ./presets/gender/discretizers/50_quantile
rm -rf ./presets/age/discretizers/100_quantile
rm -rf ./presets/age/discretizers/50_quantile

PYTHONPATH=. python scripts/python_scripts/train_discretizers.py 'age'
PYTHONPATH=. python scripts/python_scripts/train_discretizers.py 'gender'

PYTHONPATH=. python scripts/python_scripts/build_vocabs.py 'age'
PYTHONPATH=. python scripts/python_scripts/build_vocabs.py 'gender'





