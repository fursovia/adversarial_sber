#!/usr/bin/env bash


rm -rf ./presets/gender/vocabs/100_quantile
rm -rf ./presets/age/vocabs/100_quantile
rm -rf ./presets/age_short/vocabs/100_quantile
rm -rf ./presets/age_tinkoff/vocabs/100_quantile
rm -rf ./presets/rosbank/vocabs/100_quantile

rm -rf ./presets/gender/discretizers/100_quantile
rm -rf ./presets/gender/discretizers/50_quantile

rm -rf ./presets/age/discretizers/100_quantile
rm -rf ./presets/age/discretizers/50_quantile

rm -rf ./presets/age_short/discretizers/100_quantile
rm -rf ./presets/age_short/discretizers/50_quantile

rm -rf ./presets/age_tinkoff/discretizers/100_quantile
rm -rf ./presets/age_tinkoff/discretizers/50_quantile

rm -rf ./presets/rosbank/discretizers/100_quantile
rm -rf ./presets/rosbank/discretizers/50_quantile


PYTHONPATH=. python scripts/python_scripts/train_discretizers.py 'age'
PYTHONPATH=. python scripts/python_scripts/train_discretizers.py 'gender'
PYTHONPATH=. python scripts/python_scripts/train_discretizers.py 'age_short'
PYTHONPATH=. python scripts/python_scripts/train_discretizers.py 'age_tinkoff'
PYTHONPATH=. python scripts/python_scripts/train_discretizers.py 'rosbank'

PYTHONPATH=. python scripts/python_scripts/build_vocabs.py 'age'
PYTHONPATH=. python scripts/python_scripts/build_vocabs.py 'gender'
PYTHONPATH=. python scripts/python_scripts/build_vocabs.py 'age_short'
PYTHONPATH=. python scripts/python_scripts/build_vocabs.py 'age_tinkoff'
PYTHONPATH=. python scripts/python_scripts/build_vocabs.py 'rosbank'




