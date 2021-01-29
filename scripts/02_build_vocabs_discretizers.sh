#!/usr/bin/env bash

PYTHONPATH=. python scripts/python_scripts/train_discretizers.py 'age'
PYTHONPATH=. python scripts/python_scripts/train_discretizers.py 'gender'

PYTHONPATH=. python scripts/python_scripts/build_vocabs.py 'age'
PYTHONPATH=. python scripts/python_scripts/build_vocabs.py 'gender'





