#!/bin/bash

# find . -type d -name __pycache__ -exec rm -r {} \+
python -X pycache_prefix=./cache train_GAugO.py