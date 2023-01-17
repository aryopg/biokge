#!/bin/bash
echo "Training Starting..."

python scripts/train.py \
--config_filepath=configs/complex_small.yaml

echo "Training Finished!!!"