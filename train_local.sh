#!/bin/bash
echo "Training Starting..."

python scripts/train.py \
--config_filepath=configs/complex.yaml

echo "Training Finished!!!"