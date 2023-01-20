#!/bin/bash
echo "Training Starting..."

python scripts/train.py \
--config_filepath=configs/complex_base_biokg.yaml

echo "Training Finished!!!"
