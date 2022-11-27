#!/bin/bash
python scripts/train_ogb.py \
--hidden_channels=256 \
--reg_lambda=1e-2 \
--epochs=100 \
--batch_size=256 \
--runs=1