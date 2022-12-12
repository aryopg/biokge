#!/bin/bash
python scripts/train_ogb.py \
--hidden_channels=256 \
--n3_lambda=1e-2 \
--epochs=100 \
--batch_size=256 \
--runs=1