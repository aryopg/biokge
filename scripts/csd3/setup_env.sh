#!/bin/bash

## INPUT ARGS:
# $1: dataset type (see options of scripts/data/download.py)

# Load modules
module load cuda/10.2 cudnn/7.6_cuda-10.2

# Initialise conda
conda init
source ~/.bashrc

# Create new env
conda env create -f environment.yaml

# Activate 
conda activate kge_playground

# Download libkge
git clone git@github.com:uma-pi1/kge.git

# Download data
python scripts/data/download.py --type=$1

# Move kge to scratch
mv kge /rds/user/${USER}/hpc-work

# Move data to scratch
mv biokg* /rds/user/${USER}/hpc-work/kge/data

# Install kge
cd /rds/user/${USER}/hpc-work/kge
pip install -e .

# Cleanup
conda deactivate