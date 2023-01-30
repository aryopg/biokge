#!/bin/bash

# Load modules
module load cuda/10.2 cudnn/7.6_cuda-10.2

# Initialise conda
conda init
source ~/.bashrc

# Create new env
conda env create -f environment.yaml

# Activate 
conda activate kge_playground

# Download data
python scripts/data_download.py

# Move kge to scratch
mv kge /rds/user/${USER}

# Install kge
cd /rds/user/${USER}/kge
pip install -e .

# Cleanup
conda deactivate