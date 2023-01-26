#!/bin/bash

# Load modules
module load cuda/10.2 cudnn/7.6_cuda-10.2

# Initialise conda
conda init
source ~/.bashrc

# Create new env
conda env create -f environment.yaml