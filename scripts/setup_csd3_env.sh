#!/bin/bash

# Load modules
module load cuda/11.2 cudnn/8.1_cuda-11.2

# Initialise conda
conda init
source ~/.bashrc

# Create new env
conda env create -f environment.yaml