#!/bin/bash

# Initialise conda
conda init

# Create new env
conda create -n kge_playground

# Activate
conda activate kge_playground

# Install required packages
conda install -c conda-forge matplotlib wandb python-dotenv pydantic ogb
pip3 install torch torchvision torchaudio

# Deactivate
conda deactivate