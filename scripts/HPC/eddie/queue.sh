#!/bin/bash
#$ -cwd
#$ -l h_rt=74:00:00
#$ -l h_vmem=256G
#$ -pe gpu 8

## INPUT ARGS:
# $1: config file path
# $2: output dir name

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load required modules
module load cuda/11.0.2

# Activate env
source ~/.bashrc
conda activate kge_new

# Make output directory
OUTPUT_DIR=/exports/eddie/scratch/s2408107/kge/local/experiments/$2
mkdir -p $OUTPUT_DIR

# Copy config to output directory
cp $1 $OUTPUT_DIR/config.yaml

# Run
devices=$(echo $CUDA_VISIBLE_DEVICES | sed 's/^.*: //' | sed -E 's/([^,]+)/cuda:\1/g')
echo $devices
echo $OUTPUT_DIR
kge resume $OUTPUT_DIR --search.device_pool cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7 --search.num_workers 8

echo ""
echo "============"