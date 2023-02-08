#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=eval_%A.out
#SBATCH -N 1                            # nodes requested
#SBATCH -n 1                            # tasks requested
#SBATCH --gres=gpu:1                    # use 1 GPU
#SBATCH --mem=20000                     # memory in Mb
#SBATCH --partition=pascal
#SBATCH --account=BMAI-CDT-SL2-GPU
#SBATCH -t 2:00:00                     # time requested in hour:minute:seconds
#SBATCH --cpus-per-task=3

# Load required modules
module load cuda/10.2 cudnn/7.6_cuda-10.2

# Activate env
source ~/.bashrc
conda activate kge_playground

# Run
kge train /rds/user/$USER/hpc-work/kge/local/experiments/$2

echo ""
echo "============"