#!/bin/bash

## INPUT ARGS:
# $1: config file path
# $2: output dir name

# Make output directory
OUTPUT_DIR=/rds/user/$USER/hpc-work/kge/local/experiments/$2
mkdir -p $OUTPUT_DIR

# Copy config to output directory
cp $1 $OUTPUT_DIR/config.yaml

# Bash script
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$2
#SBATCH --output=$(echo $2 | tr "/" "_")_%A.out
#SBATCH -N 1                            # nodes requested
#SBATCH -n 1                            # tasks requested
#SBATCH --gres=gpu:4                    # use 4 GPUs
#SBATCH --mem=20000                     # memory in Mb
#SBATCH --partition=pascal
#SBATCH --account=BMAI-CDT-SL2-GPU
#SBATCH -t 24:00:00                     # time requested in hour:minute:seconds
#SBATCH --cpus-per-task=12

# Load required modules
module load cuda/10.2 cudnn/7.6_cuda-10.2

# Activate env
source ~/.bashrc
conda activate kge_playground 

# Run
kge resume $OUTPUT_DIR --search.device_pool cuda:0,cuda:1,cuda:2,cuda:3 --search.num_workers 8

echo ""
echo "============"
EOT