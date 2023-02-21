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
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:4
#SBATCH --mem=100000
#SBATCH --cpus-per-gpu=32
#SBATCH --partition=ampere
#SBATCH --account=BMAI-CDT-SL2-GPU
#SBATCH -t 36:00:00

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment

# Load required modules
module load cuda/11.1 cudnn/8.0_cuda-11.1

# Activate env
source ~/.bashrc
conda activate kge_new

# Run
export LD_LIBRARY_PATH=/home/$USER/.conda/envs/kge_playground/lib/python3.10/site-packages/nvidia/cublas/lib/:$LD_LIBRARY_PATH
kge resume $OUTPUT_DIR --search.device_pool cuda:0,cuda:1,cuda:2,cuda:3 --search.num_workers 10

echo ""
echo "============"
EOT
