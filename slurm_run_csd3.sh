#!/bin/bash
# # SBATCH -o /home/%u/slogs/sl_%A.out
# # SBATCH -e /home/%u/slogs/sl_%A.out
#SBATCH -N 1                            # nodes requested
#SBATCH -n 1                            # tasks requested
#SBATCH --gres=gpu:1                    # use 1 GPU
#SBATCH --mem=20000                     # memory in Mb
#SBATCH --partition=pascal
#SBATCH --account=BMAI-CDT-SL2-GPU
#SBATCH -t 24:00:00                     # time requested in hour:minute:seconds
#SBATCH --cpus-per-task=3

echo "Job running on ${SLURM_JOB_NODELIST}"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"

# Load required modules
module load cuda/10.2 cudnn/7.6_cuda-10.2

# Set up scratch working folder
SCRATCH_HOME=/rds/user/${USER}/hpc-work/kge_playground
mkdir -p ${SCRATCH_HOME}

# Activate env
source ~/.bashrc
conda activate kge_playground

# Run
echo "Running experiment"
PYSTOW_HOME=${SCRATCH_HOME} \
python scripts/train.py \
--config=$1 \
--log_to_wandb \
--output_path=${SCRATCH_HOME}/outputs

# Get outputs
mkdir -p ${PWD}/outputs
rsync --archive --update --compress --progress ${SCRATCH_HOME}/outputs ${PWD}/outputs

# Cleanup
rm -rf ${SCRATCH_HOME}

echo ""
echo "============"
