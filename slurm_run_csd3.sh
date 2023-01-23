#!/bin/bash
# # SBATCH -o /home/%u/slogs/sl_%A.out
# # SBATCH -e /home/%u/slogs/sl_%A.out
#SBATCH -N 1	                        # nodes requested
#SBATCH -n 1	                        # tasks requested
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
module load python/3.8 cuda/11.0 cuda/11.2 cudnn/8.1_cuda-11.2

# Set up scratch working folder
SCRATCH_DISK=/rds/user/
SCRATCH_HOME=${SCRATCH_DISK}/${USER}/hpc-work
mkdir -p ${SCRATCH_HOME}

# Copy data to scratch working folder
echo "Moving input data to the compute node's scratch space: $SCRATCH_DISK"
src_path=/home/${USER}/kge-playground/datasets/biokg
dest_path=${SCRATCH_HOME}/kge-playground/datasets/biokg
mkdir -p ${dest_path}  # make it if required
rsync --archive --update --compress --progress ${src_path}/ ${dest_path}

# Set up env
ENV_NAME=env
echo "Setting up environment"
virtualenv --system-site-packages ${ENV_NAME}
source ${ENV_NAME}/bin/activate
pip3 install matplotlib wandb python-dotenv pydantic torch torchvision torchaudio numpy==1.21 ogb 

# Run
echo "Running experiment"
LD_LIBRARY_PATH=/rds/user/hpcwulf1/hpc-work/lib/python3.8/site-packages/torch/lib/nvidia/cublas/lib/:$LD_LIBRART_PATH python scripts/train.py \
--config_filepath=$1 \
--log_to_wandb

# Outputs
OUTPUT_DIR=${SCRATCH_HOME}/kge-playground/outputs/
OUTPUT_HOME=${PWD}/exps/
mkdir -p ${OUTPUT_HOME}
rsync --archive --update --compress --progress ${OUTPUT_DIR} ${OUTPUT_HOME}

# Cleanup
rm -rf ${OUTPUT_DIR}

echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"