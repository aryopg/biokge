#!/bin/bash
# # SBATCH -o /home/%u/slogs/sl_%A.out
# # SBATCH -e /home/%u/slogs/sl_%A.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:1  # use 1 GPU
#SBATCH --mem=2000  # memory in Mb
#SBATCH --partition=pascal
#SBATCH --account=BMAI-CDT-SL2-GPU
#SBATCH -t 24:00:00  # time requested in hour:minute:seconds
#SBATCH --cpus-per-task=3

module add cuda/11.1 cudnn/8.0_cuda-11.1 miniconda/3
# module add miniconda/3
# module add cuda/11.0  # avoids compatibility complaint
# module add cudnn/8.0_cuda-11.1
# module add cuda/11.1
# module add git-2.14.1-gcc-5.4.0-acb553e
source ~/.bashrc
conda init

echo "Job running on ${SLURM_JOB_NODELIST}"

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"

echo "Setting up bash enviroment"
source ~/.bashrc
set -e
SCRATCH_DISK=/rds/user/
SCRATCH_HOME=${SCRATCH_DISK}/${USER}/hpc-work
mkdir -p ${SCRATCH_HOME}

# Activate your conda environment
CONDA_ENV_NAME=kge_playground
echo "Setting up conda environment: ${CONDA_ENV_NAME}"
conda create --name ${CONDA_ENV_NAME} python=3.8
conda activate ${CONDA_ENV_NAME}
conda install -y ogb wandb python-dotenv pre-commit black isort pydantic -c conda-forge
#conda install -y python-dotenv -c conda-forge
conda install -y pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia

echo "Moving input data to the compute node's scratch space: $SCRATCH_DISK"
src_path=/home/${USER}/kge-playground/datasets/biokg
dest_path=${SCRATCH_HOME}/kge-playground/datasets/biokg
mkdir -p ${dest_path}  # make it if required
rsync --archive --update --compress --progress ${src_path}/ ${dest_path}

echo "Running experiment"
# limit of 12 GB GPU is hidden 256 and batch size 256
which python
python scripts/train.py \
--config_filepath=$1
--log_to_wandb

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
