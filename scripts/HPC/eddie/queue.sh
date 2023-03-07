#!/bin/bash

## INPUT ARGS:
# $1: config file path
# $2: output dir name

# Make output directory
OUTPUT_DIR=/exports/eddie/scratch/s2408107/kge/local/experiments/$2
mkdir -p $OUTPUT_DIR

# Copy config to output directory
cp $1 $OUTPUT_DIR/config.yaml

# Bash script
qsub <<EOT
#!/bin/bash
#$ -N $(echo $2 | tr "/" "_")
#$ -cwd
#$ -l h_rt=48:00:00
#$ -l h_vmem=32G
#$ -pe gpu 2

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load required modules
module load cuda/11.0.2

# Activate env
source ~/.bashrc
conda activate kge_new

# Run
#export LD_LIBRARY_PATH=/home/$USER/.conda/envs/kge_playground/lib/python3.10/site-packages/nvidia/cublas/lib/:$LD_LIBRARY_PATH
kge resume $OUTPUT_DIR --search.device_pool cuda:0,cuda:1 --search.num_workers 2

echo ""
echo "============"
EOT
