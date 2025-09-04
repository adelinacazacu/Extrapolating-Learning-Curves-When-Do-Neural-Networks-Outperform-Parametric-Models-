#!/bin/bash
#SBATCH --partition=general --qos=short
#SBATCH --time=1:00:00
#SBATCH --mincpus=2
#SBATCH --mem=4GB
#SBATCH --gres=gpu:l40:1

#SBATCH --job-name=pfn_stage_%A
#SBATCH --output=out_stage_%A.txt
#SBATCH --error=err_stage_%A.txt

export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# ------------------------------------------------------------------------------
# Setting up the environment
# ------------------------------------------------------------------------------

echo "----------------- Environment ------------------"
module use /opt/insy/modulefiles
module load cuda/11.3
module load miniconda/3.9

conda activate /tudelft.net/staff-umbrella/lcdb2/adelina/env/lcdb-pfn

export PYTHONDONTWRITEBYTECODE=abc
export PYTHONUNBUFFERED=TRUE
export PYTHONPATH="${PYTHONPATH}:${PWD}"

cd /tudelft.net/staff-umbrella/lcdb2/adelina/Extrapolating-Learning-Curves-When-Do-Neural-Networks-Outperform-Parametric-Models-

# Get stage from command line argument
STAGE_ID=$1
EPOCHS_PER_STAGE=100
START_EPOCH=$((STAGE_ID * EPOCHS_PER_STAGE))

echo "Running stage $STAGE_ID (epochs $START_EPOCH to $((START_EPOCH + EPOCHS_PER_STAGE)))"

srun python experiment2/training-pfn-lcdb11-experiment2.py \
    --stage $STAGE_ID \
    --epochs_per_stage $EPOCHS_PER_STAGE \
    --total_stages 10 \
    --seed 42

conda deactivate