#!/bin/bash
#SBATCH --partition=general --qos=medium
#SBATCH --time=8:00:00
#SBATCH --mincpus=8
#SBATCH --mem=40000
#SBATCH --gres=gpu:1

#SBATCH --job-name=parallel_evaluation_exp1
#SBATCH --output=out_trainpfn_%A_%a.txt
#SBATCH --error=err_trainpfn_%A_%a.txt
#SBATCH --array=0-0

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
srun python experiment1/evaluation_parametric_vs_lcpfn.py --seed $SLURM_ARRAY_TASK_ID

conda deactivate