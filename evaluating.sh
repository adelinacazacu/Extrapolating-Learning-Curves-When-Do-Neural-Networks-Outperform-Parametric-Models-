#!/bin/bash
#SBATCH --partition=general --qos=short
#SBATCH --time=3:00:00
#SBATCH --mincpus=16
#SBATCH --mem=16000
#SBATCH --gres=gpu:1

#SBATCH --job-name=paralleleval
#SBATCH --output=out_paralleleval_%A_%a.txt
#SBATCH --error=err_paralleleval_%A_%a.txt
#SBATCH --array=0-0

ulimit -n 65536
echo "File descriptor limit set to: $(ulimit -n)"

# ------------------------------------------------------------------------------
# Setting up the environment
# ------------------------------------------------------------------------------

top -b -d 30 -n 480 > top.log &

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