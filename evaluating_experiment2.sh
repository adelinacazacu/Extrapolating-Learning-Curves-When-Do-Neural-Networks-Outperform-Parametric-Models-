#!/bin/bash
#SBATCH --partition=general --qos=short
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16GB
#SBATCH --nodelist=influ4,influ5,influ6,gpu14,gpu15,gpu16,gpu17,gpu18,gpu19,gpu20,gpu21,gpu22,gpu23,gpu24,gpu25,gpu26,gpu27,gpu28,gpu29,gpu30,gpu31,gpu32,gpu33,gpu34,gpu35

#SBATCH --job-name=paralleleval
#SBATCH --output=out_paralleleval_%A_%a.txt
#SBATCH --error=err_paralleleval_%A_%a.txt
#SBATCH --array=44-46

ulimit -n 65536
echo "File descriptor limit set to: $(ulimit -n)"

# ------------------------------------------------------------------------------
# Setting up the environment
# ------------------------------------------------------------------------------

#top -b -d 30 -n 480 > top.log &

echo "----------------- Environment ------------------"
module use /opt/insy/modulefiles
module load cuda/11.3
module load miniconda/3.9

conda activate /tudelft.net/staff-umbrella/lcdb2/adelina/env/lcdb-pfn

export PYTHONDONTWRITEBYTECODE=abc
export PYTHONUNBUFFERED=TRUE
export PYTHONPATH="${PYTHONPATH}:${PWD}"

cd /tudelft.net/staff-umbrella/lcdb2/adelina/Extrapolating-Learning-Curves-When-Do-Neural-Networks-Outperform-Parametric-Models-
srun python experiment2/experiment2_shape_evaluation_parametric_vs_lcpfn.py --seed $SLURM_ARRAY_TASK_ID

conda deactivate