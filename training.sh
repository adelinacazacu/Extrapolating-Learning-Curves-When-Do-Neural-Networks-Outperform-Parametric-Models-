#!/bin/bash
#SBATCH --partition=general --qos=short
#SBATCH --time=4:00:00
#SBATCH --mincpus=2
#SBATCH --mem=20000

#SBATCH --job-name=trainpfn
#SBATCH --output=out_trainpfn_%A_%a.txt
#SBATCH --error=err_trainpfn_%A_%a.txt
#SBATCH --array=0-4

# ------------------------------------------------------------------------------
# Setting up the environment
# ------------------------------------------------------------------------------

echo "----------------- Environment ------------------"
module use /opt/insy/modulefiles
module load cuda/11.3
module load miniconda/3.9

conda activate /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/adelina/env/lcdb-pfn

export PYTHONDONTWRITEBYTECODE=abc
export PYTHONUNBUFFERED=TRUE


cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/adelina/
srun python test.py --seed $SLURM_ARRAY_TASK_ID

conda deactivate