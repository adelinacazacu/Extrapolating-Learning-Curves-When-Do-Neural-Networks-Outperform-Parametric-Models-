#!/bin/bash

echo "Submitting staged training jobs..."

# Submit stage 0 (no dependencies)
echo "Submitting stage 0..."
STAGE0_JOB=$(sbatch --parsable stage_training.sh 0)
echo "Stage 0 job ID: $STAGE0_JOB"

# Submit remaining stages with dependencies
PREV_JOB=$STAGE0_JOB

for STAGE in {1..9}; do
    echo "Submitting stage $STAGE (depends on job $PREV_JOB)..."
    CURRENT_JOB=$(sbatch --parsable --dependency=afterok:$PREV_JOB stage_training.sh $STAGE)
    echo "Stage $STAGE job ID: $CURRENT_JOB"
    PREV_JOB=$CURRENT_JOB
done

echo "All jobs submitted! Final job ID: $PREV_JOB"
echo "Monitor with: squeue -u \$USER"