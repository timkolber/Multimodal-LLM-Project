#!/bin/bash
#SBATCH --job-name=multimodal_job
#SBATCH --output=outputs/multimodal_job-%j.out
#SBATCH --error=outputs/multimodal_job-%j.err
#SBATCH --time=00:15:00
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64gb

srun python ./src/metrics.py