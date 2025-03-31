#!/bin/bash
#SBATCH --job-name=multimodal_job
#SBATCH --output=outputs/multimodal_job.out
#SBATCH --error=outputs/multimodal_job.err
#SBATCH --time=3:00:00
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64gb

srun python ./src/main.py