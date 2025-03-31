#!/bin/bash
#SBATCH --job-name=multimodal_job
#SBATCH --output=outputs/multimodal_job-%j.out
#SBATCH --error=outputs/multimodal_job-%j.err
#SBATCH --partition=gpu_4
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1


srun python ./src/evaluation.py