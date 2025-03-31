#!/bin/bash
#SBATCH --job-name=multimodal_job
#SBATCH --output=outputs/multimodal_job.out
#SBATCH --error=outputs/multimodal_job.err
#SBATCH --partition=gpu_4_h100
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1


srun python ./src/train.py