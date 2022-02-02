#!/bin/bash
#SBATCH --job-name=nnunet_cadpe_preprocess
#SBATCH --output=slurm_output/%x.o%j
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8

# Load necessary modules
module purge
module load anaconda3/2021.05/gcc-9.2.0
module load cuda/10.2.89/intel-19.0.3.199

# Activate anaconda environment
source activate pulmembol

# Run nnUnet commands
#nnUNet_plan_and_preprocess -t 501 --verify_dataset_integrity
nnUNet_plan_and_preprocess -t ${1:-501} 
