#!/bin/bash
#SBATCH --job-name=nnunet_hippocampus_train
#SBATCH --output=%x.o%j
#SBATCH --time=15:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8

# Load necessary modules
module purge
module load anaconda3/2021.05/gcc-9.2.0
module load cuda/10.2.89/intel-19.0.3.199

# Activate anaconda environment
source activate pulmembol

# Run nnUnet commands
nnUNet_plan_and_preprocess -t 4
nnUNet_train 3d_fullres nnUNetTrainerV2 4 0
