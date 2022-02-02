#!/bin/bash
#SBATCH --job-name=nnunet_cadpe_train_cascade
#SBATCH --output=/gpfs/users/berteauxc/cvn-pulmonary-embolism/slurm_output/nnUNet/%x.o%j
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=corentin.berteaux@student-cs.fr
#SBATCH --mail-type=ALL

# Load necessary modules
module purge
module load anaconda3/2021.05/gcc-9.2.0
module load cuda/10.2.89/intel-19.0.3.199

# Activate anaconda environment
source activate pulmembol

# Run nnUnet commands
nnUNet_train 3d_cascade_fullres nnUNetTrainerV2CascadeFullRes 501 4 --npz -c
