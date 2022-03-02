#!/bin/bash
#SBATCH --job-name=nnunet_cadpe_predict
#SBATCH --output=slurm_output/predict/%x.o%j
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=ALL

# Load necessary modules
module purge
module load anaconda3/2021.05/gcc-9.2.0
module load cuda/10.2.89/intel-19.0.3.199

# Activate anaconda environment
source activate pulmembol

# Run nnUnet commands

nnUNet_predict -i $1 -o $2 -m $3 -t $4