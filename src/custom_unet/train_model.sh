#!/bin/bash
#SBATCH --job-name=custom_unet_train
#SBATCH --output=slurm_output/custom_unet/%x.o%j
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --mem=128G

# Load necessary modules
module purge
module load anaconda3/2021.05/gcc-9.2.0
module load cuda/10.2.89/intel-19.0.3.199

# Activate anaconda environment
source activate pulmembol

# Run python script
python -m src.custom_unet.train_script $@
