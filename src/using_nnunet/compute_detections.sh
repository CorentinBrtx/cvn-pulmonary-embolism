#!/bin/bash
#SBATCH --job-name=dectect
#SBATCH --output=slurm_output/nnunet/%x.o%j
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --partition=cpu_long


# Load necessary modules
module purge
module load anaconda3/2021.05/gcc-9.2.0

# Activate anaconda environment
source activate pulmembol

python -m src.using_nnunet.compute_detection_from_nnunet $@