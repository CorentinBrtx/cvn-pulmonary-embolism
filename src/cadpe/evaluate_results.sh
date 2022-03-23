#!/bin/bash
#SBATCH --job-name=cadpe_evaluate_results
#SBATCH --output=%x.o%j
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --partition=cpu_med
#SBATCH --mem=64G

# Load necessary modules
module purge
module load anaconda3/2021.05/gcc-9.2.0

# Activate anaconda environment
source activate pulmembol

# Run nnUnet commands
python -m src.cadpe.evaluate_detections --ground_truth_folders $PULMEMBOL/rs_nifti $@
