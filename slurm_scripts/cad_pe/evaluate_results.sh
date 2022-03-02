#!/bin/bash
#SBATCH --job-name=cadpe_evaluate_results
#SBATCH --output=/workdir/shared/pulmembol/slurm_outputs/nnUNet/%x.o%j
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu_med

# Load necessary modules
module purge
module load anaconda3/2021.05/gcc-9.2.0

# Activate anaconda environment
source activate pulmembol

# Run nnUnet commands
python -m src.cadpe.evaluate_detections $@
