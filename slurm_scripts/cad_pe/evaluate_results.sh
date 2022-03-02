#!/bin/bash
#SBATCH --job-name=nnunet_cadpe_evaluate_results
#SBATCH --output=/workdir/shared/pulmembol/slurm_outputs/nnUNet/%x.o%j
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=cpu_med
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=ALL

# Load necessary modules
module purge
module load anaconda3/2021.05/gcc-9.2.0

# Activate anaconda environment
source activate pulmembol

# Run nnUnet commands
python -m src.cadpe.evaluate_detections $@
