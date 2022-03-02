#!/bin/bash
#SBATCH --job-name=check_inclusion_frangi
#SBATCH --output=slurm_output/check_inclusion_frangi/%x.o%j
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=cpu_med
#SBATCH --mem=64G

# Load necessary modules
module purge
module load anaconda3/2021.05/gcc-9.2.0

# Activate anaconda environment
source activate pulmembol

# Run python script
python -u -m src.cat_scan_filters.check_inclusion_frangi $@
