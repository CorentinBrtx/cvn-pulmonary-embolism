#!/bin/bash
#SBATCH --job-name=frangi_cadpe
#SBATCH --output=slurm_output/frangi/%x.o%j
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu_med
#SBATCH --mem=128G
#SBATCH --array=0-92
#SBATCH --mail-type=ALL

# Load necessary modules
module purge
module load anaconda3/2021.05/gcc-9.2.0
module load cuda/10.2.89/intel-19.0.3.199

# Activate anaconda environment
source activate pulmembol

# Run python script
python -m src.cadpe.frangi.compute_frangi $@
