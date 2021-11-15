#!/bin/bash
#SBATCH --job-name=scancovia_cadpe
#SBATCH --output=$HOME/slurm_output/scancovia/%x.o%j
#SBATCH --time=01:00:00
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --array=0-92

# Load necessary modules
module purge
module load anaconda3/2020.02/gcc-9.2.0
module load cuda/10.2.89/intel-19.0.3.199

# Activate anaconda environment
source activate pulmembol

# Run python script
python -m src.python_scripts.scancovia_cadpe -g
