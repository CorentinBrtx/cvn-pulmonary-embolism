#!/bin/bash
#SBATCH --job-name=custom_unet_train
#SBATCH --output=%x.o%j
#SBATCH --time=23:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

# Load necessary modules
module purge
module load anaconda3/2021.05/gcc-9.2.0
module load cuda/10.2.89/intel-19.0.3.199

# Activate anaconda environment
source activate pulmembol

# Run python script
python -m src.custom_unet.train_script --train_img_path $PULMEMBOL/nnUNet/nnUNet_raw/nnUNet_raw_data/Task501_EmbolismCADPE/imagesTr --train_seg_path $PULMEMBOL/nnUNet/nnUNet_raw/nnUNet_raw_data/Task501_EmbolismCADPE/labelsTr $@
