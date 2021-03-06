#!/bin/bash
#SBATCH --job-name=custom_unet_train
#SBATCH --output=%x.o%j
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G

# Load necessary modules
module purge
module load anaconda3/2021.05/gcc-9.2.0
module load cuda/10.2.89/intel-19.0.3.199

# Activate anaconda environment
source activate pulmembol

# Run python script
python -um src.custom_unet.train_script \
--train_img_path $PULMEMBOL/nnUNet/nnUNet_raw/nnUNet_raw_data/Task501_EmbolismCADPE/imagesTr \
--train_seg_path $PULMEMBOL/nnUNet/nnUNet_raw/nnUNet_raw_data/Task501_EmbolismCADPE/labelsTr \
--patch_size=128,128,128 \
--n_layers=3 \
--iterable \
--internal_channels 16 \
--batch_size 1 \
--model_path=$PULMEMBOL/custom_unet/16_channels
