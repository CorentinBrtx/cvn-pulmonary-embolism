#!/bin/bash
#SBATCH --job-name=nnunet_cadpe_predict
#SBATCH --output=/gpfs/users/berteauxc/cvn-pulmonary-embolism/slurm_output/nnUNet/%x.o%j
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=ALL

# Load necessary modules
module purge
module load anaconda3/2021.05/gcc-9.2.0
module load cuda/10.2.89/intel-19.0.3.199

# Activate anaconda environment
source activate pulmembol

# Run nnUnet commands
# Here is how you should predict test cases. Run in sequential order and replace all input and output folder names with your personalized ones

FOLDER_WITH_TEST_CASES="/gpfs/workdir/shared/pulmembol/nnUNet/nnUNet_raw/nnUNet_raw_data/Task501_EmbolismCADPE/imagesTs"
OUTPUT_FOLDER_MODEL1="/gpfs/workdir/shared/pulmembol/nnUNet/nnUNet_results/predictionsTs_model1"
OUTPUT_FOLDER_MODEL2="/gpfs/workdir/shared/pulmembol/nnUNet/nnUNet_results/predictionsTs_model2"
OUTPUT_FOLDER_ENSEMBLE="/gpfs/workdir/shared/pulmembol/nnUNet/nnUNet_results/predictionsTs"

nnUNet_predict -i $FOLDER_WITH_TEST_CASES -o $OUTPUT_FOLDER_MODEL1 -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 -t Task501_EmbolismCADPE --save_npz
nnUNet_predict -i $FOLDER_WITH_TEST_CASES -o $OUTPUT_FOLDER_MODEL2 -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_cascade_fullres -p nnUNetPlansv2.1 -t Task501_EmbolismCADPE --save_npz
nnUNet_ensemble -f $OUTPUT_FOLDER_MODEL1 $OUTPUT_FOLDER_MODEL2 -o $OUTPUT_FOLDER_ENSEMBLE -pp /gpfs/workdir/shared/pulmembol/nnUNet/nnUNet_trained_models/nnUNet/ensembles/Task501_EmbolismCADPE/ensemble_3d_fullres__nnUNetTrainerV2__nnUNetPlansv2.1--3d_cascade_fullres__nnUNetTrainerV2CascadeFullRes__nnUNetPlansv2.1/postprocessing.json
