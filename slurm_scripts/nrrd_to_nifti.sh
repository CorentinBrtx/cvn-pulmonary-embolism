#!/bin/bash
#SBATCH --job-name=nrrd_to_nifti
#SBATCH --output=%x.o%j
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu_med

# Load necessary modules
module purge
module load anaconda3/2021.05/gcc-9.2.0

# Activate anaconda environment
source activate pulmembol

# Run python script
#python ~/cvn-pulmonary-embolism/src/utils/nrrd_normalize.py -r $WORKDIR/CAD_PE/nrrd/rs
#python ~/cvn-pulmonary-embolism/src/utils/nrrd_normalize.py -r $WORKDIR/CAD_PE/nrrd/lungs_scancovia
#python ~/cvn-pulmonary-embolism/src/utils/nrrd_to_nifti.py -r $WORKDIR/CAD_PE/nrrd/images $WORKDIR/CAD_PE/nifti/images
#python ~/cvn-pulmonary-embolism/src/utils/nrrd_to_nifti.py -r $WORKDIR/CAD_PE/nrrd/rs $WORKDIR/CAD_PE/nifti/rs
#python ~/cvn-pulmonary-embolism/src/utils/nrrd_to_nifti.py -r $WORKDIR/CAD_PE/nrrd/frangi $WORKDIR/CAD_PE/nifti/frangi
python ~/cvn-pulmonary-embolism/src/utils/nrrd_to_nifti.py -r $WORKDIR/CAD_PE/nrrd/lungs_scancovia $WORKDIR/CAD_PE/nifti/lungs_scancovia
