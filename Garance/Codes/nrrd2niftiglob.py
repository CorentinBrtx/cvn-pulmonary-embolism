# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 12:03:10 2021

@author: garan

Convert multiple nnrd images into nifti """
import os
from glob import glob

import nibabel as nib  # pip install nibabel, if nibabel is not already installed
import nrrd  # pip install pynrrd, if pynrrd is not already installed
import numpy as np

## Change baseDir with the path of your images or labels

# baseDir for images
# baseDir = os.path.normpath('D:/Data_stage_IMA/CAD-PE/images/images')

# baseDir for labels
baseDir = os.path.normpath("D:/Data_stage_IMA/CAD-PE/nrrd/rs/rs")


files = glob(baseDir + "/*.nrrd")

for file in files:

    # load nrrd
    _nrrd = nrrd.read(file)
    # load data and header informations
    data = _nrrd[0]
    header = _nrrd[1]

    # create affine of transformation for nifti file with header infos
    # you need to adapt the signs between each coefficient in the affine to match
    # the wanted orientation
    affine = np.zeros((4, 4), dtype="float")
    affine[0:3, 0:3] = -header["space directions"]
    affine[3, 3] = 1
    affine[2, 2] = -affine[2, 2]

    affine[0:3, 3] = -header["space origin"]
    affine[2, 3] = -affine[2, 3]

    # labels binarization (option)

    #    data = np.where(data>0,1,0)

    # conversion into nifti
    img = nib.Nifti1Image(data, affine)

    # save nifti
    nib.save(img, os.path.join("D:/Data_stage_IMA/CAD-PE/nrrd/rs", file[-15:-5] + ".nii.gz"))
#   nib.save(img,os.path.join('D:/Data_stage_IMA/CAD-PE/images', file[-8:-5] + '.nii'))
#   nib.save(img,os.path.join(baseDir, file[-8:-5] + '.nii.gz'))
