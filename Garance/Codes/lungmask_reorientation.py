# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 14:18:25 2021

@author: garan
"""

import os
from glob import glob
import nrrd  # pip install pynrrd, if pynrrd is not already installed
import nibabel as nib  # pip install nibabel, if nibabel is not already installed
import numpy as np


""" Reorient lung masks obtained with scancovia """


def read_nifti(path):
    img = nib.load(path)
    header = img.header
    return np.array(img.dataobj), header


def save_nifti(array, path, affine):
    res = nib.Nifti1Image(array, affine=affine)
    nib.save(res, path)


def get_affine(header):

    x = header["srow_x"]
    y = header["srow_y"]
    z = header["srow_z"]

    #    affine = np.array([x,y,z,[0,0,0,1]])
    affine = np.array([y, x, z, [0, 0, 0, 1]])
    #    affine[0,0] = - affine[0,0]
    #    affine[1,1] = - affine[1,1]
    #    affine[2,2] = - affine[2,2]

    affine[0, 3] = x[3]
    affine[1, 3] = y[3]
    # print(x,y,z)
    print(affine)
    return affine


img, header = read_nifti("D:/Data_stage_IMA/CAD-PE/nii_gz/images_nii_gz/002.nii.gz")

lungmask, _ = read_nifti(
    "D:/Data_stage_IMA/CAD-PE/scancovia_results/lung_masks/PE_002_00lung_mask.nii.gz"
)

save_nifti(
    lungmask,
    "D:/Data_stage_IMA/CAD-PE/scancovia_results/lung_masks_reoriented/CAD-PE_002_lungmask_r.nii.gz",
    get_affine(header),
)
