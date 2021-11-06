# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 09:45:29 2021

@author: garan
"""

""" Sort labels by size """

import os
from glob import glob
import nrrd #pip install pynrrd, if pynrrd is not already installed
import nibabel as nib #pip install nibabel, if nibabel is not already installed
import numpy as np
from skimage import morphology,util,measure
#from skimage.measure import regionprops
import pandas as pd

import matplotlib.pyplot as plt


def read_nifti(path):
    img = nib.load(path)
    header = img.header
    return np.array(img.dataobj),header

def save_nifti(array, path, affine):
    res = nib.Nifti1Image(array, affine=affine)
    nib.save(res, path)
    

def get_affine(header):
    
    x = header['srow_x']
    y = header['srow_y']
    z = header['srow_z']
    
    affine = np.array([x,y,z,[0,0,0,1]])
    print(affine)
    return affine


    
baseDir = os.path.normpath('D:/Data_stage_IMA/CAD-PE/gt_bin_inf/')
files = glob(baseDir+'/*.nii.gz')



for file in files:

    
    img, hdr = read_nifti(file)

    big = morphology.area_opening(img,25000)
    small = img - morphology.area_opening(img,12500)
    middle = img - (small + big)

    save_nifti(small,os.path.join('D:/Data_stage_IMA/CAD-PE/gt_small/', file[-16:-13]+'_small.nii.gz'),get_affine(hdr))
    save_nifti(big,os.path.join('D:/Data_stage_IMA/CAD-PE/gt_big/', file[-16:-13]+'_big.nii.gz'),get_affine(hdr))
    save_nifti(middle,os.path.join('D:/Data_stage_IMA/CAD-PE/gt_middle/', file[-16:-13]+'_middle.nii.gz'),get_affine(hdr))
