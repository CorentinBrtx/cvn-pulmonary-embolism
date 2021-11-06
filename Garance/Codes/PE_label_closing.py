# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 16:59:25 2021

@author: garan
"""

""" Remove holes in the GT"""

import os
from glob import glob
import nrrd #pip install pynrrd, if pynrrd is not already installed
import nibabel as nib #pip install nibabel, if nibabel is not already installed
import numpy as np
from skimage import morphology,measure


baseDir = os.path.normpath('D:/Data_stage_IMA/CAD-PE/gt_bin')
files = glob(baseDir+'/*.nii.gz')


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



for file in files:
    
    img, hdr = read_nifti(file)
    
    #fill holes with mathematical morphology (area closing)
    image_closed = morphology.remove_small_holes(img,1000)
    
    image_closed = image_closed.astype('int32')
    
    #save nifti
    save_nifti(image_closed,os.path.join('D:/Data_stage_IMA/CAD-PE/gt_bin_closed/', file[-16:-13]+'.nii.gz'),get_affine(hdr))
