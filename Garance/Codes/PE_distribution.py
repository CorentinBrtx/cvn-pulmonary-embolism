# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 16:59:05 2021

@author: garan
"""


"""Compute histogramme of PE sizes """
import os
from glob import glob
import nrrd #pip install pynrrd, if pynrrd is not already installed
import nibabel as nib #pip install nibabel, if nibabel is not already installed
import numpy as np
from skimage import morphology,util,measure
#from skimage.measure import regionprops
import pandas as pd

import matplotlib.pyplot as plt

baseDir = os.path.normpath('D:/Data_stage_IMA/CAD-PE/nii_gz/rs_nii_gz/')
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


#dict = {'id':file,'label':}
    
baseDir = os.path.normpath('D:/Data_stage_IMA/CAD-PE/nii_gz/rs_nii_gz/')
files = glob(baseDir+'/*.nii.gz')


patient_id = []
dicts = {}

x = []

for file in files:
    dic_patient = {}
    
    img, hdr = read_nifti(file)
    props = measure.regionprops(img)
    label_nb = []
    areas = []
    
    for prop in range(len(props)):
    #    print(prop, props[prop].area)
        
        label_nb.append(prop)
        areas.append(props[prop].area)
    x.append(areas)
 
    
plt.figure()    
plt.hist(x)  
plt.xlim(xmax= 125000, xmin = 0)
plt.ylim(ymax = 15, ymin = 0)
plt.xlabel('Embolism size (nb of pixels)')
plt.ylabel('Number of embolism')
plt.title('Number of PE by size')   


# Table woth PE label and its size in number of pixel
#    dic_patient = {'label': label_nb, 'area' : areas}
#    dicts[file[-16:-13]] = dic_patient
    

#df = pd.DataFrame(dicts)
#df.to_csv('D:/Data_stage_IMA/CAD-PE/pe_test.csv')
