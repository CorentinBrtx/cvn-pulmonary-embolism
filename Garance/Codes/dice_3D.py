# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 09:35:03 2021

@author: garan
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 09:36:09 2021

@author: garan
"""

""" Compute Dice score for 3D images and other metrics"""

import sys

sys.path.insert(0, "../../sources")

from glob import glob
import os
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_bf

# from skimage.metrics import structural_similarity as ssim1
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from scipy.spatial.distance import directed_hausdorff
import numpy as np
import nibabel as nib  # pip install nibabel, if nibabel is not already installed


def normalizeImage(image, max_value):
    min = np.amin(image)
    max = np.amax(image)
    n_im = ((image.astype(np.float) - min) / (max - min)) * max_value
    return n_im


def dice(img_bin, gt):
    tn, fp, fn, tp = confusionMatrix(img_bin, gt)
    return 2 * tp / (2 * tp + fn + fp)


def get_dice_3D(pred, gt):
    """
    red = False positive
    white = true positive
    green = false negative
    black = true negative
    """

    print(pred.shape)
    print(gt.shape)

    TP = np.sum((gt > 0) & (pred > 0))
    FP = np.sum((gt == 0) & (pred > 0))
    FN = np.sum((gt > 0) & (pred == 0))

    #
    dice_2d = dice(pred, gt)
    dice_3d = 2 * TP / (2 * TP + FN + FP)

    return dice_2d, dice_3d


def confusionMatrix(img_bin, gt):
    tn, fp, fn, tp = confusion_matrix(gt.ravel(), img_bin.ravel()).ravel()
    return (tn, fp, fn, tp)


def hausdorff(img_bin, gt):
    return max(directed_hausdorff(img_bin, gt)[0], directed_hausdorff(gt, img_bin)[0])


def roc(img, gt):
    seuillages = np.arange(0, 1, 0.01)
    MCC = 0
    best_seuill = np.zeros(img.shape)
    for i in seuillages:
        img_seuil = (img > float(i)) * 1.0
        mcc = MCC(img_seuil, gt)
        if mcc >= MCC:
            best_seuill = img_seuil
            MCC = mcc
    return MCC, best_seuill


def evaluate_image_binaire(img_bin, gt, mask):
    mcc = MCC(img_bin, gt)
    (tn, fp, fn, tp) = confusionMatrix_DRIVE(img_bin, gt, mask)
    d = dice_DRIVE(img_bin, gt, mask)
    h = hausdorff(img_bin, gt)
    acc = (tp + tn) / (tn + fp + fn + tp)
    sp = tn / (tn + fp)
    se = tp / (tp + fn)
    fpr = 1 - sp
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    metrics = {
        "matrix_confusion": (tn, fp, fn, tp),
        "acc": acc,
        "sp": sp,
        "se": se,
        "fpr": fpr,
        "ppv": ppv,
        "npv": npv,
        "mcc": mcc,
        "dice": d,
        "hausdorff": h,
    }
    return metrics


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

    affine = np.array([x, y, z, [0, 0, 0, 1]])
    # affine[0,0] = - affine[0,0]
    #    affine[1,1] = - affine[1,1]
    #    affine[2,2] = - affine[2,2]
    #    affine[1,1] =  affine[1,1]
    #    affine[2,2] =  affine[2,2]
    # print(x,y,z)
    print(affine)
    return affine


# gt, gt_header = read_nifti('D:/Data_stage_IMA/nnUNet/fold_0/test_set/gt_bin/CAD-PE_083.nii.gz')
# pred, pred_header = read_nifti('D:/Data_stage_IMA/nnUNet/fold_0/inference/CAD-PE_083.nii.gz')
#
##get_confusion_image(pred,gt)
# confusion_image, dice_score = get_confusion_image(pred,gt)
# confusion_affine = get_affine(gt_header)
#
# print(dice_score)
#
#
#
# save_nifti(confusion_image,'D:/Data_stage_IMA/nnUNet/fold_0/confusion_83.nii.gz',confusion_affine)

gt, gt_header = read_nifti("D:/Data_stage_IMA/CAD-PE/gt_middle/inf/751_middle.nii.gz")
pred, pred_header = read_nifti("D:/Data_stage_IMA/nnUNet/fold_0/inference/CAD-PE_084.nii.gz")

# dice_score2d, dice_score3d = get_dice_3D(pred,gt)
# confusion_affine = get_affine(gt_header)
dice_score = dice(pred, gt)
print(dice_score)
# print(dice_score2d)
# print(dice_score3d)


# save_nifti(confusion_image,'D:/Data_stage_IMA/nnUNet/fold_0/confusion_91.nii.gz',confusion_affine)
