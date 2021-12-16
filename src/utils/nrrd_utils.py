"""Useful functions to deal with nrrd files."""

import os

import nibabel as nib
import nrrd
import numpy as np


def nrrd_get_data(filename: str) -> np.ndarray:
    """Get the data from a nrrd file"""
    data, _ = nrrd.read(filename)
    return data


def nrrd_to_nifti(filename: str, target_filename: str) -> None:
    """Transform a nrrd file into a nifti file"""

    data, header = nrrd.read(filename)

    affine = np.zeros((4, 4), dtype="float")
    affine[0:3, 0:3] = -header["space directions"]
    affine[3, 3] = 1
    affine[2, 2] = -affine[2, 2]

    affine[0:3, 3] = -header["space origin"]
    affine[2, 3] = -affine[2, 3]

    img = nib.Nifti1Image(data, affine)

    os.makedirs(os.path.dirname(target_filename), exist_ok=True)

    nib.save(img, target_filename)
