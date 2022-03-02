"""Useful functions to deal with nrrd files."""

import os

import SimpleITK as sitk
import nrrd
import numpy as np


def nrrd_get_data(filename: str) -> np.ndarray:
    """Get the data from a nrrd file"""
    data, _ = nrrd.read(filename)
    return data


def nrrd_normalize(filename: str) -> None:
    """Normalize a nrrd file"""

    data, header = nrrd.read(filename)
    data = np.where(data > 0, 1.0, 0.0)
    nrrd.write(filename, data, header)


def nrrd_to_nifti(filename: str, target_filename: str) -> None:
    """Transform a nrrd file into a nifti file"""

    img = sitk.ReadImage(filename)

    os.makedirs(os.path.dirname(target_filename), exist_ok=True)

    sitk.WriteImage(img, target_filename)
