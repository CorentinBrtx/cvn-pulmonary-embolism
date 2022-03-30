import os
from typing import Sequence

import nibabel as nib
import nrrd
import numpy as np
from skimage.filters import frangi


def save_frangi(
    filename: str, target_dir: str, suffix="_frangi", remove_suffix="", force=False
) -> None:

    # Preparing the frangi file's name
    [base_name, extension] = os.path.basename(filename).split(".", 1)
    if remove_suffix:
        base_name = base_name.replace(remove_suffix, "")
    target_filename = os.path.join(target_dir, f"{base_name}{suffix}.{extension}")

    if not force and os.path.exists(target_filename):
        print(f"{target_filename} already exists, skipping")
        return

    # Reading the data
    if extension == ".nrrd":
        header, data = nrrd.read(filename)
    else:
        image = nib.load(filename)
        data = image.get_fdata()

    filtered_image = compute_frangi(image)

    # Computing Frangi
    # Writing down our result
    os.makedirs(os.path.dirname(target_filename), exist_ok=True)

    if extension == ".nrrd":
        nrrd.write(target_filename, filtered_image, header)
    else:
        new_image = nib.Nifti1Image(filtered_image, image.affine, image.header)
        nib.save(new_image, target_filename)
    print("Frangi computed")


def compute_frangi(image, sigmas: Sequence[float] = (0.5, 0.8, 1.1, 1.4, 1.8, 2.2)):
    inverted_image = np.max(image) - image - 1024

    filtered_image = frangi(inverted_image, sigmas=sigmas)
    filtered_image = filtered_image * 1000 / np.max(filtered_image)
    return filtered_image


def save_frangi_seg(frangi_mask_path: str, target_filename: str, threshold: int):

    if os.path.exists(target_filename):
        print(f"{target_filename} already exists, aborting")
        return

    data, header = nrrd.read(frangi_mask_path)
    data = np.where(data > threshold, 1, 0)

    os.makedirs(os.path.dirname(target_filename), exist_ok=True)

    nrrd.write(target_filename, data, header)
