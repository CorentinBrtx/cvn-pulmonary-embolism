import os
from typing import Dict, Sequence, Tuple

import nrrd
import nibabel as nib
import numpy as np
from skimage.filters import frangi

from .utils import evaluate_vessels_detection


def save_frangi(
    filename: str,
    target_dir: str,
    sigmas: Sequence[float] = (0.5, 0.8, 1.1, 1.4, 1.8, 2.2),
    suffix="_frangi",
    remove_suffix="",
    force=False
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

    # Computing Frangi
    inverted_data = np.max(data) - data - 1024

    filtered_image = frangi(inverted_data, sigmas=sigmas)
    filtered_image = filtered_image * 1000 / np.max(filtered_image)

    # Writing down our result
    os.makedirs(os.path.dirname(target_filename), exist_ok=True)

    if extension == ".nrrd":
        nrrd.write(target_filename, filtered_image, header)
    else:
        new_image = nib.Nifti1Image(filtered_image, image.affine, image.header)
        nib.save(new_image, target_filename)
    print("Frangi computed")


def save_frangi_seg(frangi_mask_path: str, target_filename: str, threshold: int):

    if os.path.exists(target_filename):
        print(f"{target_filename} already exists, aborting")
        return

    data, header = nrrd.read(frangi_mask_path)
    data = np.where(data > threshold, 1, 0)

    os.makedirs(os.path.dirname(target_filename), exist_ok=True)

    nrrd.write(target_filename, data, header)


def find_best_threshold(
    frangi_masks: Sequence[str],
    lungs_segs: Sequence[str],
    embolisms_segs: Sequence[str],
    thresholds: Sequence[int] = np.linspace(0, 10),
) -> Tuple[Dict[float, Dict], float]:

    best_threshold = 0
    best_score = 0
    results = {}

    for threshold in thresholds:
        current_results = []
        for i, frangi_mask in enumerate(frangi_masks):
            evaluation = evaluate_vessels_detection(
                frangi_mask, lungs_segs[i], embolisms_segs[i], threshold
            )
            current_results.append(evaluation)

        results[threshold] = {}
        for key in current_results[0]:
            results[threshold][key] = np.mean([x[key] for x in current_results])

        if results[threshold]["overall_score"] > best_score:
            best_threshold = threshold
            best_score = results[threshold]["overall_score"]

    return results, best_threshold
