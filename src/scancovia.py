import os

import nrrd
from scancovia import AiSegment

from .utils import nrrd_to_nifti


def save_lungs_seg(filename: str, target_filename: str, device: str = "cpu"):

    _, header = nrrd.read(filename)

    nrrd_to_nifti(filename, os.path.splitext(filename)[0] + ".nii.gz")
    filename = filename.replace(".nrrd", ".nii.gz")

    ai_segment = AiSegment(device=device)
    output = ai_segment(filename)

    if not os.path.exists(os.path.dirname(target_filename)):
        os.makedirs(os.path.dirname(target_filename))

    nrrd.write(target_filename, output["lungs_mask"].transpose(1, 0, 2), header)

    os.remove(filename)

    return output
