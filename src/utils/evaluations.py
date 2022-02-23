from typing import Dict

import numpy as np

from .nrrd_utils import nrrd_get_data


def evaluate_vessels_detection(
    vessels_seg_path: str, lungs_seg_path: str, embolisms_seg_path: str, threshold: int = None
) -> Dict:

    embolisms = np.where(nrrd_get_data(embolisms_seg_path) > 0, 1, 0)
    lungs = np.where(nrrd_get_data(lungs_seg_path) > 0, 1, 0)

    if threshold is not None:
        vessels = np.where(nrrd_get_data(vessels_seg_path) > threshold, 1, 0)
    else:
        vessels = np.where(nrrd_get_data(vessels_seg_path) > 0, 1, 0)

    lungs_proportion = np.sum(vessels * lungs) / np.sum(lungs)
    embolisms_recall = np.sum(embolisms * vessels * lungs) / np.sum(embolisms)

    return {
        "lungs_proportion": lungs_proportion,
        "embo_recall": embolisms_recall,
        "overall_score": embolisms_recall / lungs_proportion,
    }
