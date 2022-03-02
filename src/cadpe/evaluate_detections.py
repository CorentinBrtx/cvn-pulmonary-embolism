import argparse
from typing import Tuple

import numpy as np
from load_reference_standard import cadpe_load_reference_standard


def cadpe_evaluate_detections(
    detections_path: str,
    ground_truth_folder: str,
    threshold: float,
    gt_prefix: str = "",
    gt_suffix: str = "RefStd.nrrd",
) -> Tuple[float, float, float]:
    """
    Evaluate detections for the CAD_PE challenge.

    Parameters
    ----------
    detections_path : str
        path to the text file containing the detections
    ground_truth_folder : str
        folder containing the ground truth nrrd files
    threshold : float
        at which confidence score the detections should be computed
    gt_prefix : str, optional
        prefix for the ground truth files, by default ""
    gt_suffix : str, optional
        suffix for the ground truth files, by default "RefStd.nrrd"

    Returns
    -------
    ss, fps, ppv: Tuple[float, float, float]
        ss: sensitivity: true positives/all positives
        fps: average number of false positives per scan
        ppv: positive predictive value
    """

    with open(detections_path, "r") as f:
        detections = np.array(list(map(lambda x: x.split(" "), f.readlines())))

    rs, total_clots = cadpe_load_reference_standard(
        ground_truth_folder, file_prefix=gt_prefix, file_suffix=gt_suffix
    )

    nb_cases = len(rs.keys())
    det_clots = np.zeros(len(detections), dtype=np.int64)

    for idx, det in enumerate(detections):
        x, y, z = int(det[1]), int(det[2]), int(det[3])
        det_clots[idx] = rs[det[0]]["mask"][x, y, z]

    # Removes detections below the threshold
    idx_thresh_elim = detections[:, 4] < threshold
    det_clots[idx_thresh_elim] = -1

    # See how many clots have been detected per volume
    clots_detected = np.zeros(nb_cases)
    false_positives = np.zeros(nb_cases)
    false_positives_idx = set(np.where(det_clots == 0)[0])

    for i, case_id in enumerate(rs.keys()):
        idx_case = set(np.where(detections[:, 0] == case_id)[0])
        clots_labels_detected = set(det_clots[idx_case])

        # Gets the clots that have been detected
        total_clots_detected = 0
        for i in range(1, rs[case_id]["n_clots"] + 1):
            if i in clots_labels_detected:
                total_clots_detected += 1
        clots_detected[i] = total_clots_detected

        # Obtains the false positives in the volume
        false_positives[i] = len(false_positives_idx & idx_case)

    ss = np.sum(clots_detected) / total_clots
    fps = np.sum(false_positives) / nb_cases
    ppv = np.sum(clots_detected) / (np.sum(false_positives) + np.sum(clots_detected))

    return ss, fps, ppv


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate CAD_PE detections")

    parser.add_argument(
        "detections_path", type=str, help="path to the text file containing the detections"
    )
    parser.add_argument(
        "ground_truth_folder", type=str, help="folder containing the ground truth nrrd files"
    )
    parser.add_argument(
        "threshold", type=float, help="at which confidence score the detections should be computed"
    )
    parser.add_argument(
        "--gt_prefix", type=str, default="", help="prefix for the ground truth files"
    )
    parser.add_argument(
        "--gt_suffix", type=str, default="RefStd.nrrd", help="suffix for the ground truth files"
    )

    args = parser.parse_args()

    cadpe_evaluate_detections(
        args.detections_path,
        args.ground_truth_folder,
        args.threshold,
        args.gt_prefix,
        args.gt_suffix,
    )
