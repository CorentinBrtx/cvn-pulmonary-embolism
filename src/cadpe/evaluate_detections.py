import argparse
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from src.cadpe.load_reference_standard import cadpe_load_reference_standard


def get_detected_clot(center: Tuple[int, int, int], max_distance: float, mask: np.ndarray) -> int:
    detected_clot = 0
    min_distance = np.inf
    for x in range(center[0] - int(max_distance), center[0] + int(max_distance) + 1):
        for y in range(center[1] - int(max_distance), center[1] + int(max_distance) + 1):
            for z in range(center[2] - int(max_distance), center[2] + int(max_distance) + 1):
                if (
                    (0 <= x < mask.shape[0])
                    and (0 <= y < mask.shape[1])
                    and (0 <= z < mask.shape[2])
                    and mask[x, y, z] > 0
                ):
                    if np.linalg.norm(np.array([x, y, z]) - center) <= max_distance:
                        if np.linalg.norm(np.array([x, y, z]) - center) < min_distance:
                            min_distance = np.linalg.norm(np.array([x, y, z]) - center)
                            detected_clot = mask[x, y, z]
    return detected_clot


def cadpe_evaluate_detections(
    detections_path: str,
    ground_truth_folders: List[str],
    threshold: float,
    epsilon: int = 0,
    gt_prefix: str = "",
    gt_suffix: str = "RefStd.nii.gz",
    target_file: str = "evaluation.txt",
) -> Tuple[float, float, float]:
    """
    Evaluate detections for the CAD_PE challenge.

    Parameters
    ----------
    detections_path : str
        path to the text file containing the detections
    ground_truth_folder : str
        folder containing the ground truth nifti files
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
        ground_truth_folders, file_prefix=gt_prefix, file_suffix=gt_suffix
    )

    nb_cases = len(rs.keys())
    det_clots = np.zeros(len(detections), dtype=np.int64)

    for idx, det in tqdm(enumerate(detections)):
        x, y, z = int(det[1]), int(det[2]), int(det[3])
        if det[0] in rs:
            det_clots[idx] = get_detected_clot(
                (x, y, z), epsilon / np.mean(rs[det[0]]["voxels_sizes"]), rs[det[0]]["mask"]
            )

    # Removes detections below the threshold
    idx_thresh_elim = detections[:, 4].astype(np.float64) < threshold
    det_clots[idx_thresh_elim] = -1

    # See how many clots have been detected per volume
    clots_detected = np.zeros(nb_cases)
    false_positives = np.zeros(nb_cases)
    false_positives_idx = set(np.where(det_clots == 0)[0])

    results = ""

    for i, case_id in enumerate(rs.keys()):
        idx_case = np.nonzero(detections[:, 0] == case_id)[0]
        clots_labels_detected = set(det_clots[idx_case])

        # Gets the clots that have been detected
        total_clots_detected = 0
        for j in range(1, rs[case_id]["n_clots"] + 1):
            if j in clots_labels_detected:
                total_clots_detected += 1
        clots_detected[i] = total_clots_detected

        # Obtains the false positives in the volume
        false_positives[i] = len(false_positives_idx & set(idx_case))

        results += f"Case {case_id}: {total_clots_detected} / {rs[case_id]['n_clots']}\n"

    ss = np.sum(clots_detected) / total_clots
    fps = np.sum(false_positives) / nb_cases
    if np.sum(false_positives) + np.sum(clots_detected) == 0:
        ppv = 0
    else:
        ppv = np.sum(clots_detected) / (np.sum(false_positives) + np.sum(clots_detected))

    with open(target_file, "w") as f:
        f.write(f"Results for a tolerance of {epsilon}mm.\n\n")
        f.write(f"Sensitivity: {ss}\n")
        f.write(f"False positives per sample: {fps}\n")
        f.write(f"Positive predictive value: {ppv}\n\n")
        f.write(results)

    return ss, fps, ppv


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate CAD_PE detections")

    parser.add_argument(
        "--detections_path", type=str, help="path to the text file containing the detections"
    )
    parser.add_argument(
        "--ground_truth_folders",
        type=str,
        nargs="+",
        help="folders containing the ground truth nifti files",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0,
        help="at which confidence score the detections should be computed",
    )
    parser.add_argument(
        "--gt_prefix", type=str, default="", help="prefix for the ground truth files"
    )
    parser.add_argument(
        "--gt_suffix", type=str, default="RefStd.nii.gz", help="suffix for the ground truth files"
    )
    parser.add_argument(
        "--target_file", type=str, default="evaluation.txt", help="file to store the results"
    )
    parser.add_argument(
        "--epsilon", type=int, default=0, help="tolerance in mm"
    )

    args = parser.parse_args()

    cadpe_evaluate_detections(
        detections_path=args.detections_path,
        ground_truth_folders=args.ground_truth_folders,
        threshold=args.threshold,
        gt_prefix=args.gt_prefix,
        gt_suffix=args.gt_suffix,
        target_file=args.target_file,
        epsilon=args.epsilon
    )
