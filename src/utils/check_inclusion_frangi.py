import os
from argparse import ArgumentParser
from typing import List, Optional

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from tqdm import tqdm


def compute_threshold_effects(
    frangi: np.ndarray,
    ground_truth: np.ndarray,
    n_thresholds: int = 10,
    thresholds: Optional[List[float]] = None,
):
    gt_sum = np.sum(ground_truth)
    frangi = frangi / np.max(frangi)
    product = frangi * ground_truth
    inclusion, covering = [], []
    if thresholds is None:
        thresholds = np.arange(1, n_thresholds + 1) / n_thresholds
    for threshold in thresholds:
        inclusion.append(np.sum(product > threshold) / gt_sum)
        covering.append(
            np.sum(frangi > threshold) / (frangi.shape[0] * frangi.shape[1] * frangi.shape[2])
        )
    return inclusion, covering


def plot_save_effects(
    thresholds, inclusion, covering, filename, inclusion_std=None, covering_std=None
):
    plt.figure()
    plt.plot(thresholds, inclusion, label="PE included in Frangi")
    if inclusion_std is not None:
        plt.fill_between(
            thresholds, inclusion - inclusion_std, inclusion + inclusion_std, alpha=0.5
        )
    plt.plot(thresholds, covering, label="Image included in Frangi")
    if covering_std is not None:
        plt.fill_between(thresholds, covering - covering_std, covering + covering_std, alpha=0.5)
    plt.xlabel("Threshold")
    plt.ylabel("Proportion of pixels")
    plt.legend()
    plt.savefig(filename)


if __name__ == "__main__":
    parser = ArgumentParser(description="Check if Frangi intersects well with the ground truth")

    parser.add_argument("--gt_folder", help="Ground Truth directory name")
    parser.add_argument("--frangi_folder", help="Frangi vesselness files directory name")
    parser.add_argument(
        "--destination", help="Name of the file to output the plottings of the threshlds effects"
    )
    parser.add_argument(
        "--add_summary", action="store_true", help="Also saves a summary of all the plots"
    )
    parser.add_argument(
        "--only_summary", action="store_true", help="Only saves a summary of all the plots"
    )
    parser.add_argument(
        "--n_thresholds", help="The number of thresholds to test", default=10, type=int
    )
    parser.add_argument("--thresholds", nargs="*", help="The thresholds to test")
    args = parser.parse_args()

    os.makedirs(args.destination, exist_ok=True)
    if args.thresholds == []:
        thresholds = np.arange(1, args.n_thresholds + 1) / args.n_thresholds
    else:
        thresholds = [float(thresh) for thresh in args.thresholds]

    print("thresholds: ", thresholds)

    threshold_inclusion, threshold_covering = [], []
    for gt_filename in tqdm(os.listdir(args.gt_folder)):
        image_name = os.path.basename(gt_filename).split(".", 1)[0]

        ground_truth = nib.load(os.path.join(args.gt_folder, gt_filename))
        ground_truth_data = ground_truth.get_fdata()

        frangi = nib.load(os.path.join(args.frangi_folder, image_name + "_frangi.nii.gz"))
        frangi_data = frangi.get_fdata()

        inclusion, covering = compute_threshold_effects(
            frangi_data, ground_truth_data, thresholds=thresholds
        )
        threshold_inclusion.append(inclusion)
        threshold_covering.append(covering)
        if not args.only_summary:
            plot_save_effects(
                thresholds, inclusion, covering, os.path.join(args.destination, image_name + ".png")
            )

        if not np.isfinite(threshold_inclusion[-1]).all():
            threshold_inclusion.pop()
            threshold_covering.pop()

    if args.only_summary or args.add_summary:
        mean_inclusion = np.mean(np.array(threshold_inclusion), 0)
        std_inclusion = np.std(np.array(threshold_inclusion), 0)
        mean_covering = np.mean(np.array(threshold_covering), 0)
        std_covering = np.std(np.array(threshold_covering), 0)
        plot_save_effects(
            thresholds,
            mean_inclusion,
            mean_covering,
            os.path.join(args.destination, "mean.png"),
            std_inclusion,
            std_covering,
        )
