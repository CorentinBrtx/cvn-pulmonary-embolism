import json
import os
from argparse import ArgumentParser
from collections import defaultdict
from typing import Dict

import numpy as np

#######################################################################
# Gather all metrics from different folds and test folders


def get_overall_metrics(data_folder: str) -> Dict:

    folds = [folder for folder in os.listdir(data_folder) if folder.startswith("fold")]

    metrics = defaultdict(list)

    for fold in folds:
        with open(os.path.join(fold, "validation_raw_postprocessed/summary.json"), "r") as f:
            metrics_fold = json.load(f)

        for result in metrics_fold["results"]["all"]:
            for key in result["1"]:
                metrics[key].append(result["1"][key])

    mean_metrics = {key: np.mean(metrics[key]) for key in metrics}

    return mean_metrics


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("results_folder", help="Folder containing the different folds folders")
    parser.add_argument("target_filename", help="Name of the file to output the mean metrics")
    args = parser.parse_args()

    overall_metrics = get_overall_metrics(args.results_folder)

    os.makedirs(args.target_filename, exist_ok=True)
    with open(args.target_filename, "w") as f:
        json.dump(overall_metrics, f, indent=4)
