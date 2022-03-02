import os
from argparse import ArgumentParser

import nibabel as nib
from tqdm import tqdm

from src.utils.detection_from_segmentation import compute_centers

parser = ArgumentParser(description="Compute embolism centers from the nnunet masks")

# parser.add_argument(
#     "--dirs", nargs='+', help="List of directories where the input files are situated"
# )
parser.add_argument(
    "--dest",
    help="Where to write the result file",
)
parser.add_argument(
    "--mode",
    default="semi-smart",
    help="The mode with which we select the centers",
)
parser.add_argument(
    "--centers",
    default=5,
    type=int,
    help="The number of centers we want to have for each embolism",
)
args = parser.parse_args()

dirs = [
    "/workdir/shared/pulmembol/nnUNet/nnUNet_results/Task501_EmbolismCADPE/",
    *[
        "../pulmembol_workdir/nnUNet/nnUNet_trained_models/nnUNet/3d_fullres/"
        f"Task501_EmbolismCADPE/nnUNetTrainerV2__nnUNetPlansv2.1/fold_{i}/validation_raw"
        for i in range(5)
    ],
]

all_filenames = []
for dir_name in dirs:
    all_filenames += [
        os.path.join(dir_name, filename)
        for filename in os.listdir(dir_name)
        if filename.endswith(".nii.gz")
    ]

res = []
for filename in tqdm(all_filenames):
    segmentation = nib.load(filename).get_fdata()
    res += [
        f"{os.path.basename(filename)} {center[0]} {center[1]} {center[2]} 1\n"
        for center in compute_centers(segmentation, args.mode, args.centers)
    ]

with open(args.dest, "w") as dest_file:
    dest_file.writelines(res)
