import os
from argparse import ArgumentParser
from multiprocessing import Pool

import nibabel as nib
from tqdm import tqdm

from src.cadpe.detection_from_segmentation import compute_centers

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
parser.add_argument(
    "--n_processes",
    default=1,
    type=int,
    help="The number of processes we want to run in parallel",
)
args = parser.parse_args()

dirs = [
    "/workdir/shared/pulmembol/nnUNet/nnUNet_results/Task501_EmbolismCADPE/predictionsTs",
    "/gpfs/users/prevotb/pulmembol_workdir/nnUNet/nnUNet_trained_models/nnUNet/ensembles/"
    "Task501_EmbolismCADPE/ensemble_3d_fullres__nnUNetTrainerV2__nnUNetPlansv2."
    "1--3d_cascade_fullres__nnUNetTrainerV2CascadeFullRes__nnUNetPlansv2.1/ensembled_postprocessed",
]

all_filenames = []
for dir_name in dirs:
    all_filenames += [
        os.path.join(dir_name, filename)
        for filename in os.listdir(dir_name)
        if filename.endswith(".nii.gz")
    ]

print("Compute centers")


def centers_from_fname(fname):
    segmentation = nib.load(fname).get_fdata()
    return fname, compute_centers(segmentation, args.mode, args.centers)


pool = Pool(args.n_processes)
fnames_n_center_lists = pool.imap_unordered(
    centers_from_fname,
    all_filenames,
)

pool.close()

res = []
for fname, center_list in tqdm(fnames_n_center_lists, total=len(all_filenames)):
    for center in center_list:
        res += [f"{os.path.basename(fname)} {center[0]} {center[1]} {center[2]} 1\n"]

print("Write down results")
with open(args.dest, "w") as dest_file:
    dest_file.writelines(res)

print("All done")
