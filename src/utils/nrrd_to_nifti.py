import os
from argparse import ArgumentParser

from nrrd_utils import nrrd_to_nifti
from tqdm import tqdm

parser = ArgumentParser(description="Transform a nrrd file into a nifti file")

parser.add_argument("-r", "--recursive", action="store_true", help="Recursive search")
parser.add_argument(
    "filename",
    help="path to the original nrrd file. Can be a directory if the -r flag is set.",
)
parser.add_argument(
    "target_filename",
    help="path to the target nifti file. Can be a directory if the -r flag is set.",
)
args = parser.parse_args()

if args.recursive:
    for file in tqdm(os.listdir(args.filename)):
        if file.endswith(".nrrd"):
            nrrd_to_nifti(
                os.path.join(args.filename, file),
                os.path.join(args.target_filename, file.replace(".nrrd", ".nii.gz")),
            )

else:
    nrrd_to_nifti(args.filename, args.target_filename)
