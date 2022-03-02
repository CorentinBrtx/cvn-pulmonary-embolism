import os
from argparse import ArgumentParser

from nrrd_utils import nrrd_normalize
from tqdm import tqdm

parser = ArgumentParser(description="Normalize a nrrd file segmentation (keep only one class)")

parser.add_argument("-r", "--recursive", action="store_true", help="Recursive search")
parser.add_argument(
    "filename",
    help="path to the original nrrd file. Can be a directory if the -r flag is set.",
)
args = parser.parse_args()

if args.recursive:
    for file in tqdm(os.listdir(args.filename)):
        if file.endswith(".nrrd"):
            nrrd_normalize(
                os.path.join(args.filename, file),
            )

else:
    nrrd_normalize(args.filename)
