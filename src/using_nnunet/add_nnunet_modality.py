import os
from argparse import ArgumentParser

from compute_modality import compute_and_save_modality

ARRAY_ID = int(os.getenv("SLURM_ARRAY_TASK_ID"))
ORIGINAL_SUFFIX = "_0000.nii.gz"


parser = ArgumentParser(
    description="Compute Frangi vessellness for all the files in given input_dir"
)

parser.add_argument("--dir", help="Directory where the input files are situated (*_0000)")
parser.add_argument(
    "--suffix",
    help="The suffix that will be added to the input file's name in order to produce the output file's name",
    default="_0001",
)
parser.add_argument(
    "--force",
    help="Forces the output files to be recomputed if a file already exists",
    action="store_true",
)
args = parser.parse_args()


all_file_names = [
    file_name for file_name in os.listdir(args.dir) if file_name.endswith(ORIGINAL_SUFFIX)
]
if ARRAY_ID >= len(all_file_names):
    print("too many jobs for this job")
    quit()

file_name = os.path.join(args.dir, all_file_names[ARRAY_ID])
print(f"Processing {file_name}")

dest_file_name = file_name.replace(ORIGINAL_SUFFIX, args.suffix + ".nii.gz")
compute_and_save_modality(file_name, dest_file_name, force=args.force)
