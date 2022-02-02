import os
import re

from argparse import ArgumentParser
from src.frangi import save_frangi

#######################################################################
# Run Frangi on CAD_PE dataset

WORKDIR = os.getenv("WORKDIR")
ARRAY_ID = int(os.getenv("SLURM_ARRAY_TASK_ID"))

parser = ArgumentParser(description="Compute Frangi vessellness for all the files in given input_dir")

parser.add_argument(
    "--input_dir", help="Directory where the input files are", default=os.path.join(WORKDIR, "CAD_PE/images")
)
parser.add_argument(
    "--output_dir", help="Directory where the output files are to be written", default=os.path.join(WORKDIR, "CAD_PE/frangi")
)
parser.add_argument(
    "--input_regex", help="regex to identify input files", default=""
)
parser.add_argument(
    "--remove_suffix", help="Suffix that should me removed from the input file's name in order to produce the frangi file's name", default=""
)
parser.add_argument(
    "--suffix", help="The suffix that will be added to the input file's name in order to produce the frangi file's name", default="_frangi"
)
parser.add_argument(
    "--force", help="Forces the frangi vesselness to be recomputed if a file already exists", action="store_true"
)
args = parser.parse_args()


all_file_names = os.listdir(args.input_dir)
if ARRAY_ID >= len(all_file_names):
    print ("too many jobs for this job")
    quit()
file_name = all_file_names[ARRAY_ID]

correct_extension = file_name.split(".", 1)[1] in ["nrrd", "nii.gz"]
correct_name = (not args.input_regex or re.match(args.input_regex, file_name))

if correct_extension and correct_name:
    print(f"Processing {file_name}")
    save_frangi(
        os.path.join(args.input_dir, file_name),
        args.output_dir,
        suffix=args.suffix,
        remove_suffix=args.remove_suffix,
        force=args.force
    )
