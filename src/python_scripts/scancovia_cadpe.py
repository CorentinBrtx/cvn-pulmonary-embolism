import getopt
import os
import sys

from src.scancovia import save_lungs_seg

#######################################################################
# Choose a device to work on
try:
    opts, args = getopt.getopt(sys.argv[1:], "g")
except getopt.GetoptError as err:
    print("unhandled options")
    sys.exit(2)

gpu = 0
for o, a in opts:
    if o == "-g":
        gpu = 1
    else:
        assert False, "unhandled option"

if gpu == 0:
    device = "cpu"
else:
    device = "cuda"

print(f"Device used: {device}")

#######################################################################
# Run Scancovia on CAD_PE dataset

WORKDIR = os.getenv("WORKDIR")
ARRAY_ID = int(os.getenv("SLURM_ARRAY_TASK_ID"))

file = os.listdir(os.path.join(WORKDIR, "CAD_PE/images"))[ARRAY_ID]
if file.endswith(".nrrd"):
    print(f"Processing {file}")
    save_lungs_seg(
        os.path.join(WORKDIR, "CAD_PE/images", file),
        os.path.join(
            WORKDIR, "CAD_PE/lungs_scancovia", os.path.splitext(file)[0] + "_lungs.seg.nrrd"
        ),
        device=device,
    )
