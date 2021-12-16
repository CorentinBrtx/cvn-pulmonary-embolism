import os

from src.frangi import save_frangi

#######################################################################
# Run Frangi on CAD_PE dataset

WORKDIR = os.getenv("WORKDIR")
ARRAY_ID = int(os.getenv("SLURM_ARRAY_TASK_ID"))

file = os.listdir(os.path.join(WORKDIR, "CAD_PE/images"))[ARRAY_ID]
if file.endswith(".nrrd"):
    print(f"Processing {file}")
    save_frangi(
        os.path.join(WORKDIR, "CAD_PE/images", file),
        os.path.join(WORKDIR, "CAD_PE/frangi", os.path.splitext(file)[0] + "_frangi.nrrd"),
    )
