import os
from typing import Any, Dict, List, Tuple

import nibabel as nib
from tqdm import tqdm


def cadpe_load_reference_standard(
    folders: List[str], file_prefix: str = "", file_suffix: str = "RefStd.nii.gz"
) -> Tuple[Dict[str, Dict[str, Any]], int]:

    rs: Dict[str, Dict[str, Any]] = {}
    total_clots = 0

    files = [
        os.path.join(folder, file)
        for folder in folders
        for file in os.listdir(folder)
        if file.startswith(file_prefix) and file.endswith(file_suffix)
    ]

    for file in tqdm(files):
        name = os.path.basename(file)
        name = name.replace(file_suffix, "")
        name = name.replace(file_prefix, "")
        img = nib.load(file)
        data = img.get_data()
        header = img.header
        n_clots = int(data.max())
        rs[name] = {"mask": data, "n_clots": n_clots, "voxels_sizes": header.get_zooms()}
        total_clots += n_clots

    return rs, total_clots
