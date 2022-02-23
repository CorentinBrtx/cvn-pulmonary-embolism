import os
from typing import Any, Dict, Tuple

from src.utils import nrrd_get_data


def cadpe_load_reference_standard(
    folder: str, file_prefix: str = "", file_suffix: str = "RefStd.nrrd"
) -> Tuple[Dict[str, Dict[str, Any]], int]:

    rs: Dict[str, Dict[str, Any]] = {}
    total_clots = 0

    for filename in os.listdir(folder):
        if filename.startswith(file_prefix) and filename.endswith(file_suffix):
            data = nrrd_get_data(filename)
            n_clots = data.max()
            total_clots += n_clots
            rs[filename.removeprefix(file_prefix).removesuffix(file_suffix)]["n_clots"] = n_clots
            rs[filename.removeprefix(file_prefix).removesuffix(file_suffix)]["mask"] = data

    return rs, total_clots
