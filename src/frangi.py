import os
from typing import Sequence

import nrrd
import numpy as np
from skimage.filters import frangi


def save_frangi(
    filename: str,
    target_filename: str,
    sigmas: Sequence[float] = (0.5, 0.8, 1.1, 1.4, 1.8, 2.2),
) -> None:

    if os.path.exists(target_filename):
        print(f"{target_filename} already exists, skipping")
        return

    data, header = nrrd.read(filename)
    inverted_data = np.max(data) - data - 1024

    filtered_image = frangi(inverted_data, sigmas=sigmas)
    filtered_image = filtered_image * 1000 / np.max(filtered_image)

    os.makedirs(os.path.dirname(target_filename), exist_ok=True)

    nrrd.write(target_filename, filtered_image, header)
