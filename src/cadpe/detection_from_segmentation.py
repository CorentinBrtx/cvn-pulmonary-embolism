from typing import List, Tuple

import numpy as np
from scipy.ndimage.measurements import label
from skimage.morphology import binary_erosion


def compute_depth_map(segmentation: np.ndarray) -> np.ndarray:
    depth = np.ones_like(segmentation)
    while np.max(segmentation) > 0:
        segmentation = binary_erosion(segmentation)
        depth += segmentation
    return depth


def compute_centers(segmentation: np.ndarray, min_depth: int = 0) -> List[Tuple[int, int, int]]:
    """Takes in a segmentation and returns the center of each region that can be found"""
    labeled_segmentation, n_regions = label(segmentation, np.ones((3, 3, 3)))
    depth_map = compute_depth_map(segmentation)
    centers = []
    for i in range(1, n_regions + 1):
        flat_idx_max = np.argmax(depth_map * (labeled_segmentation == i))
        idx_max = np.unravel_index(flat_idx_max, depth_map.shape)
        if depth_map[idx_max] > min_depth:
            centers.append(idx_max)
    return centers


if __name__ == "__main__":
    a = np.zeros((10, 10, 10))
    a[:5, :5, :5] = 1
    a[7:10, 7:10, 7:10] = 1
    print(compute_centers(a))
