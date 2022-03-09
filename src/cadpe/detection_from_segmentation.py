from random import sample
from typing import Any, List, Tuple

import networkx as nx
import numpy as np
from scipy.ndimage.measurements import label
from skimage.feature import peak_local_max
from skimage.morphology import binary_erosion, skeletonize_3d


def depth_skeleton(segmentation: np.array) -> np.array:
    depth = np.ones_like(segmentation)
    while np.max(segmentation) > 0:
        segmentation = binary_erosion(segmentation)
        depth += segmentation
    return depth


def compute_centers(
    segmentation: np.ndarray, mode: str = "random", n_center_per_region: int = 5
) -> List[Tuple[int]]:
    """Takes in a segmentation and returns the center of each region that can be found"""
    centers = []
    indices = np.indices(segmentation.shape).transpose(1, 2, 3, 0)  # 3D image
    if mode in ["random", "semi-smart"]:
        skeleton = skeletonize_3d(segmentation)
        labeled_skeleton, n_regions = label(skeleton, np.ones((3, 3, 3)))
        for i in range(1, n_regions + 1):
            if mode == "random":
                centers += list(sample(indices[labeled_skeleton == i], n_center_per_region))
            else:
                centers += search_skeleton(indices[labeled_skeleton == i], n_center_per_region)

    elif mode == "smart":
        skeleton = depth_skeleton(segmentation)
        centers = peak_local_max(skeleton, threshold_abs=4)
    return centers


def search_skeleton(skel_indices: np.ndarray, n_centers: int) -> List[Tuple[Any, ...]]:
    """Go through the skeleton with a DFS and samples the centers as it goes"""
    skel_graph = graph_from_skeleton(skel_indices)
    if len(skel_indices) < 5:
        return []
    elif len(skel_indices) < 10:
        ordered_nodes = list(
            nx.dfs_preorder_nodes(
                skel_graph, [node for node in skel_graph.nodes if skel_graph.degree(node) == 1][0]
            )
        )
        return [ordered_nodes[len(skel_indices) // 2]]
    else:
        ordered_nodes = list(nx.dfs_preorder_nodes(skel_graph))
        return ordered_nodes[:: len(skel_indices) // (n_centers + 1)][1 : n_centers + 1]


def graph_from_skeleton(skel_indices: np.ndarray) -> nx.Graph:
    edgelist = []
    for i in range(len(skel_indices)):
        for j in range(i + 1, len(skel_indices)):
            if np.max(np.abs(skel_indices[i] - skel_indices[j])) == 1:
                edgelist.append((tuple(skel_indices[i]), tuple(skel_indices[j])))
    graph = nx.from_edgelist(edgelist)
    return graph


if __name__ == "__main__":
    a = np.zeros((10, 10, 10))
    a[:5, :5, :5] = 1
    a[7:10, 7:10, 7:10] = 1
    print(compute_centers(a))
