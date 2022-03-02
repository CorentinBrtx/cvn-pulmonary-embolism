from random import sample
from typing import Any, List, Tuple, Union

import nibabel as nib
import networkx as nx
import numpy as np
from scipy.ndimage.measurements import label
from skimage.morphology import skeletonize_3d


def compute_centers(
    segmentation: Union[str, np.ndarray], mode: str = "random", n_center_per_region: int = 5
) -> List[Tuple[int]]:
    """Takes in a segmentation and returns the center of each region that can be found"""
    if type(segmentation) == str: # it's the filename
        print("Need to load data")
        segmentation = nib.load(segmentation).get_fdata()
        print("Loaded data")

    centers = []
    indices = np.indices(segmentation.shape).transpose(1, 2, 3, 0)  # 3D image
    if mode in ["random", "semi-smart"]:
        print("Compute skeleton")
        skeleton = skeletonize_3d(segmentation)
        print("Compute regions")
        labeled_skeleton, n_regions = label(skeleton, np.ones((3, 3, 3)))
<<<<<<< Updated upstream
=======
        print("Going over regions")
>>>>>>> Stashed changes
        for i in range(1, n_regions + 1):
            if mode == "random":
                centers += list(sample(indices[labeled_skeleton == i], n_center_per_region))
            else:
                centers += search_skeleton(indices[labeled_skeleton == i], n_center_per_region)

    elif mode == "smart":
        # use skimage.morphology.medial_axis on each layer
        # in order to get the distance from each border
        pass
    print("Computed centers")
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
