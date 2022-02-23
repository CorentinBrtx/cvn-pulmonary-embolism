from typing import List, Tuple

import networkx as nx
import numpy as np
from networkx.algorithms.traversal.edgebfs import edge_bfs


def compute_centers(segmentation: np.ndarray) -> List[Tuple[int]]:
    """Takes in a segmentation and returns the center of each group that can be found"""
    pixel_graph = make_graph_from_seg(segmentation)
    centers = []

    for connected_component in nx.connected_components(pixel_graph):
        subgraph = pixel_graph.subgraph(connected_component)
        outter_nodes = [node for node in subgraph.nodes if subgraph.degree(node) < 4]
        sub_center = list(edge_bfs(subgraph, outter_nodes))[-1][1]
        centers.append(sub_center)

    return centers


DIRS = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]


def make_graph_from_seg(seg: np.ndarray) -> nx.Graph:
    n_pixels = np.prod(seg.shape)
    edgelist = []
    for i in range(n_pixels):
        idx = np.unravel_index(i, seg.shape)
        if seg[idx] != 0:
            for dir in DIRS:
                neigh_idx = tuple(np.array(idx) + dir)
                if (neigh_idx < np.array(seg.shape)).all() and seg[neigh_idx] != 0:
                    edgelist.append((idx, neigh_idx))
    return nx.from_edgelist(edgelist)


if __name__ == "__main__":
    a = np.zeros((10, 10, 10))
    a[:5, :5, :5] = 1
    a[7:10, 7:10, 7:10] = 1
    print(compute_centers(a))
