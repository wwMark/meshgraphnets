import numpy as np
import scipy.spatial as spatial
import torch

def find_influential_nodes(input, radius, topk):

    nodes = input.cpu()
    node_tree = spatial.cKDTree(nodes)
    indices = node_tree.query_ball_point(nodes, radius)
    neighbors = list(map(sum, indices))
    neighbors = torch.tensor(neighbors)
    topk_values, topk_indices = torch.topk(neighbors, topk)

    return topk_indices
