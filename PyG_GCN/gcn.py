import torch
from torch import nn as nn
import torch.nn.functional as F

import common
import normalization
import encode_process_decode

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

device = torch.device('cuda')


class Model(nn.Module):
    """Model for static cloth simulation."""

    def __init__(self, params, normalizer, in_channels=12, out_channels=3):
        super(Model, self).__init__()
        self.gcn_conv1 = GCNConv(in_channels, out_channels)
        self.gcn_conv2 = GCNConv(out_channels, out_channels)
        # self.edge_conv1 = EdgeConv(in_channels, out_channels)
        # self.edge_conv2 = EdgeConv(out_channels, out_channels)

    def _build_graph(self, inputs, is_training):
        """Builds input graph."""
        world_pos = inputs['world_pos']
        prev_world_pos = inputs['prev|world_pos']
        node_type = inputs['node_type']
        velocity = world_pos - prev_world_pos
        node_type = F.one_hot(node_type[:, 0].to(torch.int64), common.NodeType.SIZE)

        node_features = torch.cat((velocity, node_type), dim=-1)

        cells = inputs['cells']
        senders, receivers = common.triangles_to_edges(cells)

        mesh_pos = inputs['mesh_pos']
        relative_world_pos = (torch.index_select(input=world_pos, dim=0, index=senders) -
                              torch.index_select(input=world_pos, dim=0, index=receivers))
        relative_mesh_pos = (torch.index_select(mesh_pos, 0, senders) -
                             torch.index_select(mesh_pos, 0, receivers))
        edge_features = torch.cat((
            relative_world_pos,
            torch.norm(relative_world_pos, dim=-1, keepdim=True),
            relative_mesh_pos,
            torch.norm(relative_mesh_pos, dim=-1, keepdim=True)), dim=-1)

        mesh_edges = encode_process_decode.EdgeSet(
            name='mesh_edges',
            # features=self._edge_normalizer(edge_features, is_training),
            features=edge_features,
            receivers=receivers,
            senders=senders)
        return encode_process_decode.MultiGraph(
            # node_features=self._node_normalizer(node_features, is_training),
            node_features=node_features,
            edge_sets=[mesh_edges])

    def forward(self, inputs, is_training):
        graph = self._build_graph(inputs, is_training=is_training)
        edge_index = torch.stack((graph.edge_sets[0].receivers, graph.edge_sets[0].senders), dim=0)
        edge_index, _ = add_self_loops(edge_index, num_nodes=graph.node_features.shape[0])
        node_features = graph.node_features

        if is_training:
            node_features = self.gcn_conv1(node_features, edge_index)
            return self.gcn_conv2(node_features, edge_index)
        else:
            node_features = self.gcn_conv1(node_features, edge_index)
            node_features =  self.gcn_conv2(node_features, edge_index)
            return self._update(inputs, node_features)

    def _update(self, inputs, per_node_network_output):
        """Integrate model outputs."""
        # return per_node_network_output

        # acceleration = self._output_normalizer.inverse(per_node_network_output)
        acceleration = per_node_network_output
        # integrate forward
        cur_position = inputs['world_pos']
        prev_position = inputs['prev|world_pos']
        position = 2 * cur_position + acceleration - prev_position
        # print(position)
        return position

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.to(device)

    def evaluate(self):
        self.eval()


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

'''
class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='max')  # "Max" aggregation.
        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)
'''
