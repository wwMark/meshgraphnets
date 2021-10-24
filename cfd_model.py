# Lint as: python3
# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Model for CylinderFlow."""

import common
import normalization

import torch
import torch.nn as nn
import torch.nn.functional as F
import encode_process_decode
import encode_process_decode_max_pooling
import encode_process_decode_lstm
import encode_process_decode_graph_structure_watcher
import encode_process_decode_hub

device = torch.device('cuda')
core_name = encode_process_decode


# core_name = encode_process_decode

class Model(nn.Module):
    """Model for fluid simulation."""

    def __init__(self, params):
        super(Model, self).__init__()
        self._params = params
        self._output_normalizer = normalization.Normalizer(size=2, name='output_normalizer')
        self._node_normalizer = normalization.Normalizer(size=2 + common.NodeType.SIZE, name='node_normalizer')
        self._edge_normalizer = normalization.Normalizer(size=3, name='edge_normalizer')  # 2D coord + length

        self.message_passing_steps = 7
        self.learned_model = core_name.EncodeProcessDecode(
            output_size=params['size'],
            latent_size=128,
            num_layers=2,
            message_passing_steps=self.message_passing_steps)
        self.core_model = str(core_name)

    def get_core_model_name(self):
        return repr(self.core_model)

    def get_message_passing_steps(self):
        return str(self.message_passing_steps)

    def _build_graph(self, inputs, is_training):
        """Builds input graph."""
        node_type = inputs['node_type']
        velocity = inputs['velocity']
        node_type = F.one_hot(node_type[:, 0].to(torch.int64), common.NodeType.SIZE)

        node_features = torch.cat((velocity, node_type), dim=-1)

        senders, receivers = common.triangles_to_edges(inputs['cells'])
        mesh_pos = inputs['mesh_pos']
        relative_mesh_pos = (torch.index_select(mesh_pos, 0, senders) -
                             torch.index_select(mesh_pos, 0, receivers))
        edge_features = torch.cat([
            relative_mesh_pos,
            torch.norm(relative_mesh_pos, dim=-1, keepdim=True)], dim=-1)

        mesh_edges = core_name.EdgeSet(
            name='mesh_edges',
            features=self._edge_normalizer(edge_features, is_training),
            receivers=receivers,
            senders=senders)
        return core_name.MultiGraph(
            node_features=self._node_normalizer(node_features, is_training),
            edge_sets=[mesh_edges])

    def forward(self, inputs, is_training):
        graph = self._build_graph(inputs, is_training=is_training)
        if is_training:
            return self.learned_model(graph)
        else:
            return self._update(inputs, self.learned_model(graph))

    def _update(self, inputs, per_node_network_output):
        """Integrate model outputs."""
        velocity_update = self._output_normalizer.inverse(per_node_network_output)
        # integrate forward
        cur_velocity = inputs['velocity']
        return cur_velocity + velocity_update

    def get_output_normalizer(self):
        return self._output_normalizer

    def save_model(self, path):
        torch.save(self.learned_model, path + "_learned_model.pth")
        torch.save(self._output_normalizer, path + "_output_normalizer.pth")
        torch.save(self._edge_normalizer, path + "_edge_normalizer.pth")
        torch.save(self._node_normalizer, path + "_node_normalizer.pth")

    def load_model(self, path):
        self.learned_model = torch.load(path + "_learned_model.pth")
        self._output_normalizer = torch.load(path + "_output_normalizer.pth")
        self._edge_normalizer = torch.load(path + "_edge_normalizer.pth")
        self._node_normalizer = torch.load(path + "_node_normalizer.pth")

    def evaluate(self):
        self.eval()
        self.learned_model.eval()
