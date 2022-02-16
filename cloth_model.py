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
"""Model for FlagSimple."""

import torch
from torch import nn as nn
import torch.nn.functional as F
# from torch_cluster import random_walk
import functools

import torch_scatter
import common
import normalization
import encode_process_decode

device = torch.device('cuda')


class Model(nn.Module):
    """Model for static cloth simulation."""

    def __init__(self, params, core_model_name=encode_process_decode, message_passing_aggregator='sum',
                 message_passing_steps=15, attention=False, ripple_used=False, ripple_generation=None,
                 ripple_generation_number=None,
                 ripple_node_selection=None, ripple_node_selection_random_top_n=None, ripple_node_connection=None,
                 ripple_node_ncross=None):
        super(Model, self).__init__()
        self._params = params
        self._output_normalizer = normalization.Normalizer(size=3, name='output_normalizer')
        self._node_normalizer = normalization.Normalizer(
            size=3 + common.NodeType.SIZE, name='node_normalizer')
        self._node_dynamic_normalizer = normalization.Normalizer(size=1, name='node_dynamic_normalizer')
        self._mesh_edge_normalizer = normalization.Normalizer(
            size=7, name='mesh_edge_normalizer')  # 2D coord + 3D coord + 2*length = 7
        self._world_edge_normalizer = normalization.Normalizer(size=4, name='world_edge_normalizer')
        self._model_type = params['model'].__name__

        self.core_model_name = core_model_name
        self.core_model = encode_process_decode
        self.message_passing_steps = message_passing_steps
        self.message_passing_aggregator = message_passing_aggregator
        self._attention = attention
        self._ripple_used = ripple_used
        if self._ripple_used:
            self._ripple_generation = ripple_generation
            self._ripple_generation_number = ripple_generation_number
            self._ripple_node_selection = ripple_node_selection
            self._ripple_node_selection_random_top_n = ripple_node_selection_random_top_n
            self._ripple_node_connection = ripple_node_connection
            self._ripple_node_ncross = ripple_node_ncross
            self.learned_model = self.core_model.EncodeProcessDecode(
                output_size=params['size'],
                latent_size=128,
                num_layers=2,
                message_passing_steps=self.message_passing_steps,
                message_passing_aggregator=self.message_passing_aggregator, attention=self._attention,
                ripple_used=self._ripple_used,
                ripple_generation=self._ripple_generation, ripple_generation_number=self._ripple_generation_number,
                ripple_node_selection=self._ripple_node_selection,
                ripple_node_selection_random_top_n=self._ripple_node_selection_random_top_n,
                ripple_node_connection=self._ripple_node_connection,
                ripple_node_ncross=self._ripple_node_ncross)
        else:
            self.learned_model = self.core_model.EncodeProcessDecode(
                output_size=params['size'],
                latent_size=128,
                num_layers=2,
                message_passing_steps=self.message_passing_steps,
                message_passing_aggregator=self.message_passing_aggregator, attention=self._attention,
                ripple_used=self._ripple_used)

    def unsorted_segment_operation(self, data, segment_ids, num_segments, operation):
        """
        Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

        :param data: A tensor whose segments are to be summed.
        :param segment_ids: The segment indices tensor.
        :param num_segments: The number of segments.
        :return: A tensor of same data type as the data argument.
        """
        assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

        # segment_ids is a 1-D tensor repeat it to have the same shape as data
        if len(segment_ids.shape) == 1:
            s = torch.prod(torch.tensor(data.shape[1:])).long().to(device)
            segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:]).to(device)

        assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

        shape = [num_segments] + list(data.shape[1:])
        result = torch.zeros(*shape)
        if operation == 'sum':
            result = torch_scatter.scatter_add(data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'max':
            result, _ = torch_scatter.scatter_max(data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'mean':
            result = torch_scatter.scatter_mean(data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'min':
            result, _ = torch_scatter.scatter_min(data.float(), segment_ids, dim=0, dim_size=num_segments)
        else:
            raise Exception('Invalid operation type!')
        result = result.type(data.dtype)
        return result

    def _build_graph(self, inputs, is_training):
        """Builds input graph."""
        world_pos = inputs['world_pos']
        prev_world_pos = inputs['prev|world_pos']
        node_type = inputs['node_type']
        velocity = world_pos - prev_world_pos
        one_hot_node_type = F.one_hot(node_type[:, 0].to(torch.int64), common.NodeType.SIZE)

        node_features = torch.cat((velocity, one_hot_node_type), dim=-1)

        cells = inputs['cells']
        decomposed_cells = common.triangles_to_edges(cells)
        senders, receivers = decomposed_cells['two_way_connectivity']

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

        mesh_edges = self.core_model.EdgeSet(
            name='mesh_edges',
            features=self._mesh_edge_normalizer(edge_features, is_training),
            receivers=receivers,
            senders=senders)

        if self._ripple_used:
            num_nodes = node_type.shape[0]
            max_node_dynamic = self.unsorted_segment_operation(torch.norm(relative_world_pos, dim=-1), receivers,
                                                               num_nodes,
                                                               operation='max').to(device)
            min_node_dynamic = self.unsorted_segment_operation(torch.norm(relative_world_pos, dim=-1), receivers,
                                                               num_nodes,
                                                               operation='min').to(device)
            node_dynamic = self._node_dynamic_normalizer(max_node_dynamic - min_node_dynamic)

            return (self.core_model.MultiGraphWithPos(node_features=self._node_normalizer(node_features),
                                                      edge_sets=[mesh_edges], target_feature=world_pos,
                                                      model_type=self._model_type,
                                                      node_dynamic=node_dynamic))
        else:
            return (self.core_model.MultiGraph(node_features=self._node_normalizer(node_features),
                                               edge_sets=[mesh_edges]))

    def forward(self, inputs, is_training):
        graph = self._build_graph(inputs, is_training=is_training)
        if is_training:
            return self.learned_model(graph,
                                      world_edge_normalizer=self._world_edge_normalizer, is_training=is_training)
        else:
            return self._update(inputs, self.learned_model(graph,
                                                           world_edge_normalizer=self._world_edge_normalizer,
                                                           is_training=is_training))

    def _update(self, inputs, per_node_network_output):
        """Integrate model outputs."""

        acceleration = self._output_normalizer.inverse(per_node_network_output)

        # integrate forward
        cur_position = inputs['world_pos']
        prev_position = inputs['prev|world_pos']
        position = 2 * cur_position + acceleration - prev_position
        return position

    def get_output_normalizer(self):
        return self._output_normalizer

    def save_model(self, path):
        torch.save(self.learned_model, path + "_learned_model.pth")
        torch.save(self._output_normalizer, path + "_output_normalizer.pth")
        torch.save(self._mesh_edge_normalizer, path + "_mesh_edge_normalizer.pth")
        torch.save(self._world_edge_normalizer, path + "_world_edge_normalizer.pth")
        torch.save(self._node_normalizer, path + "_node_normalizer.pth")
        torch.save(self._node_dynamic_normalizer, path + "_node_dynamic_normalizer.pth")

    def load_model(self, path):
        self.learned_model = torch.load(path + "_learned_model.pth")
        self._output_normalizer = torch.load(path + "_output_normalizer.pth")
        self._mesh_edge_normalizer = torch.load(path + "_mesh_edge_normalizer.pth")
        self._world_edge_normalizer = torch.load(path + "_world_edge_normalizer.pth")
        self._node_normalizer = torch.load(path + "_node_normalizer.pth")
        self._node_dynamic_normalizer = torch.load(path + "_node_dynamic_normalizer.pth")

    def evaluate(self):
        self.eval()
        self.learned_model.eval()
