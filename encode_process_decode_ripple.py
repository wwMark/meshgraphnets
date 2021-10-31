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
"""Core learned graph net model."""

import collections
from math import ceil
from collections import OrderedDict
import functools
import torch
from torch import nn as nn
import torch_scatter
from torch_scatter.composite import scatter_softmax
import torch.nn.functional as F

EdgeSet = collections.namedtuple('EdgeSet', ['name', 'features', 'senders',
                                             'receivers'])
MultiGraph = collections.namedtuple('Graph', ['node_features', 'edge_sets'])
MultiGraphWithPos = collections.namedtuple('Graph', ['node_features', 'edge_sets', 'world_pos', 'mesh_pos'])

device = torch.device('cuda')


class LazyMLP(nn.Module):
    def __init__(self, output_sizes):
        super().__init__()
        num_layers = len(output_sizes)
        self._layers_ordered_dict = OrderedDict()
        for index, output_size in enumerate(output_sizes):
            self._layers_ordered_dict["linear_" + str(index)] = nn.LazyLinear(output_size)
            if index < (num_layers - 1):
                self._layers_ordered_dict["relu_" + str(index)] = nn.ReLU()
        self.layers = nn.Sequential(self._layers_ordered_dict)

    def forward(self, input):
        input = input.to(device)
        y = self.layers(input)
        return y


class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.LazyLinear(1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, index):
        latent = self.linear_layer(input)
        latent = self.leaky_relu(latent)
        result = torch.zeros(*latent.shape)
        result = scatter_softmax(latent.float(), index, dim=0)
        result = result.type(result.dtype)
        return result


class GraphNetBlock(nn.Module):
    """Multi-Edge Interaction Network with residual connections."""

    def __init__(self, model_fn, output_size, message_passing_aggregator, attention=False):
        super().__init__()
        self.edge_model = model_fn(output_size)
        self.node_model = model_fn(output_size)
        self.attention = attention
        if attention:
            self.attention_model = AttentionModel()
        self.message_passing_aggregator = message_passing_aggregator

    def _update_edge_features(self, node_features, edge_set):
        """Aggregrates node features, and applies edge function."""
        senders = edge_set.senders.to(device)
        receivers = edge_set.receivers.to(device)
        sender_features = torch.index_select(input=node_features, dim=0, index=senders)
        receiver_features = torch.index_select(input=node_features, dim=0, index=receivers)
        features = [sender_features, receiver_features, edge_set.features]
        features = torch.cat(features, dim=-1)
        return self.edge_model(features)

    '''
    def _update_node_features_mp_helper(self, features, receivers, add_intermediate):
        for index, feature_tensor in enumerate(features):
            des_index = receivers[index]
            add_intermediate[des_index].add(feature_tensor)
        return add_intermediate
    '''

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
        if operation == 'add':
            result = torch_scatter.scatter_add(data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'max':
            result, _ = torch_scatter.scatter_max(data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'mean':
            result = torch_scatter.scatter_mean(data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'min':
            result, _ = torch_scatter.scatter_min(data.float(), segment_ids, dim=0, dim_size=num_segments)
        result = result.type(data.dtype)
        return result

    def _update_node_features(self, node_features, edge_sets):
        """Aggregrates edge features, and applies node function."""
        num_nodes = node_features.shape[0]
        features = [node_features]
        for edge_set in edge_sets:
            if self.attention:
                receiver_features = edge_set.features[edge_set.receivers]
                sender_features = edge_set.features[edge_set.senders]
                attention = self.attention_model(torch.cat((sender_features, receiver_features), dim=1), edge_set.receivers)
                attention = attention.repeat_interleave(
                    torch.prod(torch.tensor(edge_set.features.shape[1:])).long().to(device))
                attention = attention.view(edge_set.features.shape[0], *edge_set.features.shape[1:]).to(device)
                features.append(
                    self.unsorted_segment_operation(torch.mul(edge_set.features, attention), edge_set.receivers,
                                                    num_nodes, operation=self.message_passing_aggregator))
            else:
                features.append(
                    self.unsorted_segment_operation(edge_set.features, edge_set.receivers, num_nodes,
                                                    operation=self.message_passing_aggregator))
        features = torch.cat(features, dim=-1)
        return self.node_model(features)

    def forward(self, graph):
        """Applies GraphNetBlock and returns updated MultiGraph."""
        # apply edge functions
        new_edge_sets = []
        for edge_set in graph.edge_sets:
            updated_features = self._update_edge_features(graph.node_features, edge_set)
            new_edge_sets.append(edge_set._replace(features=updated_features))

        # apply node function
        new_node_features = self._update_node_features(graph.node_features, new_edge_sets)

        # add residual connections
        new_node_features += graph.node_features
        new_edge_sets = [es._replace(features=es.features + old_es.features)
                         for es, old_es in zip(new_edge_sets, graph.edge_sets)]
        return MultiGraph(new_node_features, new_edge_sets)


class Encoder(nn.Module):
    """Encodes node and edge features into latent features."""

    def __init__(self, make_mlp, latent_size):
        super().__init__()
        self._make_mlp = make_mlp
        self._latent_size = latent_size
        self.node_model = self._make_mlp(latent_size)
        self.mesh_edge_model = self._make_mlp(latent_size)
        '''
        for _ in graph.edge_sets:
          edge_model = make_mlp(latent_size)
          self.edge_models.append(edge_model)
        '''

    def forward(self, graph):
        node_latents = self.node_model(graph.node_features)
        new_edges_sets = []

        for index, edge_set in enumerate(graph.edge_sets):
            feature = edge_set.features
            latent = self.mesh_edge_model(feature)
            new_edges_sets.append(edge_set._replace(features=latent))
        return MultiGraph(node_latents, new_edges_sets)


class Decoder(nn.Module):
    """Decodes node features from graph."""
    # decoder = self._make_mlp(self._output_size, layer_norm=False)
    # return decoder(graph.node_features)

    """Encodes node and edge features into latent features."""

    def __init__(self, make_mlp, output_size):
        super().__init__()
        self.model = make_mlp(output_size)

    def forward(self, graph):
        return self.model(graph.node_features)


class RippleConnectionGenerator(nn.Module):
    '''
    ripple_size_generator will generate for each ripple a node size according to some math equation
    '''

    def __init__(self, make_mlp, output_size, normalize_connection, num_ripples, ripple_sample_size_generator,
                 num_or_percentage={'option': 'percentage', 'value': 0.01}, equal_generator_sample_size=10):
        super().__init__()
        self._num_or_percentage = num_or_percentage['option']
        self._num_or_percentage_value = num_or_percentage['value']
        self._num_ripples = num_ripples
        self._normalize_connection = normalize_connection
        self._ripple_sample_size_generator = ripple_sample_size_generator
        self._equal_generator_sample_size = equal_generator_sample_size

        # ripple model will learn ripple influence based on source feature and source-destination distance
        self.ripple_model = make_mlp(output_size)
        if self._normalize_connection:
            self.ripple_model = nn.Sequential(nn.LayerNorm(normalized_shape=17), self.ripple_model)

    def generate_ripple_sample_size(self, ripple_order, ripple_size, ripple_sample_size_generator):
        '''
        Currently assume that exponential
        '''
        if ripple_sample_size_generator == 'equal':
            equal_generator_sample_size = self._equal_generator_sample_size
            if equal_generator_sample_size > ripple_size:
                print(equal_generator_sample_size)
                print(ripple_size)
                raise Exception(
                    'Equal ripple sample size generator does not work properly. Sample number of nodes in ripple are greater than number of all ripple nodes!')
            return equal_generator_sample_size
        elif ripple_sample_size_generator == 'exponential':
            sample_size = (ripple_order + 1) * (ripple_order + 1)
            if sample_size > ripple_size:
                raise Exception(
                    'Exponential ripple size generator does not work properly. Sample number of nodes in ripple are greater than number of all ripple nodes!')
            return sample_size

    def forward(self, latent_graph, graph):
        # get graph node features with the highest velocity
        world_pos_matrix = graph.world_pos
        mesh_pos_matrix = graph.mesh_pos
        velocity_matrix = graph.node_features[:, 0:3]
        num_nodes = world_pos_matrix.shape[0]
        num_influential_nodes = ceil(world_pos_matrix.shape[
                                         0] * self._num_or_percentage_value) if self._num_or_percentage == 'percentage' else self._num_or_percentage_value
        sort_by_velocity = torch.square(velocity_matrix)
        sort_by_velocity = torch.sum(sort_by_velocity, dim=-1)
        _, sort_indices = torch.sort(sort_by_velocity, dim=0, descending=True)

        # virtual highest node is the sum of a number of the highest node
        highest_velocity_node_feature = torch.sum(velocity_matrix[sort_indices[0:num_influential_nodes]], dim=0)
        highest_velocity_node_world_pos = torch.sum(world_pos_matrix[sort_indices[0:num_influential_nodes]], dim=0)
        highest_velocity_node_mesh_pos = torch.sum(mesh_pos_matrix[sort_indices[0:num_influential_nodes]], dim=0)

        # get all nodes that need to establish a connection with highest_velocity_node from velocity_matrix
        ripple_size = num_nodes // self._num_ripples
        ripple_size_rest = num_nodes % self._num_ripples
        ripple_nodes_feature = []
        ripple_nodes_index = []
        ripple_nodes_world_pos = []
        ripple_nodes_mesh_pos = []
        for i in range(self._num_ripples):
            start_index = i * ripple_size
            ripple_actual_size = ripple_size if i < self._num_ripples - 1 else ripple_size + ripple_size_rest
            ripple_sample_size = self.generate_ripple_sample_size(i, ripple_actual_size,
                                                                  ripple_sample_size_generator='equal')
            end_index = start_index + ripple_actual_size

            random_select_mask = torch.randperm(n=ripple_actual_size)[0:ripple_sample_size]
            random_select_mask = random_select_mask[0:ripple_sample_size]

            info_of_a_ripple = velocity_matrix[start_index:end_index]
            info_of_a_ripple = info_of_a_ripple[random_select_mask]
            ripple_nodes_feature.append(info_of_a_ripple)
            index = random_select_mask + start_index
            ripple_nodes_index.append(index)
            info_of_a_ripple = world_pos_matrix[start_index:end_index]
            info_of_a_ripple = info_of_a_ripple[random_select_mask]
            ripple_nodes_world_pos.append(info_of_a_ripple)
            info_of_a_ripple = mesh_pos_matrix[start_index:end_index]
            info_of_a_ripple = info_of_a_ripple[random_select_mask]
            ripple_nodes_mesh_pos.append(info_of_a_ripple)
        ripple_nodes_feature = torch.cat(ripple_nodes_feature, dim=0)
        ripple_nodes_index = torch.cat(ripple_nodes_index, dim=0)
        ripple_nodes_world_pos = torch.cat(ripple_nodes_world_pos, dim=0)
        ripple_nodes_mesh_pos = torch.cat(ripple_nodes_mesh_pos, dim=0)

        relative_world_pos = torch.sub(ripple_nodes_world_pos, highest_velocity_node_world_pos)
        relative_mesh_pos = torch.sub(ripple_nodes_mesh_pos, highest_velocity_node_mesh_pos)

        ripple_and_highest_info = torch.cat((highest_velocity_node_feature.repeat(ripple_nodes_feature.shape[0],
                                                                                  highest_velocity_node_feature.shape[
                                                                                      0]), ripple_nodes_feature,
                                             relative_world_pos, relative_mesh_pos), dim=-1)
        ripple_and_highest_result = self.ripple_model(ripple_and_highest_info)
        latent_graph.node_features[ripple_nodes_index] += ripple_and_highest_result
        return latent_graph


class Processor(nn.Module):
    '''
    This class takes the nodes with the most influential feature (sum of square)
    The the chosen numbers of nodes in each ripple will establish connection(features and distances) with the most influential nodes and this connection will be learned
    Then the result is add to output latent graph of encoder and the modified latent graph will be feed into original processor

    Option: choose whether to normalize the high rank node connection
    '''

    def __init__(self, make_mlp, output_size, message_passing_steps, message_passing_aggregator, attention=False):
        super().__init__()
        self._submodules_ordered_dict = OrderedDict()
        for index in range(message_passing_steps):
            self._submodules_ordered_dict[str(index)] = GraphNetBlock(model_fn=make_mlp, output_size=output_size,
                                                                      message_passing_aggregator=message_passing_aggregator,
                                                                      attention=attention)
        self.submodules = nn.Sequential(self._submodules_ordered_dict)

    def forward(self, latent_graph):
        return self.submodules(latent_graph)


class EncodeProcessDecode(nn.Module):
    """Encode-Process-Decode GraphNet model."""

    def __init__(self,
                 output_size,
                 latent_size,
                 num_layers,
                 message_passing_steps, message_passing_aggregator):
        super().__init__()
        self._latent_size = latent_size
        self._output_size = output_size
        self._num_layers = num_layers
        self._message_passing_steps = message_passing_steps
        self._message_passing_aggregator = message_passing_aggregator
        self.encoder = Encoder(make_mlp=self._make_mlp, latent_size=self._latent_size)
        self.ripple_connection_generator = RippleConnectionGenerator(make_mlp=self._make_mlp,
                                                                     output_size=self._latent_size,
                                                                     normalize_connection=True, num_ripples=6,
                                                                     ripple_sample_size_generator='equal',
                                                                     num_or_percentage={'option': 'percentage',
                                                                                        'value': 0.01},
                                                                     equal_generator_sample_size=10)
        self.processor = Processor(make_mlp=self._make_mlp, output_size=self._latent_size,
                                   message_passing_steps=self._message_passing_steps,
                                   message_passing_aggregator=self._message_passing_aggregator, attention=True)
        self.decoder = Decoder(make_mlp=functools.partial(self._make_mlp, layer_norm=False),
                               output_size=self._output_size)

    def _make_mlp(self, output_size, layer_norm=True):
        """Builds an MLP."""
        widths = [self._latent_size] * self._num_layers + [output_size]
        network = LazyMLP(widths)
        if layer_norm:
            network = nn.Sequential(network, nn.LayerNorm(normalized_shape=widths[-1]))
        return network

    def forward(self, graph):
        """Encodes and processes a multigraph, and returns node features."""
        latent_graph = self.encoder(graph)
        latent_graph = self.ripple_connection_generator(latent_graph, graph)
        latent_graph = self.processor(latent_graph)
        return self.decoder(latent_graph)
