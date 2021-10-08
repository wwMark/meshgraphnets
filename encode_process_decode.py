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
from collections import OrderedDict
import functools
import torch
from torch import nn as nn
# from torch import multiprocessing as mp

EdgeSet = collections.namedtuple('EdgeSet', ['name', 'features', 'senders',
                                             'receivers'])
MultiGraph = collections.namedtuple('Graph', ['node_features', 'edge_sets'])

device = torch.device('cuda')


class LazyMLP(nn.Module):
    def __init__(self, output_sizes):
        super().__init__()
        num_layers = len(output_sizes)
        self._layers_ordered_dict = OrderedDict()
        for index, output_size in enumerate(output_sizes):
            self._layers_ordered_dict["linear_" + str(index)] = nn.LazyLinear(output_size)
            # self.layers.add_module("linear_%d" % index, LazyLinear(output_size))
            if index < (num_layers - 1):
                self._layers_ordered_dict["relu_" + str(index)] = nn.ReLU()
        self.layers = nn.Sequential(self._layers_ordered_dict)

    def forward(self, input):
        input = input.to(device)
        y = self.layers(input)
        return y

class GraphNetBlock(nn.Module):
    """Multi-Edge Interaction Network with residual connections."""

    def __init__(self, model_fn, output_size):
        super().__init__()
        self.edge_model = model_fn(output_size)
        self.node_model = model_fn(output_size)
        # self._cpu_cores = mp.cpu_count()
        # self._cpu_cores = 4
        # self._update_node_features_thread_pool = mp.Pool(self._cpu_cores)

    def _update_edge_features(self, node_features, edge_set):
        """Aggregrates node features, and applies edge function."""
        senders = edge_set.senders.to(device)
        receivers = edge_set.receivers.to(device)
        sender_features = torch.index_select(input=node_features, dim=0, index=senders)
        receiver_features = torch.index_select(input=node_features, dim=0, index=receivers)
        features = [sender_features, receiver_features, edge_set.features]
        features = torch.cat(features, dim=-1)
        # print("GNP features shape after update edge", features.shape)
        # with tf.variable_scope(edge_set.name+'_edge_fn'):
        return self.edge_model(features)

    '''
    def _update_node_features_mp_helper(self, features, receivers, add_intermediate):
        for index, feature_tensor in enumerate(features):
            des_index = receivers[index]
            add_intermediate[des_index].add(feature_tensor)
        return add_intermediate
    '''

    def unsorted_segment_sum(self, data, segment_ids, num_segments):
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
        zeros = torch.zeros(*shape).to(device)
        tensor = zeros.scatter_add(0, segment_ids.to(device), data.float().to(device))
        tensor = tensor.type(data.dtype)
        return tensor

    def _update_node_features(self, node_features, edge_sets):
        """Aggregrates edge features, and applies node function."""
        num_nodes = node_features.shape[0]
        features = [node_features]
        for edge_set in edge_sets:
            features.append(self.unsorted_segment_sum(edge_set.features, edge_set.receivers, num_nodes))
            '''
            add_intermediate = torch.zeros(num_nodes, edge_set.features.shape[1])
            for index, feature_tensor in enumerate(edge_set.features.to('cpu')):
              des_index = edge_set.receivers[index].to('cpu')
              add_intermediate[des_index].add(feature_tensor)
            features.append(add_intermediate)
            '''
            '''
            # use multiprocessing to decrease execution time
            
            payload_per_core = edge_set.features.shape[0] // self._cpu_cores
            argument_list = []
            for i in range(self._cpu_cores - 1):
              i = i * payload_per_core
              add_intermediate = torch.zeros(num_nodes, edge_set.features.shape[1]).to(device=cuda0)
              argument_list.append((edge_set.features[i : i + payload_per_core, : ], edge_set.receivers[i : i + payload_per_core], add_intermediate))
            i = (self._cpu_cores - 1) * payload_per_core
            add_intermediate = torch.zeros(num_nodes, edge_set.features.shape[1]).to(device=cuda0)
            argument_list.append((edge_set.features[i : , : ], edge_set.receivers[i : ], add_intermediate))
            result_iterable = self._update_node_features_thread_pool.starmap(self._update_node_features_mp_helper, argument_list)
            result = torch.zeros(num_nodes, edge_set.features.shape[1]).to(device=cuda0)
            for add_intermediate in result_iterable:
              result.add(add_intermediate)
            features.append(result)
            '''
        features = torch.cat(features, axis=-1)
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


class Processor(nn.Module):
    def __init__(self, make_mlp, output_size, message_passing_steps):
        super().__init__()
        self._submodules_ordered_dict = OrderedDict()
        for index in range(message_passing_steps):
            self._submodules_ordered_dict[str(index)] = GraphNetBlock(model_fn=make_mlp, output_size=output_size)
        self.submodules = nn.Sequential(self._submodules_ordered_dict)
        '''
        super().__init__()
        self._message_passing_steps = message_passing_steps
        self.submodule = GraphNetBlock(model_fn=make_mlp, output_size=output_size)
        '''

    def forward(self, graph):
        # print("Processor----------------")
        # print("Processor self._submodules", self._submodules)
        # print(repr(self.modules()))
        return self.submodules(graph)
        '''
        latent_graph = graph
        for _ in range(self._message_passing_steps):
            latent_graph = self.submodule(latent_graph)
        return latent_graph
        '''

class EncodeProcessDecode(nn.Module):
    """Encode-Process-Decode GraphNet model."""

    def __init__(self,
                 output_size,
                 latent_size,
                 num_layers,
                 message_passing_steps):
        super().__init__()
        self._latent_size = latent_size
        self._output_size = output_size
        self._num_layers = num_layers
        self._message_passing_steps = message_passing_steps
        self.encoder = Encoder(make_mlp=self._make_mlp, latent_size=self._latent_size)
        self.processor = Processor(make_mlp=self._make_mlp, output_size=self._latent_size,
                                    message_passing_steps=self._message_passing_steps)
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
        # print("EncodeProcessDecode input graph", graph)
        latent_graph = self.processor(latent_graph)
        return self.decoder(latent_graph)
