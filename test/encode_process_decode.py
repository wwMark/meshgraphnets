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
from torch import multiprocessing as mp

EdgeSet = collections.namedtuple('EdgeSet', ['name', 'features', 'senders',
                                             'receivers'])
MultiGraph = collections.namedtuple('Graph', ['node_features', 'edge_sets'])

cuda0 = torch.device('cuda:0')

device = torch.device('cuda')

class LazyMLP(nn.Module):
  def __init__(self, output_sizes):
    super().__init__()
    self.layers = nn.Sequential()
    for index, output_size in enumerate(output_sizes):
      # self.layers.add_module("linear_%d" % index, LazyLinear(output_size))
      self.layers.add_module("linear_%d" % index, nn.LazyLinear(output_size))

  def forward(self, input):
    # print("-------------")
    # print("1. LazyMLP input is cuda", input.is_cuda)
    input = input.cuda(cuda0)
    # print("2. LazyMLP input is cuda", input.is_cuda)
    # print("1. MLP is cuda", next(self.layers.parameters()).is_cuda)
    y = self.layers(input)
    # print("2. MLP is cuda", next(self.layers.parameters()).is_cuda)
    # print("LazyMLP self.layers", self.layers)
    # print("-------------")
    return y

'''
output_sizes = [1, 2, 5, 7]
model = LazyMLP(output_sizes)
print("model", model)
for parameter in model.parameters():
  print(parameter)
'''

class GraphNetBlock(nn.Module):
  """Multi-Edge Interaction Network with residual connections."""

  def __init__(self, model_fn, output_size, name='GraphNetBlock'):
    super().__init__()
    self._edge_model = model_fn(output_size).cuda()
    self._node_model = model_fn(output_size).cuda()
    # self._cpu_cores = mp.cpu_count()
    # self._cpu_cores = 4
    # self._update_node_features_thread_pool = mp.Pool(self._cpu_cores)

  def _update_edge_features(self, node_features, edge_set):
    """Aggregrates node features, and applies edge function."""
    # print("GNP in update edge features......")
    # print("GNP update edge features node_features is cuda", node_features.is_cuda)
    # print("GNP update edge features edge_set.senders is cuda", edge_set.senders.is_cuda)
    senders = edge_set.senders.cuda()
    receivers = edge_set.receivers.cuda()
    sender_features = torch.index_select(input=node_features, dim=0, index=senders)
    receiver_features = torch.index_select(input=node_features, dim=0, index=receivers)
    features = [sender_features, receiver_features, edge_set.features]
    features = torch.cat(features, axis=-1)
    # print("GNP features shape after update edge", features.shape)
    # with tf.variable_scope(edge_set.name+'_edge_fn'):
    return self._edge_model(features)

  def _update_node_features_mp_helper(self, features, receivers, add_intermediate):
    for index, feature_tensor in enumerate(features):
      des_index = receivers[index]
      add_intermediate[des_index].add(feature_tensor)
    return add_intermediate

  def _update_node_features(self, node_features, edge_sets):
    """Aggregrates edge features, and applies node function."""
    # print("GNP in update node features......")
    # num_nodes = tf.shape(node_features)[0]
    # print("GNB update node features, node feature shape", node_features.shape)
    dim_to_be_got = node_features.shape
    num_nodes = dim_to_be_got[0]
    features = [node_features.to('cpu')]
    for edge_set in edge_sets:
      # print("edge_set.features", edge_set.features.shape)
      # print("edge_set.receivers", edge_set.receivers.shape)
      # print("num_nodes", num_nodes)
      add_intermediate = torch.zeros(num_nodes, edge_set.features.shape[1])
      for index, feature_tensor in enumerate(edge_set.features.to('cpu')):
        des_index = edge_set.receivers[index].to('cpu')
        add_intermediate[des_index].add(feature_tensor)
      features.append(add_intermediate)
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

      '''
      features.append(tf.math.unsorted_segment_sum(edge_set.features,
                                                   edge_set.receivers,
                                                   num_nodes)) m
      '''
    # with tf.variable_scope('node_fn'):
    features = torch.cat(features, axis=-1)
    # print("GNP features shape after update node", features.shape)
    return self._node_model(features)

  def forward(self, graph):
    """Applies GraphNetBlock and returns updated MultiGraph."""
    # print("GNP node features shape of input graph", graph.node_features.shape)
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
      self.edge_models = nn.ModuleList()
      '''
      for _ in graph.edge_sets:
        edge_model = make_mlp(latent_size)
        self.edge_models.append(edge_model)
      '''

    def forward(self, graph):
      node_latents = self.node_model(graph.node_features)
      new_edges_sets = []

      for index, edge_set in enumerate(graph.edge_sets):
        try:
          edge_model = self.edge_models[index]
        except IndexError:
          edge_model = self._make_mlp(self._latent_size)
          self.edge_models.append(edge_model)
        feature_cuda = edge_set.features.cuda()
        latent = edge_model(feature_cuda)
        new_edges_sets.append(edge_set._replace(features=latent))
      return MultiGraph(node_latents, new_edges_sets)

      '''
      for edge_set, edge_model in zip(graph.edge_sets, self.edge_models):
        feature_cuda = edge_set.features.cuda()
        latent = edge_model(feature_cuda)
        new_edges_sets.append(edge_set._replace(features=latent))
      return MultiGraph(node_latents, new_edges_sets)
      '''

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
    self._submodules = nn.Sequential(self._submodules_ordered_dict).cuda()

  def forward(self, graph):
    # print("Processor----------------")
    # print("Processor self._submodules", self._submodules)
    return self._submodules(graph)

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
    self._encoder = Encoder(make_mlp=self._make_mlp, latent_size=self._latent_size).to(device)
    self._processor = Processor(make_mlp=self._make_mlp, output_size=self._latent_size, message_passing_steps=self._message_passing_steps).to(device)
    self._decoder = Decoder(make_mlp=functools.partial(self._make_mlp, layer_norm=False), output_size=self._output_size).to(device)

  def _make_mlp(self, output_size, layer_norm=True):
      """Builds an MLP."""
      widths = [self._latent_size] * self._num_layers + [output_size]
      network = LazyMLP(widths)
      if layer_norm:
        network = nn.Sequential(network, nn.LayerNorm(normalized_shape=widths[-1])).cuda()
      return network.to(device)

  def forward(self, graph):
    """Encodes and processes a multigraph, and returns node features."""
    # print("encoder is cuda", next(self._encoder.parameters()).is_cuda)
    latent_graph = self._encoder(graph)
    # print("EncodeProcessDecode input graph", graph)
    latent_graph = self._processor(latent_graph)
    return self._decoder(latent_graph)
