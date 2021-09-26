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
import functools
import torch
from torch import nn

EdgeSet = collections.namedtuple('EdgeSet', ['name', 'features', 'senders',
                                             'receivers'])
MultiGraph = collections.namedtuple('Graph', ['node_features', 'edge_sets'])

class LazyMLP(nn.Module):
  def __init__(self, output_sizes):
    super().__init__()
    self.layers = nn.Sequential()
    for index, output_size in enumerate(output_sizes):
      # self.layers.add_module("linear_%d" % index, LazyLinear(output_size))
      self.layers.add_module("linear_%d" % index, nn.Linear(output_size, output_size))

  def forward(self, input):
    y = self.layers(input)
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

  def __init__(self, model_fn, name='GraphNetBlock'):
    super().__init__()
    self._model_fn = model_fn

  def _update_edge_features(self, node_features, edge_set):
    """Aggregrates node features, and applies edge function."""
    sender_features = torch.gather(input=node_features, index=edge_set.senders)
    receiver_features = torch.gather(input=node_features, index=edge_set.receivers)
    features = [sender_features, receiver_features, edge_set.features]
    # with tf.variable_scope(edge_set.name+'_edge_fn'):
    return self._model_fn(torch.cat(features, axis=-1))

  def _update_node_features(self, node_features, edge_sets):
    """Aggregrates edge features, and applies node function."""
    # num_nodes = tf.shape(node_features)[0]
    dim_to_be_got = node_features.shape
    num_nodes = len(dim_to_be_got)
    features = [node_features]
    for edge_set in edge_sets:
      scatter_add_result = torch.scatter_add(torch.zeros(num_nodes,1), 0, edge_set.receivers, edge_set.features)
      features.append(scatter_add_result)
      '''
      features.append(tf.math.unsorted_segment_sum(edge_set.features,
                                                   edge_set.receivers,
                                                   num_nodes))
      '''
    # with tf.variable_scope('node_fn'):
      return self._model_fn(torch.cat(features, axis=-1))

  def forward(self, graph):
    """Applies GraphNetBlock and returns updated MultiGraph."""

    # apply edge functions
    new_edge_sets = []
    for edge_set in graph.edge_sets:
      updated_features = self._update_edge_features(graph.node_features,
                                                    edge_set)
      new_edge_sets.append(edge_set._replace(features=updated_features))

    # apply node function
    new_node_features = self._update_node_features(graph.node_features,
                                                   new_edge_sets)

    # add residual connections
    new_node_features += graph.node_features
    new_edge_sets = [es._replace(features=es.features + old_es.features)
                     for es, old_es in zip(new_edge_sets, graph.edge_sets)]
    return MultiGraph(new_node_features, new_edge_sets)

class Encoder(nn.Module):
    """Encodes node and edge features into latent features."""
    def __init__(self, graph, make_mlp, latent_size):
      super().__init__()
      self.node_model = make_mlp(latent_size)
      self.edge_models = []
      for _ in graph.edge_sets:
        edge_model = make_mlp(latent_size)
        self.edge_models.append(edge_model)

    def forward(self, graph):
      node_latents = self.node_model(graph.node_features)
      new_edges_sets = []
      for edge_set, edge_model in zip(graph.edge_sets, self.edge_models):
        latent = edge_model(edge_set.features)
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
  def __init__(self, mlp, message_passing_steps):
    super().__init__()
    for _ in range(message_passing_steps):
      self.submodules.add_module(GraphNetBlock(model_fn=mlp))

  def forward(self, graph):
    return super().forward(graph)

class EncodeProcessDecode(nn.Module):
  """Encode-Process-Decode GraphNet model."""
  def __init__(self,
               output_size,
               latent_size,
               num_layers,
               message_passing_steps,
               graph):
    super().__init__()
    self._latent_size = latent_size
    self._output_size = output_size
    self._num_layers = num_layers
    self._message_passing_steps = message_passing_steps
    self._graph = graph
    self._encoder = Encoder(graph=graph, make_mlp=self._make_mlp, latent_size=self._latent_size)
    self._processor = Processor(mlp=self._make_mlp(output_size=self._latent_size), message_passing_steps=self._message_passing_steps)
    self._decoder = Decoder(make_mlp=functools.partial(self._make_mlp, layer_norm=False), output_size=self._output_size)

  def _make_mlp(self, output_size, layer_norm=True):
      """Builds an MLP."""
      widths = [self._latent_size] * self._num_layers + [output_size]
      network = LazyMLP(widths)
      if layer_norm:
        network = nn.Sequential([network, nn.LayerNorm()])
      return network

  def forward(self, graph):
    """Encodes and processes a multigraph, and returns node features."""
    latent_graph = self._encoder(graph)
    latent_graph = self._processor(latent_graph)
    return self._decoder(latent_graph)
