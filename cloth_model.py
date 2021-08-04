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

# import sonnet as snt
# import tensorflow.compat.v1 as tf
import torch
from torch.nn import Module
from torch.nn.functional import one_hot

from meshgraphnets import common
from meshgraphnets import core_model
from meshgraphnets import normalization


class Model(Module):
  """Model for static cloth simulation."""

  def __init__(self, learned_model, name='Model'):
    super(Model, self).__init__()
    self._learned_model = learned_model
    # all normalizer will be bypassed and they return their input
    self._output_normalizer = normalization.Normalizer(
        size=3, name='output_normalizer')
    self._node_normalizer = normalization.Normalizer(
        size=3+common.NodeType.SIZE, name='node_normalizer')
    self._edge_normalizer = normalization.Normalizer(
        size=7, name='edge_normalizer')  # 2D coord + 3D coord + 2*length = 7
      

  def _build_graph(self, inputs, is_training):
    """Builds input graph."""
    print('in build graph')
    # construct graph nodes
    velocity = inputs['world_pos'] - inputs['prev|world_pos']
    # node_type = tf.one_hot(inputs['node_type'][:, 0], common.NodeType.SIZE)
    node_type = one_hot(inputs['node_type'][:, 0], common.NodeType.SIZE)
    node_features = torch.cat([velocity, node_type], axis=-1)

    '''
    print('velocity in build graph: ' + str(velocity))
    print('type of velocity in build graph: ' + str(type(velocity)))
    print('node type in build graph: ' + str(node_type))
    print('type of node type in build graph: ' + str(type(node_type)))  
    print('shape of node type in build graph: ' + str(tf.shape(node_type)))
    print('node features in build graph: ' + str(node_features))
    print('shape of node features in build graph: ' + str(tf.shape(node_features)))
    '''
    # construct graph edges
    senders, receivers = common.triangles_to_edges(inputs['cells'])
    relative_world_pos = (torch.gather(inputs['world_pos'], senders) -
                          torch.gather(inputs['world_pos'], receivers))
    relative_mesh_pos = (torch.gather(inputs['mesh_pos'], senders) -
                         torch.gather(inputs['mesh_pos'], receivers))
    edge_features = torch.cat([
        relative_world_pos,
        torch.norm(relative_world_pos, dim=-1, keepdims=True),
        relative_mesh_pos,
        torch.norm(relative_mesh_pos, dim=-1, keepdims=True)], axis=-1)

    mesh_edges = core_model.EdgeSet(
        name='mesh_edges',
        features=self._edge_normalizer(edge_features, is_training),
        receivers=receivers,
        senders=senders)
    return core_model.MultiGraph(
        node_features=self._node_normalizer(node_features, is_training),
        edge_sets=[mesh_edges])

  def forward(self, inputs):
    # print('in cloth model build')
    graph = self._build_graph(inputs, is_training=False)
    # show graph
    '''
    print('type of graph: ' + type(graph))
    print('type of graph.node_features: ' + type(graph.node_features))
    print('type of graph.edge_sets: ' + type(graph.edge_sets))
    quit()
    '''
    per_node_network_output = self._learned_model(graph)
    return self._update(inputs, per_node_network_output)

  # @snt.reuse_variables
  def loss(self, inputs):
    """L2 loss on position."""
    print('in loss')
    graph = self._build_graph(inputs, is_training=True)
    print('in loss, after build graph')
    network_output = self._learned_model(graph)

    # build target acceleration
    cur_position = inputs['world_pos']
    prev_position = inputs['prev|world_pos']
    target_position = inputs['target|world_pos']
    target_acceleration = target_position - 2*cur_position + prev_position
    target_normalized = self._output_normalizer(target_acceleration)

    # build loss
    loss_mask = torch.equal(inputs['node_type'][:, 0], common.NodeType.NORMAL)
    error = torch.sum((target_normalized - network_output)**2, dim=1)
    loss = torch.mean(error[loss_mask])
    return loss

  def train(self, input, optimizer):
    # build target acceleration
    cur_position = input['world_pos']
    prev_position = input['prev|world_pos']
    target_position = input['target|world_pos']
    target_acceleration = target_position - 2*cur_position + prev_position
    target_normalized = self._output_normalizer(target_acceleration)

    # build loss
    graph = self._build_graph(input, is_training=True)
    network_output = self._learned_model(graph)
    optimizer.zero_grad()
    loss = torch.nn.MSELoss(target_normalized, network_output)
    # loss_mask = torch.equal(input['node_type'][:, 0], common.NodeType.NORMAL)
    loss.backward()

  def _update(self, inputs, per_node_network_output):
    """Integrate model outputs."""
    acceleration = self._output_normalizer.inverse(per_node_network_output)
    # integrate forward
    cur_position = inputs['world_pos']
    prev_position = inputs['prev|world_pos']
    position = 2*cur_position + acceleration - prev_position
    return position
