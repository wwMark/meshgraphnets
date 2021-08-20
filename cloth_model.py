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
from torch import nn as nn
from torch.nn import Module
import torch.nn.functional as F
# from torch.nn.functional import one_hot

import common
# from meshgraphnets import core_model
import normalization
from test import encode_process_decode


class Model(nn.Module):
  """Model for static cloth simulation."""
  def __init__(self, trajectory_state, params, name='Model'):
    super(Model, self).__init__()
    self._params = params
    self._output_normalizer = normalization.Normalizer(
        size=3, name='output_normalizer')
    self._node_normalizer = normalization.Normalizer(
        size=3+common.NodeType.SIZE, name='node_normalizer')
    self._edge_normalizer = normalization.Normalizer(
        size=7, name='edge_normalizer')  # 2D coord + 3D coord + 2*length = 7

    self.learned_model = encode_process_decode.EncodeProcessDecode(
                        output_size=self._params['size'],
                        latent_size=128,
                        num_layers=2,
                        message_passing_steps=15)
    
    '''
    is_training = False
    graph = self._build_graph(trajectory_state, is_training=is_training)
    self.learned_model = encode_process_decode.EncodeProcessDecode(
        output_size=self._params['size'],
        latent_size=128,
        num_layers=2,
        message_passing_steps=15,
        graph=graph)
    '''

  def _build_graph(self, inputs, is_training):
    """Builds input graph."""
    # construct graph nodes
    # print("type of inputs", type(inputs))
    # print("inputs", inputs)
    velocity = inputs['world_pos'] - inputs['prev|world_pos']
    # node_type = tf.one_hot(inputs['node_type'][:, 0], common.NodeType.SIZE)
    # print("inputs node_type", inputs['node_type'][:, :, 0])
    # quit()
    node_type = F.one_hot(inputs['node_type'][:, :, 0][0].to(torch.int64), common.NodeType.SIZE)
    '''
    print("shape velocity", velocity.shape)
    print("shape node_type", node_type.shape)
    quit()
    '''
    velocity = torch.squeeze(velocity)
    node_features = torch.cat((velocity, node_type), dim=-1)

    # construct graph edges
    '''
    print("shape of cells", inputs['cells'].shape)
    quit()
    '''
    senders, receivers = common.triangles_to_edges(torch.squeeze(inputs['cells']))
    '''
    print("input shape", inputs['world_pos'].shape)
    print("index shape", senders.shape)
    print("max senders", torch.max(senders))
    print("max receivers", torch.max(receivers))
    '''
    world_pos = torch.squeeze(inputs['world_pos'])
    mesh_pos = torch.squeeze(inputs['mesh_pos'])
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
        features=self._edge_normalizer(edge_features, is_training),
        receivers=receivers,
        senders=senders)
    return encode_process_decode.MultiGraph(
        node_features=self._node_normalizer(node_features, is_training),
        edge_sets=[mesh_edges])

  def forward(self, inputs, is_training):
    # print('in cloth model build')
    graph = self._build_graph(inputs, is_training=is_training)
    '''
    if self.learned_model is None:
      self.learned_model = encode_process_decode.EncodeProcessDecode(
        output_size=self._params['size'],
        latent_size=128,
        num_layers=2,
        message_passing_steps=15,
        graph=graph)
    '''
    # show graph
    '''
    print('type of graph: ' + type(graph))
    print('type of graph.node_features: ' + type(graph.node_features))
    print('type of graph.edge_sets: ' + type(graph.edge_sets))
    quit()
    '''
    network_output = self.learned_model(graph)
    if is_training:
      # build target acceleration
      cur_position = inputs['world_pos']
      prev_position = inputs['prev|world_pos']
      target_position = inputs['target|world_pos']
      target_acceleration = target_position - 2*cur_position + prev_position
      target_normalized = self._output_normalizer(target_acceleration).cuda()

      # build loss
      loss_mask = torch.equal(inputs['node_type'][:, 0], torch.Tensor([common.NodeType.NORMAL.value]).int())
      error = torch.sum((target_normalized - network_output)**2, dim=1)
      loss = torch.mean(error[loss_mask])
      return loss
    else:
      return self._update(inputs, network_output)
  
  '''
  def loss(self, inputs):
    """L2 loss on position."""
    print('in loss')
    graph = self._build_graph(inputs, is_training=True)
    print('in loss, after build graph')
    network_output = self.learned_model(graph)

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
  '''
  '''
  def train(self, optimizer):
    for index, input in enumerate(self._ds_loader):
      # build target acceleration
      cur_position = input['world_pos']
      prev_position = input['prev|world_pos']
      target_position = input['target|world_pos']
      target_acceleration = target_position - 2*cur_position + prev_position
      target_normalized = self._output_normalizer(target_acceleration)

      # build loss
      graph = self._build_graph(input, is_training=True)
      if self.learned_model is None:
        self.learned_model = core_model.EncodeProcessDecode(
        output_size=self.params['size'],
        latent_size=128,
        num_layers=2,
        message_passing_steps=15,
        graph=graph)
      network_output = self.learned_model(graph)
      optimizer.zero_grad()
      loss = torch.nn.MSELoss(target_normalized, network_output)
      # loss_mask = torch.equal(input['node_type'][:, 0], common.NodeType.NORMAL)
      loss.backward()
  '''

  def _update(self, inputs, per_node_network_output):
    """Integrate model outputs."""
    acceleration = self._output_normalizer.inverse(per_node_network_output)
    # integrate forward
    cur_position = inputs['world_pos']
    prev_position = inputs['prev|world_pos']
    position = 2*cur_position + acceleration - prev_position
    return position
