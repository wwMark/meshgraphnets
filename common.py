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
"""Commonly used data structures and functions."""

import enum
# import tensorflow.compat.v1 as tf
import torch


class NodeType(enum.IntEnum):
  NORMAL = 0
  OBSTACLE = 1
  AIRFOIL = 2
  HANDLE = 3
  INFLOW = 4
  OUTFLOW = 5
  WALL_BOUNDARY = 6
  SIZE = 9


def triangles_to_edges(faces):
  """Computes mesh edges from triangles."""
  # collect edges from triangles
  '''
  print("shape of faces", faces.shape)
  print("0-2", faces[:, 0:2])
  print("1-3", faces[:, 1:3])
  '''
  edges = torch.cat((faces[:, 0:2],
                     faces[:, 1:3],
                     torch.stack((faces[:, 2], faces[:, 0]), dim=1)), dim=0)
  # those edges are sometimes duplicated (within the mesh) and sometimes
  # single (at the mesh boundary).
  # sort & pack edges as single tf.int64
  receivers, _ = torch.min(edges, dim=1)
  senders, _ = torch.max(edges, dim=1)

  packed_edges = torch.stack((senders, receivers), dim=1)
  # remove duplicates and unpack
  unique_edges = torch.unique(packed_edges, return_inverse=False, return_counts=False, dim=0)
  senders, receivers = torch.unbind(unique_edges, dim=1)
  senders = senders.to(torch.int64)
  receivers = receivers.to(torch.int64)
  return (torch.cat((senders, receivers), dim=0), torch.cat((receivers, senders), dim=0))



  '''
  receivers = torch.unique(receivers)
  senders = torch.unique(senders)
  # create two-way connectivity
  se = torch.cat((senders, receivers), axis=0)
  se = se.to(torch.int64)
  re = torch.cat((receivers, senders), axis=0)
  re = re.to(torch.int64)
  return (se, re)
  '''
