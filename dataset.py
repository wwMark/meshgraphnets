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
"""Utility functions for reading the datasets."""

import sys
import os

from numpy.lib.function_base import i0
cwd = os.getcwd()
sys.path.append(cwd + "/meshgraphnets/migration_utilities/")

import functools
import json
import flag_simple_torch_dataset
from flag_simple_torch_dataset import FlagSimpleDataset

import tensorflow.compat.v1 as tf
import torch
from torch.utils.data import DataLoader

from meshgraphnets.common import NodeType

def add_targets(fields, add_history):
  """Adds target and optionally history fields to dataframe."""
  def fn(trajectory):
    out = {}
    # print("printing trajectory size:")
    # print(trajectory)
    for key, val in trajectory.items():
      out[key] = val[1:-1]
      if key in fields:
        if add_history:
          out['prev|'+key] = val[0:-2]
        out['target|'+key] = val[2:]
    return out
  return fn

def split_and_preprocess(noise_field, noise_scale, noise_gamma):
  """Splits trajectories into frames, and adds training noise."""
  def add_noise(frame):
    # print("frame")
    # print(frame)
    print("noise field")
    print(noise_field)
    # print("frame[noise_field].size()")
    # print(frame[noise_field].size())
    zero_size = torch.zeros(frame[noise_field].size(), dtype=torch.float32)
    noise = torch.normal(zero_size, stddev=noise_scale)
    # don't apply noise to boundary nodes
    mask = torch.equal(frame['node_type'], NodeType.NORMAL)[:, 0]
    noise = torch.where(mask, noise, torch.zeros_like(noise))
    frame[noise_field] += noise
    frame['target|'+noise_field] += (1.0 - noise_gamma) * noise
    return frame
  '''
  ds = ds.flat_map(tf.data.Dataset.from_tensor_slices)
  ds = ds.map(add_noise, num_parallel_calls=8)
  ds = ds.shuffle(10000)
  ds = ds.repeat(None)

  return ds.prefetch(10)
  '''
  def element_operation(trajectory):
    # print("--------------trajectory world pos----------")
    # print(trajectory['world_pos'].size())
    world_pos = trajectory['world_pos']
    mesh_pos = trajectory['mesh_pos']
    node_type = trajectory['node_type']
    cells = trajectory['cells']
    target_world_pos = trajectory['target|world_pos']
    prev_world_pos = trajectory['prev|world_pos']
    split_trajectory = []
    for i in range(399):
      wp = add_noise(world_pos[i])
      mp = add_noise(mesh_pos[i])
      twp = add_noise(target_world_pos[i])
      nt = add_noise(node_type[i])
      c= add_noise(cells[i])
      pwp = add_noise(prev_world_pos[i])
      split_trajectory.append(wp, mp, twp, nt, c, pwp)
    # example = torch.flatten(frames, start_dim=-2)
    # example = add_noise(example)
    return torch.tensor(split_trajectory)
  
  return element_operation

# this function returns a torch dataloader
def load_dataset(path, split, add_targets=None, split_and_preprocess=None, batch_size=1):
  # DataLoader(FlagSimpleDataset(path='../../../mgn_dataset/flag_simple/', split='train'), batch_size=1)
  return DataLoader(FlagSimpleDataset(path=path, split=split, add_targets=add_targets, split_and_preprocess=split_and_preprocess), batch_size=batch_size, shuffle=True, prefetch_factor=10, num_workers=10)

def batch_dataset(ds, batch_size):
  """Batches input datasets."""
  shapes = ds.output_shapes
  types = ds.output_types
  def renumber(buffer, frame):
    nodes, cells = buffer
    new_nodes, new_cells = frame
    return nodes + new_nodes, tf.concat([cells, new_cells+nodes], axis=0)

  def batch_accumulate(ds_window):
    out = {}
    for key, ds_val in ds_window.items():
      initial = tf.zeros((0, shapes[key][1]), dtype=types[key])
      if key == 'cells':
        # renumber node indices in cells
        num_nodes = ds_window['node_type'].map(lambda x: tf.shape(x)[0])
        cells = tf.data.Dataset.zip((num_nodes, ds_val))
        initial = (tf.constant(0, tf.int32), initial)
        _, out[key] = cells.reduce(initial, renumber)
      else:
        merge = lambda prev, cur: tf.concat([prev, cur], axis=0)
        out[key] = ds_val.reduce(initial, merge)
    return out

  if batch_size > 1:
    ds = ds.window(batch_size, drop_remainder=True)
    ds = ds.map(batch_accumulate, num_parallel_calls=8)
  return ds
