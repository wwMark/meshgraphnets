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
"""Functions to build evaluation metrics for CFD data."""
import torch

import common

device = torch.device('cuda')


def _rollout(model, initial_state, num_steps):
    """Rolls out a model trajectory."""
    node_type = initial_state['node_type']
    mask = torch.logical_or(torch.eq(node_type[:, 0], torch.tensor([common.NodeType.NORMAL.value], device=device)),
                            torch.eq(node_type[:, 0], torch.tensor([common.NodeType.OUTFLOW.value], device=device)))
    mask = torch.stack((mask, mask), dim=1)

    def step_fn(velocity, trajectory):
        with torch.no_grad():
            prediction = model({**initial_state,
                                'velocity': velocity}, is_training=False)
            # don't update boundary nodes
            next_velocity = torch.where(mask, torch.squeeze(prediction), torch.squeeze(velocity))
            trajectory.append(velocity)
            return next_velocity, trajectory

    velocity = torch.squeeze(initial_state['velocity'], 0)
    trajectory = []
    for step in range(num_steps):
        velocity, trajectory = step_fn(velocity, trajectory)
    return torch.stack(trajectory)


def evaluate(model, trajectory):
    """Performs model rollouts and create stats."""
    initial_state = {k: torch.squeeze(v, 0)[0] for k, v in trajectory.items()}
    num_steps = trajectory['cells'].shape[0]
    prediction = _rollout(model, initial_state, num_steps)

    '''
    error = tf.reduce_mean((prediction - inputs['velocity'])**2, axis=-1)
    scalars = {'mse_%d_steps' % horizon: tf.reduce_mean(error[1:horizon+1])
               for horizon in [1, 10, 20, 50, 100, 200]}
    '''

    scalars = None

    traj_ops = {
        'faces': trajectory['cells'],
        'mesh_pos': trajectory['mesh_pos'],
        'gt_velocity': trajectory['velocity'],
        'pred_velocity': prediction
    }
    return scalars, traj_ops
