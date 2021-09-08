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
"""Plots a cloth trajectory rollout."""

import pickle

from absl import app
from absl import flags

from matplotlib import animation
import matplotlib.pyplot as plt

import torch

FLAGS = flags.FLAGS
flags.DEFINE_string('rollout_path', 'C:\\Users\\Mark\\iCloudDrive\\master_arbeit\\implementation\\meshgraphnets\\rollout\\rollout.pkl', 'Path to rollout pickle file')


def main(unused_argv):
  with open(FLAGS.rollout_path, 'rb') as fp:
    rollout_data = pickle.load(fp)
  '''
  print("rollout_data")
  print(len(rollout_data[0]['pred_pos'][0]))
  print("")
  '''
  fig = plt.figure(figsize=(8, 8))
  ax = fig.add_subplot(111, projection='3d')
  skip = 10
  num_steps = rollout_data[0]['gt_pos'].shape[1]
  # print()
  num_frames = len(rollout_data) * num_steps // skip

  # compute bounds
  bounds = []
  index_temp = 0
  for trajectory in rollout_data:
    index_temp += 1
    # print("bb_min shape", trajectory['gt_pos'].shape)
    bb_min = torch.squeeze(trajectory['gt_pos'], dim=0).cpu().numpy().min(axis=(0, 1))
    bb_max = torch.squeeze(trajectory['gt_pos'], dim=0).cpu().numpy().max(axis=(0, 1))
    bounds.append((bb_min, bb_max))

  def animate(num):
    step = (num*skip) % num_steps
    traj = (num*skip) // num_steps
    ax.cla()
    bound = bounds[traj]
    '''
    print("bounds")
    print(bounds)
    print("bound......")
    print(bound)
    print("bound[][]")
    print(bound[0][0])
    print(bound[1][0])
    quit()
    '''
    ax.set_xlim([bound[0][0], bound[1][0]])
    ax.set_ylim([bound[0][1], bound[1][1]])
    ax.set_zlim([bound[0][2], bound[1][2]])

    # print("pos shape", rollout_data[traj]['pred_pos'].shape)
    # print("squeeze pos shape", torch.squeeze(rollout_data[traj]['pred_pos'], dim=0).shape)
    # print("step", step)
    pos = torch.squeeze(rollout_data[traj]['pred_pos'], dim=0)[step].cpu().detach()
    faces = torch.squeeze(rollout_data[traj]['faces'], dim=0)[step].cpu().detach()
    ax.plot_trisurf(pos[:, 0], pos[:, 1], faces, pos[:, 2], shade=True)
    ax.set_title('Trajectory %d Step %d' % (traj, step))
    return fig,

  _ = animation.FuncAnimation(fig, animate, frames=num_frames, interval=100)
  plt.show(block=True)


if __name__ == '__main__':
  app.run(main)
