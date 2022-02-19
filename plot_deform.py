"""Plots a cloth trajectory rollout."""
import os

import pickle

import pathlib

from absl import app
from absl import flags

from matplotlib import animation
import matplotlib.pyplot as plt

import math

import numpy as np
import mpl_toolkits.mplot3d as p3d

import torch

root_dir = pathlib.Path(__file__).parent.resolve()
output_dir = os.path.join(root_dir, 'output', 'deforming_plate')
all_subdirs = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if
               os.path.isdir(os.path.join(output_dir, d))]
latest_subdir = max(all_subdirs, key=os.path.getmtime)
rollout_path = os.path.join(latest_subdir, 'rollout', 'rollout.pkl')


def main(unused_argv):
    path_prefix = 'E:\\meshgraphnets\\output\\deforming_plate\\'
    path_suffix = 'rollout\\rollout.pkl'
    rollout_paths = ['Sat-Feb-19-15-44-13-2022']
    # path_prefix = '/home/kit/anthropomatik/sn2444/meshgraphnets/output/deforming_plate/'
    # rollout_paths = ['Mon-Jan-31-05-04-38-2022/2', 'Mon-Jan-31-05-10-30-2022/2', 'Mon-Jan-31-05-20-38-2022/2', 'Mon-Jan-31-05-35-42-2022/2', 'Mon-Jan-31-05-39-05-2022/2', 'Mon-Jan-31-08-28-21-2022/2']
    for rollout_path in rollout_paths:
        run_path = os.path.join(path_prefix, rollout_path)
        all_subdirs = [os.path.join(run_path, d) for d in os.listdir(run_path) if
                       os.path.isdir(os.path.join(run_path, d))]
        save_path = max(all_subdirs, key=os.path.getmtime)
        data_path = os.path.join(path_prefix, save_path, path_suffix)
        print("Ploting run", save_path)
        with open(data_path, 'rb') as fp:
            rollout_data = pickle.load(fp)
        fig = plt.figure(figsize=(19.2, 10.8))
        ax_origin = fig.add_subplot(121, projection='3d')
        ax_pred = fig.add_subplot(122, projection='3d')

        skip = 10
        num_steps = rollout_data[0]['gt_pos'].shape[0]
        # print(num_steps)
        num_frames = num_steps

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
            # step = (num * skip) % num_steps
            # traj = 2
            skip = 10
            traj = (num * skip) // num_steps
            step = (num * skip) % num_steps
            ax_origin.cla()
            ax_pred.cla()
            bound = bounds[traj]

            ax_origin.set_xlim([bound[0][0], bound[1][0]])
            ax_origin.set_ylim([bound[0][1], bound[1][1]])
            ax_origin.set_zlim([bound[0][2], bound[1][2]])

            ax_pred.set_xlim([bound[0][0], bound[1][0]])
            ax_pred.set_ylim([bound[0][1], bound[1][1]])
            ax_pred.set_zlim([bound[0][2], bound[1][2]])


            pos = torch.squeeze(rollout_data[traj]['pred_pos'], dim=0)[step].to('cpu')
            original_pos = torch.squeeze(rollout_data[traj]['gt_pos'], dim=0)[step].to('cpu')

            faces = torch.squeeze(rollout_data[traj]['faces'], dim=0)[step].to('cpu')

            ax_origin.plot_trisurf(original_pos[:, 0], original_pos[:, 1], faces, original_pos[:, 2], shade=True, alpha=0.3)
            ax_pred.plot_trisurf(pos[:, 0], pos[:, 1], faces, pos[:, 2], shade=True, alpha=0.3)

            ax_origin.set_title('ORIGIN Trajectory %d Step %d' % (traj, step))
            ax_pred.set_title('PRED Trajectory %d Step %d' % (traj, step))
            return fig,

        anima = animation.FuncAnimation(fig, animate, frames=math.floor(num_frames * 10), interval=100)
        # writervideo = animation.FFMpegWriter(fps=30)
        # anima.save(os.path.join(save_path, 'ani.mp4'), writer=writervideo)
        plt.show(block=True)


if __name__ == '__main__':
    app.run(main)
