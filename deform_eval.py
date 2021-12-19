"""Functions to build evaluation metrics for cloth data."""

import torch
import common
import numpy as np
import mpl_toolkits.mplot3d as p3d

device = torch.device('cuda')


def _rollout(model, initial_state, num_steps, target_world_pos):
    """Rolls out a model trajectory."""
    node_type = initial_state['node_type']
    mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.NORMAL.value], device=device))
    mask = torch.stack((mask, mask, mask), dim=1)

    obstacle_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.OBSTACLE.value], device=device))
    obstacle_mask = torch.stack((obstacle_mask, obstacle_mask, obstacle_mask), dim=1)

    def step_fn(cur_pos, trajectory, target_world_pos):
        # memory_prev = torch.cuda.memory_allocated(device) / (1024 * 1024)
        with torch.no_grad():
            prediction = model({**initial_state, 'world_pos': cur_pos}, is_training=False)

        next_pos = torch.where(mask, torch.squeeze(prediction), torch.squeeze(cur_pos))
        next_pos = torch.where(obstacle_mask, torch.squeeze(target_world_pos), torch.squeeze(next_pos))

        trajectory.append(cur_pos)
        return next_pos, trajectory

    cur_pos = torch.squeeze(initial_state['world_pos'], 0)
    trajectory = []
    for step in range(num_steps):
        cur_pos, trajectory = step_fn(cur_pos, trajectory, target_world_pos[step])
    return torch.stack(trajectory)

'''def to_polygons(faces, poses):
    trajectory_result = []
    step_result = []
    for faces_step, poses_step in zip(faces, poses):
        faces_step = faces_step.to('cpu')
        poses_step = poses_step.to('cpu')
        step_result.clear()
        for i in np.arange(len(faces_step)):
            square = [poses_step[faces_step[i, 0]], poses_step[faces_step[i, 1]], poses_step[faces_step[i, 2]], poses_step[faces_step[i, 3]]]
            face = p3d.art3d.Poly3DCollection(square)
            # face.set_color(colors.rgb2hex(sp.rand(3)))
            # face.set_edgecolor('k')
            # face.set_alpha(0.5)
            # p3d.add_collection3d(face)
            step_result.append(face)
        trajectory_result.append(step_result)
    return trajectory_result'''


def evaluate(model, trajectory):
    """Performs model rollouts and create stats."""
    initial_state = {k: torch.squeeze(v, 0)[0] for k, v in trajectory.items()}
    num_steps = trajectory['cells'].shape[0]
    prediction = _rollout(model, initial_state, num_steps, trajectory['target|world_pos'])

    # error = tf.reduce_mean((prediction - trajectory['world_pos'])**2, axis=-1)
    # scalars = {'mse_%d_steps' % horizon: tf.reduce_mean(error[1:horizon+1])
    #            for horizon in [1, 10, 20, 50, 100, 200]}

    scalars = None

    # temp solution for visualization

    faces = trajectory['cells']
    faces_result = []
    # print(faces.shape)
    for faces_step in faces:
        later = torch.cat((faces_step[:, 2:4], torch.unsqueeze(faces_step[:, 0], 1)), -1)
        faces_step = torch.cat((faces_step[:, 0:3], later), 0)
        faces_result.append(faces_step)
        # print(faces_step.shape)
    faces_result = torch.stack(faces_result, 0)
    # print(faces_result.shape)
    # print(faces_result[100].shape)


    # trajectory_polygons = to_polygons(trajectory['cells'], trajectory['world_pos'])

    traj_ops = {
        # 'faces': trajectory['cells'],
        'faces': faces_result,
        'mesh_pos': trajectory['mesh_pos'],
        # 'gt_pos': trajectory_polygons,
        'gt_pos': trajectory['world_pos'],
        'pred_pos': prediction
    }
    return scalars, traj_ops
