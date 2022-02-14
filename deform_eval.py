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

    def step_fn(cur_pos, trajectory, stress_trajectory, cur_positions, cur_velocities, target_world_pos):
        # memory_prev = torch.cuda.memory_allocated(device) / (1024 * 1024)
        with torch.no_grad():
            prediction, cur_position, cur_velocity, stress = model({**initial_state, 'world_pos': cur_pos, 'target|world_pos': target_world_pos}, is_training=False)

        next_pos = torch.where(mask, prediction, target_world_pos)
        # next_pos = prediction
        # next_pos = torch.where(obstacle_mask, torch.squeeze(target_world_pos), next_pos)

        trajectory.append(next_pos)
        stress_trajectory.append(stress)
        cur_positions.append(cur_position)
        cur_velocities.append(cur_velocity)
        return next_pos, trajectory, stress_trajectory, cur_positions, cur_velocities

    cur_pos = torch.squeeze(initial_state['world_pos'], 0)
    trajectory = []
    stress_trajectory = []
    cur_positions = []
    cur_velocities = []
    for step in range(num_steps):
        cur_pos, trajectory, stress_trajectory, cur_positions, cur_velocities = step_fn(cur_pos, trajectory, stress_trajectory, cur_positions, cur_velocities, target_world_pos[step])
    return (torch.stack(trajectory), torch.stack(cur_positions), torch.stack(cur_velocities), torch.stack(stress_trajectory))

def evaluate(model, trajectory, num_steps=None):
    """Performs model rollouts and create stats."""
    initial_state = {k: torch.squeeze(v, 0)[0] for k, v in trajectory.items()}
    if num_steps is None:
        num_steps = trajectory['cells'].shape[0]
    prediction, cur_positions, cur_velocities, stress = _rollout(model, initial_state, num_steps, trajectory['target|world_pos'])

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
        'pred_pos': prediction,
        'cur_positions': cur_positions,
        'cur_velocities': cur_velocities,
        'stress': stress
    }
    return scalars, traj_ops
