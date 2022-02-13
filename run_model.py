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
"""Runs the learner/evaluator."""
import sys
import os
import pathlib
from pathlib import Path

import pickle
from absl import app
from absl import flags

import torch

import cloth_model
import cloth_eval
import cfd_model
import cfd_eval
import deform_model
import deform_eval

import dataset
import common
import logging

import numpy as np
import json
from common import NodeType

import time
import datetime

import csv

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import matplotlib

matplotlib.use('AGG')
import matplotlib.pyplot as plt

device = torch.device('cuda')

# train and evaluation configuration
FLAGS = flags.FLAGS
flags.DEFINE_enum('model', 'deform', ['cfd', 'cloth', 'deform'],
                  'Select model to run.')
flags.DEFINE_enum('mode', 'all', ['train', 'eval', 'all'],
                  'Train model, or run evaluation, or run both.')
flags.DEFINE_enum('rollout_split', 'valid', ['train', 'test', 'valid'],
                  'Dataset split to use for rollouts.')
flags.DEFINE_string('dataset', 'deforming_plate', ['flag_simple', 'cylinder_flow', 'deforming_plate'])

flags.DEFINE_integer('epochs', 2, 'No. of training epochs')
flags.DEFINE_integer('trajectories', 100, 'No. of training trajectories')
flags.DEFINE_integer('num_rollouts', 100, 'No. of rollout trajectories')

# core model configuration
flags.DEFINE_enum('core_model', 'encode_process_decode',
                  ['encode_process_decode'],
                  'Core model to be used')
flags.DEFINE_enum('message_passing_aggregator', 'sum', ['sum', 'max', 'min', 'mean', 'pna'], 'No. of training epochs')
flags.DEFINE_integer('message_passing_steps', 15, 'No. of training epochs')
flags.DEFINE_boolean('attention', False, 'whether attention is used or not')

# ripple method configuration
'''
ripple_used defines whether ripple is used, if not, core model of original paper will be used

ripple_generation defines how the ripples are generated:
    equal_size: all ripples have almost equal size of nodes
    gradient: ripples are generated according to node feature similarity
    exponential_size: ripples have a size that grows exponentially
    
ripple_node_selection defines how the nodes are selected from each ripple:
    random: a specific number of nodes are selected randomly from each ripple
    all: all nodes of the ripple are selected
    top: a specific number of nodes with the most influential features are selected

ripple_node_connection defines how the selected nodes of each ripple connect with each other to propagate message faster:
    most_influential: the most influential node connects all the other selected nodes
    fully_connected: all the selected nodes are connected with each other
    fully_ncross_connected: a specific number of nodes of the same ripple are connected with each other, and n randomly selected nodes from them will connect with n randomly selected nodes from another ripple
'''
flags.DEFINE_boolean('ripple_used', False, 'whether ripple is used or not')
flags.DEFINE_enum('ripple_generation', 'equal_size', ['equal_size', 'gradient', 'exponential_size', 'random_nodes', 'distance_density'],
                  'defines how ripples are generated')
flags.DEFINE_integer('ripple_generation_number', 5,
                     'defines how many ripples should be generated in equal size and gradient ripple generation; or the base in exponential size generation')
flags.DEFINE_enum('ripple_node_selection', 'top', ['random', 'all', 'top'],
                  'defines how the nodes are selected from each ripple')
flags.DEFINE_integer('ripple_node_selection_random_top_n', 3,
                     'defines how many nodes are selected from each ripple if node selection is random or top')
flags.DEFINE_enum('ripple_node_connection', 'most_influential',
                  ['most_influential', 'fully_connected', 'fully_ncross_connected'],
                  'defines how the selected nodes of each ripple connect with each other to propagate message faster')
flags.DEFINE_integer('ripple_node_ncross', 3,
                     'defines how many fully cross connections should be generated between ripples')

# directory setting
flags.DEFINE_string('model_last_run_dir',
                    None,
                    # os.path.join('E:\\meshgraphnets\\output\\deforming_plate', 'Sat-Feb-12-12-14-04-2022'),
                    # os.path.join('/home/i53/student/ruoheng_ma/meshgraphnets/output/deforming_plate', 'Mon-Jan--3-15-18-53-2022'),
                    'Path to the checkpoint file of a network that should continue training')

# decide whether to use the configuration from last run step
flags.DEFINE_boolean('use_prev_config', True, 'Decide whether to use the configuration from last run step')

# hpc max run time setting
flags.DEFINE_integer('hpc_default_max_time', 172800 - 3600 * 4, 'Max run time on hpc')
# flags.DEFINE_integer('hpc_default_max_time', 1500, 'Max run time on hpc')

PARAMETERS = {
    'cfd': dict(noise=0.02, gamma=1.0, field='velocity', history=False,
                size=2, batch=2, model=cfd_model, evaluator=cfd_eval, loss_type='cfd',
                stochastic_message_passing_used='False'),
    'cloth': dict(noise=0.001, gamma=0.1, field='world_pos', history=True,
                  size=3, batch=1, model=cloth_model, evaluator=cloth_eval, loss_type='cloth',
                  stochastic_message_passing_used='False'),
    'deform': dict(noise=0.003, gamma=0.1, field='world_pos', history=False,
                  size=3, batch=2, model=deform_model, evaluator=deform_eval, loss_type='deform',
                  stochastic_message_passing_used='False')
}

loaded_meta = False
shapes = {}
dtypes = {}
types = {}
steps = None

# store hpc start time for calculating rest running time
hpc_start_time = time.time()


# bwcluster max time limitation of gpu_8, in seconds
# leave 2 hours for possible evaluation
# hpc_default_max_time = 172800 - 3600 * 2
# hpc_default_max_time = 3 * 60
# hpc_max_time = hpc_start_time + hpc_default_max_time


def squeeze_data_frame(data_frame):
    for k, v in data_frame.items():
        data_frame[k] = torch.squeeze(v, 0)
    return data_frame


def add_targets(params):
    """Adds target and optionally history fields to dataframe."""
    fields = params['field']
    add_history = params['history']
    loss_type = params['loss_type']

    def fn(trajectory):
        if loss_type == 'deform':
            out = {}
            for key, val in trajectory.items():
                out[key] = val[0:-1]
                if key in fields:
                    if add_history:
                        out['prev|' + key] = val[0:-2]
                    out['target|' + key] = val[1:]
                if key == 'stress':
                    out['target|stress'] = val[1:]
            return out
        elif loss_type == 'cloth':
            out = {}
            for key, val in trajectory.items():
                out[key] = val[1:-1]
                if key in fields:
                    if add_history:
                        out['prev|' + key] = val[0:-2]
                    out['target|' + key] = val[2:]
            return out
    return fn


def split_and_preprocess(params, model_type):
    """Splits trajectories into frames, and adds training noise."""
    noise_field = params['field']
    noise_scale = params['noise']
    noise_gamma = params['gamma']
    loss_type = params['loss_type']

    def add_noise(frame):
        zero_size = torch.zeros(frame[noise_field].size(), dtype=torch.float32).to(device)
        noise = torch.normal(zero_size, std=noise_scale).to(device)
        other = torch.Tensor([NodeType.NORMAL.value]).to(device)
        mask = torch.eq(frame['node_type'], other.int())[:, 0]
        mask_sequence = []
        for i in range(noise.shape[1]):
            mask_sequence.append(mask)
        mask = torch.stack(mask_sequence, dim=1)
        noise = torch.where(mask, noise, torch.zeros_like(noise))
        frame[noise_field] += noise
        if loss_type == 'cloth':
            frame['target|' + noise_field] += (1.0 - noise_gamma) * noise
        return frame

    def element_operation(trajectory):
        trajectory_steps = []
        for i in range(steps):
            trajectory_step = {}
            for key, value in trajectory.items():
                trajectory_step[key] = value[i]
            noisy_trajectory_step = add_noise(trajectory_step)
            trajectory_steps.append(noisy_trajectory_step)
        return trajectory_steps

    return element_operation


def process_trajectory(trajectory_data, params, model_type, dataset_dir, add_targets_bool=False,
                       split_and_preprocess_bool=False):
    global loaded_meta
    global shapes
    global dtypes
    global types
    global steps

    if not loaded_meta:
        try:
            with open(os.path.join(dataset_dir, 'meta.json'), 'r') as fp:
                meta = json.loads(fp.read())
            shapes = {}
            dtypes = {}
            types = {}
            if params['loss_type'] == 'cloth':
                steps = meta['trajectory_length'] - 2
            elif params['loss_type'] == 'deform':
                steps = meta['trajectory_length'] - 1
            for key, field in meta['features'].items():
                shapes[key] = field['shape']
                dtypes[key] = field['dtype']
                types[key] = field['type']
        except FileNotFoundError as e:
            print(e)
            quit()
    trajectory = {}
    # decode bytes into corresponding dtypes
    for key, value in trajectory_data.items():
        raw_data = value.numpy().tobytes()
        mature_data = np.frombuffer(raw_data, dtype=getattr(np, dtypes[key]))
        mature_data = torch.from_numpy(mature_data).to(device)
        reshaped_data = torch.reshape(mature_data, shapes[key])
        if types[key] == 'static':
            reshaped_data = torch.tile(reshaped_data, (meta['trajectory_length'], 1, 1))
        elif types[key] == 'dynamic_varlen':
            pass
        elif types[key] != 'dynamic':
            raise ValueError('invalid data format')
        trajectory[key] = reshaped_data

    if add_targets_bool:
        trajectory = add_targets(params)(trajectory)
    if split_and_preprocess_bool:
        trajectory = split_and_preprocess(params, model_type)(trajectory)
    return trajectory

def pickle_save(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


'''
    Handles dataset preprocessing, model definition, training process definition and model training
'''


def learner(model, params, run_step_config):
    root_logger = logging.getLogger()

    loss_type = params['loss_type']
    model_type = run_step_config['model']

    # batch size can be defined in load_dataset. Default to 1.
    batch_size = 1
    prefetch_factor = 2

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1 + 1e-6, last_epoch=-1)
    trained_epoch = 0
    if run_step_config['last_run_dir'] is not None:
        optimizer.load_state_dict(
            torch.load(os.path.join(run_step_config['last_run_step_dir'], 'checkpoint', "optimizer_checkpoint.pth")))
        scheduler.load_state_dict(
            torch.load(os.path.join(run_step_config['last_run_step_dir'], 'checkpoint', "scheduler_checkpoint.pth")))
        epoch_checkpoint = torch.load(
            os.path.join(run_step_config['last_run_step_dir'], 'checkpoint', "epoch_checkpoint.pth"))
        trained_epoch = epoch_checkpoint['epoch'] + 1
        root_logger.info("Loaded optimizer, scheduler and model epoch checkpoint\n")

    # model training
    is_training = True

    epoch_training_losses = []

    count = 0
    pass_count = 500
    if run_step_config['model'] is not None:
        pass_count = 0
    all_trajectory_train_losses = []
    epoch_run_times = []
    hpc_max_time = hpc_start_time + FLAGS.hpc_default_max_time
    is_train_break = False
    for epoch in range(run_step_config['epochs'])[trained_epoch:]:

        # check whether the rest time is sufficient for running a whole epoch; stop running if not
        hpc_current_time = time.time()
        if len(epoch_run_times) != 0:
            epoch_mean_time = sum(epoch_run_times) // len(epoch_run_times)
            if hpc_current_time + epoch_mean_time >= hpc_max_time:
                root_logger.info("Exceed max run time, break training after finishing epoch " + str(epoch))
                is_train_break = True
                break

        ds_loader = dataset.load_dataset(run_step_config['dataset_dir'], 'train', batch_size=batch_size,
                                         prefetch_factor=prefetch_factor,
                                         add_targets=True, split_and_preprocess=True)
        # every time when model.train is called, model will train itself with the whole dataset
        root_logger.info("Epoch " + str(epoch + 1) + "/" + str(run_step_config['epochs']))
        epoch_training_loss = 0.0
        ds_iterator = iter(ds_loader)

        # decide single- or multi-gpu train
        gpu_count = torch.cuda.device_count()
        root_logger.info("Training with " + str(gpu_count) + " GPUs")

        for trajectory_index in range(run_step_config['trajectories']):
            root_logger.info(
                "    trajectory index " + str(trajectory_index + 1) + "/" + str(run_step_config['trajectories']))
            trajectory = next(ds_iterator)
            trajectory = process_trajectory(trajectory, params, model_type, run_step_config['dataset_dir'], True, True)
            trajectory_loss = 0.0
            for data_frame_index, data_frame in enumerate(trajectory):
                count += 1
                data_frame = squeeze_data_frame(data_frame)
                '''
                    Code for printing all node types in an input
                    deforming_plate dataset has node type [0, 1, 3]
                '''
                '''node_type_list = []
                for node_type_item in data_frame['node_type']:
                    node_type_list.append(node_type_item.item())
                print(list(set(node_type_list)))'''
                network_output = model(data_frame, is_training)
                loss = loss_fn(loss_type, data_frame, network_output, model, params)
                # if count % 1000 == 0:
                #     root_logger.info("    1000 step loss " + str(loss))
                if pass_count > 0:
                    pass_count -= 1
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    trajectory_loss += loss.detach().cpu()
            all_trajectory_train_losses.append(trajectory_loss)
            epoch_training_loss += trajectory_loss
            root_logger.info("        trajectory_loss")
            root_logger.info("        " + str(trajectory_loss))
            model.save_model(
                os.path.join(run_step_config['checkpoint_dir'],
                             "trajectory_model_checkpoint"))
            torch.save(optimizer.state_dict(),
                       os.path.join(run_step_config['checkpoint_dir'],
                                    "trajectory_optimizer_checkpoint" + ".pth"))
            torch.save(scheduler.state_dict(),
                       os.path.join(run_step_config['checkpoint_dir'],
                                    "trajectory_scheduler_checkpoint" + ".pth"))
        epoch_training_losses.append(epoch_training_loss)
        root_logger.info("Current mean of epoch training losses")
        root_logger.info(torch.mean(torch.stack(epoch_training_losses)))
        model.save_model(
            os.path.join(run_step_config['checkpoint_dir'],
                         "epoch_model_checkpoint"))
        torch.save(optimizer.state_dict(),
                   os.path.join(run_step_config['checkpoint_dir'],
                                "epoch_optimizer_checkpoint" + ".pth"))
        torch.save(scheduler.state_dict(),
                   os.path.join(run_step_config['checkpoint_dir'],
                                "epoch_scheduler_checkpoint" + ".pth"))
        if epoch == 20:
            scheduler.step()
            root_logger.info("Call scheduler in epoch " + str(epoch))
        torch.save({'epoch': epoch}, os.path.join(run_step_config['checkpoint_dir'], "epoch_checkpoint.pth"))
        epoch_run_times.append(time.time() - hpc_current_time)
    model.save_model(os.path.join(run_step_config['checkpoint_dir'], "model_checkpoint"))
    torch.save(optimizer.state_dict(), os.path.join(run_step_config['checkpoint_dir'], "optimizer_checkpoint.pth"))
    torch.save(scheduler.state_dict(), os.path.join(run_step_config['checkpoint_dir'], "scheduler_checkpoint.pth"))
    loss_record = {}
    loss_record['train_total_loss'] = torch.sum(torch.stack(epoch_training_losses))
    loss_record['train_mean_epoch_loss'] = torch.mean(torch.stack(epoch_training_losses)).item()
    loss_record['train_max_epoch_loss'] = torch.max(torch.stack(epoch_training_losses)).item()
    loss_record['train_min_epoch_loss'] = torch.min(torch.stack(epoch_training_losses)).item()
    loss_record['train_epoch_losses'] = epoch_training_losses
    loss_record['all_trajectory_train_losses'] = all_trajectory_train_losses
    loss_record['is_train_break'] = is_train_break
    return loss_record


def loss_fn(loss_type, inputs, network_output, model, params):
    """L2 loss on position."""
    # build target acceleration
    if loss_type == 'cloth':
        world_pos = inputs['world_pos']
        prev_world_pos = inputs['prev|world_pos']
        target_world_pos = inputs['target|world_pos']

        cur_position = world_pos
        prev_position = prev_world_pos
        target_position = target_world_pos
        target_acceleration = target_position - 2 * cur_position + prev_position
        target_normalized = model.get_output_normalizer()(target_acceleration).to(device)

        # build loss
        node_type = inputs['node_type']
        loss_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.NORMAL.value], device=device).int())
        error = torch.sum((target_normalized - network_output) ** 2, dim=1)
        loss = torch.mean(error[loss_mask])
        return loss
    elif loss_type == 'cfd':
        cur_velocity = inputs['velocity']
        target_velocity = inputs['target|velocity']
        target_velocity_change = target_velocity - cur_velocity
        target_normalized = model.get_output_normalizer()(target_velocity_change).to(device)

        # build loss
        node_type = inputs['node_type']
        loss_mask = torch.logical_or(
            torch.eq(node_type[:, 0], torch.tensor([common.NodeType.NORMAL.value], device=device)),
            torch.eq(node_type[:, 0], torch.tensor([common.NodeType.OUTFLOW.value], device=device)))
        error = torch.sum((target_normalized - network_output) ** 2, dim=1)
        loss = torch.mean(error[loss_mask])
        return loss
    elif loss_type == 'deform':
        world_pos = inputs['world_pos']
        target_world_pos = inputs['target|world_pos']
        target_stress = inputs['target|stress']

        cur_position = world_pos
        target_position = target_world_pos
        target_velocity = target_position - cur_position
        world_pos_normalizer, stress_normalizer = model.get_output_normalizer()
        target_normalized = world_pos_normalizer(target_velocity).to(device)
        target_normalized_stress = stress_normalizer(target_stress).to(device)

        # build loss
        # print(network_output[187])
        node_type = inputs['node_type']
        # loss_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.HANDLE.value], device=device).int())
        # loss_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.OBSTACLE.value], device=device).int())
        # loss_mask = torch.logical_not(loss_mask)
        # error = torch.sum((target_normalized - network_output) ** 2, dim=1)
        # loss = torch.mean(error[loss_mask])

        error = torch.sum((target_normalized - network_output) ** 2, dim=1) + torch.sum((target_normalized_stress - network_output) ** 2, dim=1)
        loss = torch.mean(error)
        return loss

def evaluator(params, model, run_step_config):
    root_logger = logging.getLogger()
    model_type = run_step_config['model']
    """Run a model rollout trajectory."""
    ds_loader = dataset.load_dataset(run_step_config['dataset_dir'], run_step_config['rollout_split'], add_targets=True)
    ds_iterator = iter(ds_loader)
    trajectories = []

    mse_losses = []
    l1_losses = []
    for index in range(run_step_config['num_rollouts']):
        root_logger.info("Evaluating trajectory " + str(index + 1))
        trajectory = next(ds_iterator)
        trajectory = process_trajectory(trajectory, params, model_type, run_step_config['dataset_dir'], True)
        _, prediction_trajectory = params['evaluator'].evaluate(model, trajectory)
        mse_loss_fn = torch.nn.MSELoss()
        l1_loss_fn = torch.nn.L1Loss()
        if model_type == 'cloth':
            mse_loss = mse_loss_fn(torch.squeeze(trajectory['world_pos'], dim=0), prediction_trajectory['pred_pos'])
            l1_loss = l1_loss_fn(torch.squeeze(trajectory['world_pos'], dim=0), prediction_trajectory['pred_pos'])
        elif model_type == 'cfd':
            mse_loss = mse_loss_fn(torch.squeeze(trajectory['velocity'], dim=0), prediction_trajectory['pred_velocity'])
            l1_loss = l1_loss_fn(torch.squeeze(trajectory['velocity'], dim=0), prediction_trajectory['pred_velocity'])
        elif model_type == 'deform':
            mse_loss = mse_loss_fn(torch.squeeze(trajectory['world_pos'], dim=0), prediction_trajectory['pred_pos'])
            l1_loss = l1_loss_fn(torch.squeeze(trajectory['world_pos'], dim=0), prediction_trajectory['pred_pos'])
        mse_losses.append(mse_loss.cpu())
        l1_losses.append(l1_loss.cpu())
        root_logger.info("    trajectory evaluation mse loss")
        root_logger.info("    " + str(mse_loss))
        root_logger.info("    trajectory evaluation l1 loss")
        root_logger.info("    " + str(l1_loss))
        trajectories.append(prediction_trajectory)
        # scalars.append(scalar_data)
    root_logger.info("mean mse loss of " + str(run_step_config['num_rollouts']) + " rollout trajectories")
    root_logger.info(torch.mean(torch.stack(mse_losses)))
    root_logger.info("mean l1 loss " + str(run_step_config['num_rollouts']) + " rollout trajectories")
    root_logger.info(torch.mean(torch.stack(l1_losses)))
    pickle_save(os.path.join(run_step_config['rollout_dir'], "rollout.pkl"), trajectories)
    loss_record = {}
    loss_record['eval_total_mse_loss'] = torch.sum(torch.stack(mse_losses)).item()
    loss_record['eval_total_l1_loss'] = torch.sum(torch.stack(l1_losses)).item()
    loss_record['eval_mean_mse_loss'] = torch.mean(torch.stack(mse_losses)).item()
    loss_record['eval_max_mse_loss'] = torch.max(torch.stack(mse_losses)).item()
    loss_record['eval_min_mse_loss'] = torch.min(torch.stack(mse_losses)).item()
    loss_record['eval_mean_l1_loss'] = torch.mean(torch.stack(l1_losses)).item()
    loss_record['eval_max_l1_loss'] = torch.max(torch.stack(l1_losses)).item()
    loss_record['eval_min_l1_loss'] = torch.min(torch.stack(l1_losses)).item()
    loss_record['eval_mse_losses'] = mse_losses
    loss_record['eval_l1_losses'] = l1_losses
    return loss_record

def n_step_evaluator(params, model, run_step_config, n_step_list, n_traj=1):
    model_type = run_step_config['model']

    ds_loader = dataset.load_dataset(run_step_config['dataset_dir'], run_step_config['rollout_split'], add_targets=True)
    ds_iterator = iter(ds_loader)

    n_step_mse_losses = {}
    n_step_l1_losses = {}

    # Take n_traj trajectories from valid set for n_step loss calculation
    for i in range(n_traj):
        trajectory = next(ds_iterator)
        trajectory = process_trajectory(trajectory, params, model_type, run_step_config['dataset_dir'], True)
        for n_step in n_step_list:
            mse_losses = []
            l1_losses = []
            for step in range(len(trajectory['world_pos']) - n_step):
                eval_traj = {}
                for k, v in trajectory.items():
                    eval_traj[k] = v[step:step + n_step + 1]
                _, prediction_trajectory = params['evaluator'].evaluate(model, eval_traj, n_step + 1)
                mse_loss_fn = torch.nn.MSELoss()
                l1_loss_fn = torch.nn.L1Loss()
                if model_type == 'cloth':
                    mse_loss = mse_loss_fn(torch.squeeze(eval_traj['world_pos'], dim=0), prediction_trajectory['pred_pos'])
                    l1_loss = l1_loss_fn(torch.squeeze(eval_traj['world_pos'], dim=0), prediction_trajectory['pred_pos'])
                elif model_type == 'cfd':
                    mse_loss = mse_loss_fn(torch.squeeze(eval_traj['velocity'], dim=0), prediction_trajectory['pred_velocity'])
                    l1_loss = l1_loss_fn(torch.squeeze(eval_traj['velocity'], dim=0), prediction_trajectory['pred_velocity'])
                elif model_type == 'deform':
                    mse_loss = mse_loss_fn(torch.squeeze(eval_traj['world_pos'], dim=0), prediction_trajectory['pred_pos'])
                    l1_loss = l1_loss_fn(torch.squeeze(eval_traj['world_pos'], dim=0), prediction_trajectory['pred_pos'])
                mse_losses.append(mse_loss.cpu())
                l1_losses.append(l1_loss.cpu())
            if n_step not in n_step_mse_losses and n_step not in n_step_l1_losses:
                n_step_mse_losses[n_step] = torch.stack(mse_losses)
                n_step_l1_losses[n_step] = torch.stack(l1_losses)
            elif n_step in n_step_mse_losses and n_step in n_step_l1_losses:
                n_step_mse_losses[n_step] = n_step_mse_losses[n_step] + torch.stack(mse_losses)
                n_step_l1_losses[n_step] = n_step_l1_losses[n_step] + torch.stack(l1_losses)
            else:
                raise Exception('Error when computing n step losses!')
    for (kmse, vmse), (kl1, vl1) in zip(n_step_mse_losses.items(), n_step_l1_losses.items()):
        n_step_mse_losses[kmse] = torch.div(vmse, i + 1)
        n_step_l1_losses[kl1] = torch.div(vl1, i + 1)

    return {'n_step_mse_loss': n_step_mse_losses, 'n_step_l1_loss': n_step_l1_losses}


def plot_data(data):
    return None


def find_nth_latest_run_step(run_dir, n):
    all_run_step_dirs = os.listdir(run_dir)
    all_run_step_dirs = map(lambda d: os.path.join(run_dir, d), all_run_step_dirs)
    all_run_step_dirs = [d for d in all_run_step_dirs if os.path.isdir(d)]
    nth_latest_run_step_dir = sorted(all_run_step_dirs, key=os.path.getmtime)[-n]
    return nth_latest_run_step_dir


def prepare_files_and_directories(last_run_dir, output_dir):
    '''
        The following code is about creating all the necessary files and directories for the run
    '''
    # if last run dir is not specified, then new run dir should be created, otherwise use run specified by argument
    if last_run_dir is not None:
        run_dir = last_run_dir
    else:
        run_create_time = time.time()
        run_create_datetime = datetime.datetime.fromtimestamp(run_create_time).strftime('%c')
        run_create_datetime_datetime_dash = run_create_datetime.replace(" ", "-").replace(":", "-")
        run_dir = os.path.join(output_dir, run_create_datetime_datetime_dash)
        Path(run_dir).mkdir(parents=True, exist_ok=True)

    # check for last run step dir and if exists, create a new run step dir with incrementing dir name, otherwise create the first run step dir
    all_run_step_dirs = os.listdir(run_dir)
    if not all_run_step_dirs:
        run_step_dir = os.path.join(run_dir, '1')
    else:
        latest_run_step_dir = find_nth_latest_run_step(run_dir, 1)
        run_step_dir = str(int(Path(latest_run_step_dir).name) + 1)
        run_step_dir = os.path.join(run_dir, run_step_dir)

    # make all the necessary directories
    checkpoint_dir = os.path.join(run_step_dir, 'checkpoint')
    log_dir = os.path.join(run_step_dir, 'log')
    rollout_dir = os.path.join(run_step_dir, 'rollout')
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(rollout_dir).mkdir(parents=True, exist_ok=True)

    return run_step_dir


def logger_setup(log_path):
    # set log configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # console_output_handler = logging.StreamHandler(sys.stdout)
    # console_output_handler.setLevel(logging.INFO)
    file_log_handler = logging.FileHandler(filename=log_path, mode='w', encoding='utf-8')
    file_log_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s - %(message)s')
    # console_output_handler.setFormatter(formatter)
    file_log_handler.setFormatter(formatter)
    # root_logger.addHandler(console_output_handler)
    root_logger.addHandler(file_log_handler)
    return root_logger


def log_run_summary(root_logger, run_step_config, run_step_dir):
    root_logger.info("")
    root_logger.info("=======================Run Summary=======================")
    root_logger.info("Simulation task is " + str(run_step_config['model']) + " simulation")
    root_logger.info("Mode is " + run_step_config['mode'])
    if run_step_config['mode'] == 'eval' or run_step_config['mode'] == 'all':
        root_logger.info("Evaluation set is " + run_step_config['rollout_split'])
    elif run_step_config['mode'] == 'train':
        root_logger.info("No Evaluation")
    root_logger.info(
        "Train and/or evaluation configuration are " + str(run_step_config['epochs']) + " epochs, " + str(
            run_step_config['trajectories']) + " trajectories each epoch, number of rollouts is " + str(
            run_step_config['num_rollouts']))
    root_logger.info("Core model is " + run_step_config['core_model'])
    root_logger.info("Message passing aggregator is " + run_step_config['message_passing_aggregator'])
    root_logger.info("Message passing steps are " + str(run_step_config['message_passing_steps']))
    root_logger.info("Attention used is " + str(run_step_config['attention']))
    root_logger.info("Ripple used is " + str(run_step_config['ripple_used']))
    if run_step_config['ripple_used']:
        root_logger.info("  Ripple generation method is " + str(run_step_config['ripple_generation']))
        root_logger.info("  Ripple generation number is " + str(run_step_config['ripple_generation_number']))
        root_logger.info("  Ripple node selection method is " + str(run_step_config['ripple_node_selection']))
        root_logger.info(
            "  Ripple node selection number is " + str(run_step_config['ripple_node_selection_random_top_n']))
        root_logger.info("  Ripple node connection method is " + str(run_step_config['ripple_node_connection']))
        root_logger.info("  Ripple node ncross number is " + str(run_step_config['ripple_node_ncross']))
    root_logger.info("Run output directory is " + run_step_dir)
    root_logger.info("=========================================================")
    root_logger.info("")


def main(argv):
    # record start time
    run_step_start_time = time.time()
    run_step_start_datetime = datetime.datetime.fromtimestamp(run_step_start_time).strftime('%c')

    # load config from previous run step if last run dir is specified
    last_run_dir = FLAGS.model_last_run_dir
    use_prev_config = FLAGS.use_prev_config
    continue_prev_run = False
    if last_run_dir is not None and use_prev_config:
        last_run_step_dir = find_nth_latest_run_step(last_run_dir, 1)
        print(last_run_dir)
        run_step_config = pickle_load(os.path.join(last_run_step_dir, 'log', 'config.pkl'))
        run_step_config['last_run_dir'] = last_run_dir
        run_step_config['last_run_step_dir'] = last_run_step_dir
        dataset_name = run_step_config['dataset']
        continue_prev_run = True
    else:
        dataset_name = FLAGS.dataset

    # setup directory structure for saving checkpoint, train configuration, rollout result and log
    root_dir = pathlib.Path(__file__).parent.resolve()
    # dataset_dir = os.path.join('/home/temp_store/ruoheng_ma', 'data', dataset_name)
    dataset_dir = os.path.join('data', dataset_name)
    output_dir = os.path.join(root_dir, 'output', dataset_name)
    run_step_dir = prepare_files_and_directories(last_run_dir, output_dir)
    checkpoint_dir = os.path.join(run_step_dir, 'checkpoint')
    log_dir = os.path.join(run_step_dir, 'log')
    rollout_dir = os.path.join(run_step_dir, 'rollout')

    # setup logger
    root_logger = logger_setup(os.path.join(log_dir, 'log.log'))
    if continue_prev_run:
        root_logger.info("=========================================================")
        root_logger.info("Continue run in " + str(run_step_dir))
        root_logger.info("=========================================================")

    # if last run dir is not specified, save all the run configuration in log dir for next run, otherwise load the last run step configuration and continue the last run
    if last_run_dir is None or not use_prev_config:
        run_step_config = {'model': FLAGS.model, 'mode': FLAGS.mode, 'rollout_split': FLAGS.rollout_split,
                           'dataset': FLAGS.dataset, 'epochs': FLAGS.epochs, 'trajectories': FLAGS.trajectories,
                           'num_rollouts': FLAGS.num_rollouts, 'core_model': FLAGS.core_model,
                           'message_passing_aggregator': FLAGS.message_passing_aggregator,
                           'message_passing_steps': FLAGS.message_passing_steps, 'attention': FLAGS.attention,
                           'ripple_used': FLAGS.ripple_used,
                           'ripple_generation': FLAGS.ripple_generation,
                           'ripple_generation_number': FLAGS.ripple_generation_number,
                           'ripple_node_selection': FLAGS.ripple_node_selection,
                           'ripple_node_selection_random_top_n': FLAGS.ripple_node_selection_random_top_n,
                           'ripple_node_connection': FLAGS.ripple_node_connection,
                           'ripple_node_ncross': FLAGS.ripple_node_ncross, 'dataset_dir': dataset_dir,
                           'last_run_dir': None}
        root_logger.info("=========================================================")
        root_logger.info("Start new run in " + str(run_step_dir))
        root_logger.info("=========================================================")
    run_step_config['checkpoint_dir'] = checkpoint_dir
    run_step_config['rollout_dir'] = rollout_dir
    run_step_config_save_path = os.path.join(log_dir, 'config.pkl')
    Path(run_step_config_save_path).touch()
    pickle_save(run_step_config_save_path, run_step_config)

    # save program configuration in file title
    run_config_record = str(run_step_config['mode']) + "_epoch" + str(run_step_config['epochs']) + "_trajectory" + str(
        run_step_config['trajectories']) + "_rollout" + str(run_step_config['num_rollouts'])
    Path(os.path.join(run_step_dir, run_config_record)).touch()

    root_logger.info("Program started at time " + str(run_step_start_datetime))
    params = PARAMETERS[run_step_config['model']]

    # create or load model
    root_logger.info("Start training......")
    model = params['model'].Model(params, run_step_config['core_model'], run_step_config['message_passing_aggregator'],
                                  run_step_config['message_passing_steps'], run_step_config['attention'],
                                  run_step_config['ripple_used'],
                                  run_step_config['ripple_generation'], run_step_config['ripple_generation_number'],
                                  run_step_config['ripple_node_selection'],
                                  run_step_config['ripple_node_selection_random_top_n'],
                                  run_step_config['ripple_node_connection'],
                                  run_step_config['ripple_node_ncross'])
    if last_run_dir is not None:
        last_run_step_dir = find_nth_latest_run_step(last_run_dir, 2)
        model.load_model(os.path.join(last_run_step_dir, 'checkpoint', "model_checkpoint"))
        root_logger.info(
            "Loaded checkpoint file in " + str(
                os.path.join(last_run_step_dir, 'checkpoint')) + " and starting retraining...")
    model.to(device)

    # run summary
    log_run_summary(root_logger, run_step_config, run_step_dir)

    is_train_break = False
    train_loss_record = None
    if run_step_config['mode'] == 'train' or run_step_config['mode'] == 'all':
        # record train time
        train_start = time.time()
        train_loss_record = learner(model, params, run_step_config)
        train_end = time.time()
        train_elapsed_time_in_second = train_end - train_start

        is_train_break = train_loss_record['is_train_break']

        # load train loss if exist and combine the previous and current train loss
        if last_run_dir is not None:
            saved_train_loss_record = pickle_load(os.path.join(last_run_step_dir, 'log', 'train_loss.pkl'))
            train_loss_record['train_epoch_losses'] = saved_train_loss_record['train_epoch_losses'] + \
                                                      train_loss_record['train_epoch_losses']
            train_loss_record['train_total_loss'] = torch.sum(torch.stack(train_loss_record['train_epoch_losses']))
            train_loss_record['train_mean_epoch_loss'] = torch.mean(
                torch.stack(train_loss_record['train_epoch_losses'])).item()
            train_loss_record['train_max_epoch_loss'] = torch.max(
                torch.stack(train_loss_record['train_epoch_losses'])).item()
            train_loss_record['train_min_epoch_loss'] = torch.min(
                torch.stack(train_loss_record['train_epoch_losses'])).item()
            train_loss_record['all_trajectory_train_losses'] = saved_train_loss_record['all_trajectory_train_losses'] + \
                                                               train_loss_record['all_trajectory_train_losses']
            # load train elapsed time if exist and combine the previous and current train loss
            saved_train_elapsed_time_in_second = pickle_load(
                os.path.join(last_run_step_dir, 'log', 'train_elapsed_time_in_second.pkl'))
            train_elapsed_time_in_second += saved_train_elapsed_time_in_second
        train_elapsed_time_in_second_pkl_file = os.path.join(log_dir, 'train_elapsed_time_in_second.pkl')
        Path(train_elapsed_time_in_second_pkl_file).touch()
        pickle_save(train_elapsed_time_in_second_pkl_file, train_elapsed_time_in_second)
        train_mean_elapsed_time = str(datetime.timedelta(
            seconds=train_elapsed_time_in_second // (run_step_config['epochs'] * run_step_config['trajectories'])))
        train_elapsed_time = str(datetime.timedelta(seconds=train_elapsed_time_in_second))

        # save train loss
        train_loss_pkl_file = os.path.join(log_dir, 'train_loss.pkl')
        Path(train_loss_pkl_file).touch()
        pickle_save(train_loss_pkl_file, train_loss_record)

        root_logger.info("Finished training......")
    if run_step_config['mode'] == 'eval' or run_step_config['mode'] == 'all':
        root_logger.info("Start evaluating......")
        model.evaluate()
        model.to(device)
        eval_loss_record = evaluator(params, model, run_step_config)
        step_loss = n_step_evaluator(params, model, run_step_config, n_step_list=[1], n_traj=1)
        if last_run_dir is not None and train_loss_record is None:
            train_loss_record = pickle_load(os.path.join(last_run_step_dir, 'log', 'train_loss.pkl'))
        root_logger.info("Finished evaluating......")
    run_step_end_time = time.time()
    run_step_end_datetime = datetime.datetime.fromtimestamp(run_step_end_time).strftime('%c')
    root_logger.info("Program ended at time " + run_step_end_datetime)
    elapsed_time_in_second = run_step_end_time - run_step_start_time
    if last_run_dir is not None:
        saved_elapsed_time_in_second = pickle_load(os.path.join(last_run_step_dir, 'log', 'elapsed_time_in_second.pkl'))
        elapsed_time_in_second += saved_elapsed_time_in_second
    elapsed_time = str(datetime.timedelta(seconds=elapsed_time_in_second))
    elapsed_time_in_second_pkl_file = os.path.join(log_dir, 'elapsed_time_in_second.pkl')
    Path(elapsed_time_in_second_pkl_file).touch()
    pickle_save(elapsed_time_in_second_pkl_file, elapsed_time_in_second)

    # run summary
    log_run_summary(root_logger, run_step_config, run_step_dir)
    root_logger.info("Run total elapsed time " + elapsed_time + "\n")

    root_logger.info("--------------------train loss record--------------------")
    if run_step_config['mode'] == 'train' or run_step_config['mode'] == 'all':
        for item in train_loss_record.items():
            root_logger.info(item)
    root_logger.info("---------------------------------------------------------")
    root_logger.info("")
    root_logger.info("--------------------eval loss record---------------------")
    if run_step_config['mode'] == 'eval' or run_step_config['mode'] == 'all':
        eval_loss_pkl_file = os.path.join(log_dir, 'eval_loss.pkl')
        Path(eval_loss_pkl_file).touch()
        pickle_save(eval_loss_pkl_file, eval_loss_record)
        for item in eval_loss_record.items():
            root_logger.info(item)
        step_loss_mse_pkl_file = os.path.join(log_dir, 'step_loss_mse.pkl')
        Path(step_loss_mse_pkl_file).touch()
        pickle_save(step_loss_mse_pkl_file, step_loss['n_step_mse_loss'])
        step_loss_l1_pkl_file = os.path.join(log_dir, 'step_loss_l1.pkl')
        Path(step_loss_l1_pkl_file).touch()
        pickle_save(step_loss_l1_pkl_file, step_loss['n_step_l1_loss'])
    root_logger.info("---------------------------------------------------------")

    # save result in figure
    fig_train = plt.figure(figsize=(38.4, 21.6), constrained_layout=True)
    fig_eval = plt.figure(figsize=(38.4, 21.6), constrained_layout=True)
    gs_train = fig_train.add_gridspec(2, 1)
    gs_eval = fig_eval.add_gridspec(1, 2)
    description = []
    delimiter = ", "
    description.append("Simulation model is " + str(FLAGS.model) + delimiter)
    description.append("Finished FLAGS.mode " + FLAGS.mode + delimiter)
    if FLAGS.mode == 'eval' or FLAGS.mode == 'all':
        description.append("Evaluation set is " + FLAGS.rollout_split + delimiter)
    elif FLAGS.mode == 'train':
        description.append("No Evaluation" + delimiter)
    description.append("Core model is " + FLAGS.core_model + delimiter)
    description.append("Message passing aggregator is " + FLAGS.message_passing_aggregator + delimiter)
    description.append("Message passing steps are " + str(FLAGS.message_passing_steps) + delimiter)
    description.append("Attention used is " + str(FLAGS.attention) + delimiter)
    description.append("Ripple used is " + str(FLAGS.ripple_used) + delimiter)
    if FLAGS.ripple_used:
        description.append("    Ripple generation method is " + str(FLAGS.ripple_generation) + delimiter)
        description.append("    Ripple generation number is " + str(FLAGS.ripple_generation_number) + delimiter)
        description.append("    Ripple node selection method is " + str(FLAGS.ripple_node_selection) + delimiter)
        description.append(
            "    Ripple node selection number is " + str(FLAGS.ripple_node_selection_random_top_n) + delimiter)
        description.append("    Ripple node connection method is " + str(FLAGS.ripple_node_connection) + delimiter)
        description.append("    Ripple node ncross number is " + str(FLAGS.ripple_node_ncross) + delimiter)
    description.append("Elapsed time " + elapsed_time + delimiter)
    if FLAGS.mode == 'train' or FLAGS.mode == 'all':
        description.append("Train mean elapsed time " + train_mean_elapsed_time + delimiter)
    description_txt = ""
    for item in description:
        description_txt += item
    # plt.figtext(0.5, 0.01, description_txt, wrap=True, horizontalalignment='left', fontsize=22)
    if FLAGS.mode == 'train' or FLAGS.mode == 'all':
        train_loss_ax = fig_train.add_subplot(gs_train[0, 0])
        all_trajectory_train_losses_ax = fig_train.add_subplot(gs_train[1, 0])

        train_loss_ax.set_title('Train Loss', fontsize=68)
        train_loss_ax.set_xlabel('Epoch', fontsize=50)
        train_loss_ax.set_ylabel('Loss', fontsize=50)
        train_loss_ax.tick_params(axis='both', labelsize=34)
        train_loss_ax.yaxis.get_offset_text().set_fontsize(34)

        all_trajectory_train_losses_ax.set_title('Train trajectory Loss', fontsize=68)
        all_trajectory_train_losses_ax.set_xlabel('Trajectory No.', fontsize=50)
        all_trajectory_train_losses_ax.set_ylabel('Loss', fontsize=50)
        all_trajectory_train_losses_ax.tick_params(axis='both', labelsize=34)
        all_trajectory_train_losses_ax.yaxis.get_offset_text().set_fontsize(34)

        train_loss_ax.plot(range(1, len(train_loss_record['train_epoch_losses']) + 1),
                           train_loss_record['train_epoch_losses'])
        all_trajectory_train_losses_ax.plot(range(1, len(train_loss_record['all_trajectory_train_losses']) + 1),
                                            train_loss_record['all_trajectory_train_losses'])
        fig_train.savefig(os.path.join(log_dir, "Train_Loss.png"))
    if FLAGS.mode == 'eval' or FLAGS.mode == 'all':
        eval_mse_loss_ax = fig_eval.add_subplot(gs_eval[0, 0])
        eval_l1_loss_ax = fig_eval.add_subplot(gs_eval[0, 1])

        eval_mse_loss_ax.set_title('Eval MSE Loss', fontsize=68)
        eval_mse_loss_ax.set_xlabel('Rollout No.', fontsize=50)
        eval_mse_loss_ax.set_ylabel('Loss', fontsize=50)
        eval_mse_loss_ax.tick_params(axis='both', labelsize=34)
        eval_mse_loss_ax.yaxis.get_offset_text().set_fontsize(34)

        eval_l1_loss_ax.set_title('Eval L1 Loss', fontsize=68)
        eval_l1_loss_ax.set_xlabel('Rollout No.', fontsize=50)
        eval_l1_loss_ax.set_ylabel('Loss', fontsize=50)
        eval_l1_loss_ax.tick_params(axis='both', labelsize=34)
        eval_l1_loss_ax.yaxis.get_offset_text().set_fontsize(34)

        eval_mse_loss_ax.plot(range(1, len(eval_loss_record['eval_mse_losses']) + 1),
                              eval_loss_record['eval_mse_losses'], 'o')
        eval_l1_loss_ax.plot(range(1, len(eval_loss_record['eval_l1_losses']) + 1), eval_loss_record['eval_l1_losses'],
                             'o')
        fig_eval.savefig(os.path.join(log_dir, "Eval_Loss.png"))

        # step loss figure

        fig_step_loss = plt.figure(figsize=(38.4, 21.6), constrained_layout=True)
        gs_step_loss = fig_step_loss.add_gridspec(1, 2)

        step_loss_mse_loss_ax = fig_step_loss.add_subplot(gs_step_loss[0, 0])
        step_loss_l1_loss_ax = fig_step_loss.add_subplot(gs_step_loss[0, 1])

        step_loss_mse_loss_ax.set_title('Step MSE Loss', fontsize=68)
        step_loss_mse_loss_ax.set_xlabel('Trajectory Step', fontsize=50)
        step_loss_mse_loss_ax.set_ylabel('Loss', fontsize=50)
        step_loss_mse_loss_ax.tick_params(axis='both', labelsize=34)
        step_loss_mse_loss_ax.yaxis.get_offset_text().set_fontsize(34)

        step_loss_l1_loss_ax.set_title('Step L1 Loss', fontsize=68)
        step_loss_l1_loss_ax.set_xlabel('Trajectory Step', fontsize=50)
        step_loss_l1_loss_ax.set_ylabel('Loss', fontsize=50)
        step_loss_l1_loss_ax.tick_params(axis='both', labelsize=34)
        step_loss_l1_loss_ax.yaxis.get_offset_text().set_fontsize(34)

        for k, v in step_loss['n_step_mse_loss'].items():
            label = str(k) + " step prediction"
            step_loss_mse_loss_ax.plot(range(1, len(v) + 1), v, 'o', label=label)
        step_loss_mse_loss_ax.legend(fontsize=40)
        for k, v in step_loss['n_step_l1_loss'].items():
            label = str(k) + " step prediction"
            step_loss_l1_loss_ax.plot(range(1, len(v) + 1), v, 'o', label=label)
        step_loss_l1_loss_ax.legend(fontsize=40)

        fig_step_loss.savefig(os.path.join(log_dir, "Eval_Step_Loss.png"))

    # save max, min and mean value of train and eval losses as csv
    csv_path = os.path.join(log_dir, 'result.csv')
    Path(csv_path).touch()
    entry = []
    if FLAGS.mode == 'all':
        entry = []
        entry.append(["Simulation model", str(FLAGS.model)])
        entry.append(["Finished FLAGS.mode", FLAGS.mode])
        entry.append(["Train epochs", FLAGS.epochs])
        entry.append(["Epoch trajectories", FLAGS.trajectories])
        entry.append(["Rollouts", FLAGS.num_rollouts])
        entry.append(["Evaluation set", FLAGS.rollout_split])
        entry.append(["Core model", FLAGS.core_model])
        entry.append(["Message passing aggregator", FLAGS.message_passing_aggregator])
        entry.append(["Message passing steps", str(FLAGS.message_passing_steps)])
        entry.append(["Attention used", str(FLAGS.attention)])
        if FLAGS.ripple_used:
            entry.append(["Ripple used", str(FLAGS.ripple_used)])
            entry.append(["Ripple generation method", str(FLAGS.ripple_generation)])
            entry.append(["Ripple generation number", str(FLAGS.ripple_generation_number)])
            entry.append(["Ripple node selection method", str(FLAGS.ripple_node_selection)])
            entry.append(["Ripple node selection number", str(FLAGS.ripple_node_selection_random_top_n)])
            entry.append(["Ripple node connection method", str(FLAGS.ripple_node_connection)])
            entry.append(["Ripple node ncross number", str(FLAGS.ripple_node_ncross)])
        else:
            entry.append(["Ripple used", str(FLAGS.ripple_used)])
            entry.append(["Ripple generation method", ""])
            entry.append(["Ripple generation number", ""])
            entry.append(["Ripple node selection method", ""])
            entry.append(["Ripple node selection number", ""])
            entry.append(["Ripple node connection method", ""])
            entry.append(["Ripple node ncross number", ""])
        entry.append(["Elapsed time", elapsed_time])
        entry.append(["Train mean elapsed time", train_mean_elapsed_time])
        entry.append(["Mean train epoch loss", str(train_loss_record['train_mean_epoch_loss'])])
        entry.append(["Max train epoch loss", str(train_loss_record['train_max_epoch_loss'])])
        entry.append(["Min train epoch loss", str(train_loss_record['train_min_epoch_loss'])])
        entry.append(["Mean eval mse loss", str(eval_loss_record['eval_mean_mse_loss'])])
        entry.append(["Max eval mse loss", str(eval_loss_record['eval_max_mse_loss'])])
        entry.append(["Min eval mse loss", str(eval_loss_record['eval_min_mse_loss'])])
        entry.append(["Mean eval l1 loss", str(eval_loss_record['eval_mean_l1_loss'])])
        entry.append(["Max eval l1 loss", str(eval_loss_record['eval_max_l1_loss'])])
        entry.append(["Min eval l1 loss", str(eval_loss_record['eval_min_l1_loss'])])
    elif FLAGS.mode == 'train':
        entry = []
        entry.append(["Simulation model", str(FLAGS.model)])
        entry.append(["Finished FLAGS.mode", FLAGS.mode])
        entry.append(["Train epochs", FLAGS.epochs])
        entry.append(["Epoch trajectories", FLAGS.trajectories])
        entry.append(["Rollouts", ""])
        entry.append(["Evaluation set", ""])
        entry.append(["Core model", FLAGS.core_model])
        entry.append(["Message passing aggregator", FLAGS.message_passing_aggregator])
        entry.append(["Message passing steps", str(FLAGS.message_passing_steps)])
        entry.append(["Attention used", str(FLAGS.attention)])
        if FLAGS.ripple_used:
            entry.append(["Ripple used", str(FLAGS.ripple_used)])
            entry.append(["Ripple generation method", str(FLAGS.ripple_generation)])
            entry.append(["Ripple generation number", str(FLAGS.ripple_generation_number)])
            entry.append(["Ripple node selection method", str(FLAGS.ripple_node_selection)])
            entry.append(["Ripple node selection number", str(FLAGS.ripple_node_selection_random_top_n)])
            entry.append(["Ripple node connection method", str(FLAGS.ripple_node_connection)])
            entry.append(["Ripple node ncross number", str(FLAGS.ripple_node_ncross)])
        else:
            entry.append(["Ripple used", str(FLAGS.ripple_used)])
            entry.append(["Ripple generation method", ""])
            entry.append(["Ripple generation number", ""])
            entry.append(["Ripple node selection method", ""])
            entry.append(["Ripple node selection number", ""])
            entry.append(["Ripple node connection method", ""])
            entry.append(["Ripple node ncross number", ""])
        entry.append(["Elapsed time", elapsed_time])
        entry.append(["Train mean elapsed time", train_mean_elapsed_time])
        entry.append(["Mean train epoch loss", str(train_loss_record['train_mean_epoch_loss'])])
        entry.append(["Max train epoch loss", str(train_loss_record['train_max_epoch_loss'])])
        entry.append(["Min train epoch loss", str(train_loss_record['train_min_epoch_loss'])])
        entry.append(["Mean eval mse loss", ""])
        entry.append(["Max eval mse loss", ""])
        entry.append(["Min eval mse loss", ""])
        entry.append(["Mean eval l1 loss", ""])
        entry.append(["Max eval l1 loss", ""])
        entry.append(["Min eval l1 loss", ""])
    elif FLAGS.mode == 'eval':
        entry = []
        entry.append(["Simulation model", str(FLAGS.model)])
        entry.append(["Finished FLAGS.mode", FLAGS.mode])
        entry.append(["Train epochs", ""])
        entry.append(["Epoch trajectories", ""])
        entry.append(["Rollouts", FLAGS.num_rollouts])
        entry.append(["Evaluation set", FLAGS.rollout_split])
        entry.append(["Core model", FLAGS.core_model])
        entry.append(["Message passing aggregator", FLAGS.message_passing_aggregator])
        entry.append(["Message passing steps", str(FLAGS.message_passing_steps)])
        entry.append(["Attention used", str(FLAGS.attention)])
        if FLAGS.ripple_used:
            entry.append(["Ripple used", str(FLAGS.ripple_used)])
            entry.append(["Ripple generation method", str(FLAGS.ripple_generation)])
            entry.append(["Ripple generation number", str(FLAGS.ripple_generation_number)])
            entry.append(["Ripple node selection method", str(FLAGS.ripple_node_selection)])
            entry.append(["Ripple node selection number", str(FLAGS.ripple_node_selection_random_top_n)])
            entry.append(["Ripple node connection method", str(FLAGS.ripple_node_connection)])
            entry.append(["Ripple node ncross number", str(FLAGS.ripple_node_ncross)])
        else:
            entry.append(["Ripple used", str(FLAGS.ripple_used)])
            entry.append(["Ripple generation method", ""])
            entry.append(["Ripple generation number", ""])
            entry.append(["Ripple node selection method", ""])
            entry.append(["Ripple node selection number", ""])
            entry.append(["Ripple node connection method", ""])
            entry.append(["Ripple node ncross number", ""])
        entry.append(["Elapsed time", elapsed_time])
        if FLAGS.mode == 'train' or FLAGS.mode == 'all':
            entry.append(["Train mean elapsed time", train_mean_elapsed_time])
        entry.append(["Mean train epoch loss", ""])
        entry.append(["Max train epoch loss", ""])
        entry.append(["Min train epoch loss", ""])
        entry.append(["Mean eval mse loss", str(eval_loss_record['eval_mean_mse_loss'])])
        entry.append(["Max eval mse loss", str(eval_loss_record['eval_max_mse_loss'])])
        entry.append(["Min eval mse loss", str(eval_loss_record['eval_min_mse_loss'])])
        entry.append(["Mean eval l1 loss", str(eval_loss_record['eval_mean_l1_loss'])])
        entry.append(["Max eval l1 loss", str(eval_loss_record['eval_max_l1_loss'])])
        entry.append(["Min eval l1 loss", str(eval_loss_record['eval_min_l1_loss'])])
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(entry)


if __name__ == '__main__':
    app.run(main)
