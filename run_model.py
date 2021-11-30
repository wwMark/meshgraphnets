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

import dataset
import common
import logging

import numpy as np
import json
from common import NodeType

import time
import datetime

import csv

import matplotlib

matplotlib.use('AGG')
import matplotlib.pyplot as plt

device = torch.device('cuda')

# train and evaluation configuration
FLAGS = flags.FLAGS
flags.DEFINE_enum('model', 'cloth', ['cfd', 'cloth'],
                  'Select model to run.')
flags.DEFINE_enum('mode', 'all', ['train', 'eval', 'all'],
                  'Train model, or run evaluation, or run both.')
flags.DEFINE_enum('rollout_split', 'valid', ['train', 'test', 'valid'],
                  'Dataset split to use for rollouts.')

flags.DEFINE_integer('epochs', 6, 'No. of training epochs')
flags.DEFINE_integer('trajectories', 2, 'No. of training trajectories')
flags.DEFINE_integer('num_rollouts', 2, 'No. of rollout trajectories')

# core model configuration
flags.DEFINE_enum('core_model', 'encode_process_decode',
                  ['encode_process_decode', 'encode_process_decode_max_pooling', 'encode_process_decode_lstm',
                   'encode_process_decode_graph_structure_watcher', 'encode_process_decode_ripple'],
                  'Core model to be used')
flags.DEFINE_enum('message_passing_aggregator', 'min', ['sum', 'max', 'min', 'mean'], 'No. of training epochs')
flags.DEFINE_integer('message_passing_steps', 1, 'No. of training epochs')
flags.DEFINE_boolean('attention', True, 'whether attention is used or not')

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
flags.DEFINE_enum('ripple_generation', 'gradient', ['equal_size', 'gradient', 'exponential_size'],
                  'defines how ripples are generated')
flags.DEFINE_integer('ripple_generation_number', 1,
                     'defines how many ripples should be generated in equal size and gradient ripple generation; or the base in exponential size generation')
flags.DEFINE_enum('ripple_node_selection', 'random', ['random', 'all', 'top'],
                  'defines how the nodes are selected from each ripple')
flags.DEFINE_integer('ripple_node_selection_random_top_n', 1,
                     'defines how many nodes are selected from each ripple if node selection is random or top')
flags.DEFINE_enum('ripple_node_connection', 'fully_ncross_connected',
                  ['most_influential', 'fully_connected', 'fully_ncross_connected'],
                  'defines how the selected nodes of each ripple connect with each other to propagate message faster')
flags.DEFINE_integer('ripple_node_ncross', 1,
                     'defines how many fully cross connections should be generated between ripples')

start = time.time()
start_datetime = datetime.datetime.fromtimestamp(start).strftime('%c')
start_datetime_dash = start_datetime.replace(" ", "-").replace(":", "-")

root_dir = pathlib.Path(__file__).parent.resolve()
dataset_name = 'flag_simple'
# dataset_name = 'cylinder_flow'
dataset_dir = os.path.join(root_dir, 'data', dataset_name)
output_dir = os.path.join(root_dir, 'output', dataset_name)
run_dir = os.path.join(output_dir, start_datetime_dash)

# directory setting
flags.DEFINE_string('dataset_dir',
                    dataset_dir,
                    'Directory to load dataset from.')
flags.DEFINE_string('checkpoint_dir',
                    os.path.join(run_dir, 'checkpoint_dir'),
                    'Directory to save checkpoint')
flags.DEFINE_string('rollout_path',
                    os.path.join(run_dir, 'rollout', 'rollout.pkl'),
                    'Pickle file to save eval trajectories')
flags.DEFINE_string('logging_dir',
                    os.path.join(run_dir, 'logs'),
                    'Log file directory')
flags.DEFINE_string('model_last_checkpoint_dir',
                    None,
                    # os.path.join('C:\\Users\\Mark\\iCloudDrive\\master_arbeit\\implementation\\meshgraphnets\\output\\flag_simple\\Tue-Nov-30-17-03-11-2021', 'checkpoint_dir'),
                    'Path to the checkpoint file of a network that should continue training')
flags.DEFINE_string('optimizer_last_checkpoint_file',
                    None,
                    # 'C:\\Users\\Mark\\iCloudDrive\\master_arbeit\\implementation\\meshgraphnets\\output\\Sun-Sep-26-23-06-14-2021\\epoch_optimizer_checkpoint_1.pth',
                    'Path to the checkpoint file of a network that should continue training')
flags.DEFINE_string('last_checkpoint_file',
                    None,
                    # 'C:\\Users\\Mark\\iCloudDrive\\master_arbeit\\implementation\\meshgraphnets\\output\\Sun-Sep-26-23-06-14-2021\\epoch_scheduler_checkpoint_1.pth',
                    'Path to the checkpoint file of a network that should continue training')

PARAMETERS = {
    'cfd': dict(noise=0.02, gamma=1.0, field='velocity', history=False,
                size=2, batch=2, model=cfd_model, evaluator=cfd_eval, loss_type='cfd',
                stochastic_message_passing_used='False'),
    'cloth': dict(noise=0.003, gamma=0.1, field='world_pos', history=True,
                  size=3, batch=1, model=cloth_model, evaluator=cloth_eval, loss_type='cloth',
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
hpc_default_max_time = 172800 - 3600 * 2
# hpc_default_max_time = 3 * 60
hpc_max_time = hpc_start_time + hpc_default_max_time


def squeeze_data_frame(data_frame):
    for k, v in data_frame.items():
        data_frame[k] = torch.squeeze(v, 0)
    return data_frame


def add_targets(params):
    """Adds target and optionally history fields to dataframe."""
    fields = params['field']
    add_history = params['history']

    def fn(trajectory):
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
        frame['target|' + noise_field] += (1.0 - noise_gamma) * noise
        return frame

    def element_operation(trajectory):
        '''
        if model_type == 'cloth':
            world_pos = trajectory['world_pos']
            mesh_pos = trajectory['mesh_pos']
            node_type = trajectory['node_type']
            cells = trajectory['cells']
            target_world_pos = trajectory['target|world_pos']
            prev_world_pos = trajectory['prev|world_pos']
            trajectory_steps = []
            for i in range(399):
                wp = world_pos[i]
                mp = mesh_pos[i]
                twp = target_world_pos[i]
                nt = node_type[i]
                c = cells[i]
                pwp = prev_world_pos[i]
                trajectory_step = {'world_pos': wp, 'mesh_pos': mp, 'node_type': nt, 'cells': c,
                                   'target|world_pos': twp, 'prev|world_pos': pwp}
                noisy_trajectory_step = add_noise(trajectory_step)
                trajectory_steps.append(noisy_trajectory_step)
            return trajectory_steps
        '''
        trajectory_steps = []
        for i in range(steps):
            trajectory_step = {}
            for key, value in trajectory.items():
                trajectory_step[key] = value[i]
            noisy_trajectory_step = add_noise(trajectory_step)
            trajectory_steps.append(noisy_trajectory_step)
        return trajectory_steps

    return element_operation


def process_trajectory(trajectory_data, params, model_type, add_targets_bool=False, split_and_preprocess_bool=False):
    global loaded_meta
    global shapes
    global dtypes
    global types
    global steps

    if not loaded_meta:
        try:
            with open(os.path.join(FLAGS.dataset_dir, 'meta.json'), 'r') as fp:
                meta = json.loads(fp.read())
            shapes = {}
            dtypes = {}
            types = {}
            steps = meta['trajectory_length'] - 2
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


def learner(model, params):
    # handles dataset preprocessing, model definition, training process definition and model training

    root_logger = logging.getLogger()

    loss_type = params['loss_type']
    model_type = FLAGS.model

    # batch size can be defined in load_dataset. Default to 1.
    batch_size = 1
    prefetch_factor = 2

    # ds_loader = dataset.load_dataset(FLAGS.dataset_dir, 'train', add_targets=add_targets, split_and_preprocess=split_and_preprocess, batch_size=batch_size)

    # model definition
    # dataset will be passed to model, and some specific size of the dataset will be calculated inside model
    # then networks will be initialized

    # training process definition
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1 + 1e-6, last_epoch=-1)
    trained_epoch = 0
    if FLAGS.model_last_checkpoint_dir is not None:
        optimizer.load_state_dict(torch.load(os.path.join(FLAGS.model_last_checkpoint_dir, "optimizer_checkpoint.pth")))
        scheduler.load_state_dict(torch.load(os.path.join(FLAGS.model_last_checkpoint_dir, "scheduler_checkpoint.pth")))
        epoch_checkpoint = torch.load(os.path.join(FLAGS.model_last_checkpoint_dir, "epoch_checkpoint.pth"))
        trained_epoch = epoch_checkpoint['epoch'] + 1

    # model training
    is_training = True

    epoch_training_losses = []

    count = 0
    pass_count = 500
    if FLAGS.model_last_checkpoint_dir is not None:
        pass_count = 0
    all_trajectory_train_losses = []
    epoch_run_times = []
    for epoch in range(FLAGS.epochs)[trained_epoch:]:

        # check whether the rest time is sufficient for running a whole epoch; stop running if not
        hpc_current_time = time.time()
        if len(epoch_run_times) != 0:
            epoch_mean_time = sum(epoch_run_times) // len(epoch_run_times)
            if hpc_current_time + epoch_mean_time >= hpc_max_time:
                break

        ds_loader = dataset.load_dataset(FLAGS.dataset_dir, 'train', batch_size=batch_size,
                                         prefetch_factor=prefetch_factor,
                                         add_targets=True, split_and_preprocess=True)
        # every time when model.train is called, model will train itself with the whole dataset
        root_logger.info("Epoch " + str(epoch + 1) + "/" + str(FLAGS.epochs))
        epoch_training_loss = 0.0
        ds_iterator = iter(ds_loader)
        for trajectory_index in range(FLAGS.trajectories):
            root_logger.info("    trajectory index " + str(trajectory_index + 1) + "/" + str(FLAGS.trajectories))
            trajectory = next(ds_iterator)
            trajectory = process_trajectory(trajectory, params, model_type, True, True)
            trajectory_loss = 0.0
            for data_frame_index, data_frame in enumerate(trajectory):
                count += 1
                data_frame = squeeze_data_frame(data_frame)
                network_output = model(data_frame, is_training)
                loss = loss_fn(loss_type, data_frame, network_output, model, params)
                if count % 1000 == 0:
                    root_logger.info("    1000 step loss " + str(loss))
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
                os.path.join(FLAGS.checkpoint_dir,
                             "trajectory_model_checkpoint" + "_" + str((trajectory_index + 1) % 2)))
            torch.save(optimizer.state_dict(),
                       os.path.join(FLAGS.checkpoint_dir,
                                    "trajectory_optimizer_checkpoint" + "_" + str((epoch + 1) % 2) + ".pth"))
            torch.save(scheduler.state_dict(),
                       os.path.join(FLAGS.checkpoint_dir,
                                    "trajectory_scheduler_checkpoint" + "_" + str((epoch + 1) % 2) + ".pth"))
        epoch_training_losses.append(epoch_training_loss)
        root_logger.info("Current mean of epoch training losses")
        root_logger.info(torch.mean(torch.stack(epoch_training_losses)))
        model.save_model(
            os.path.join(FLAGS.checkpoint_dir, "epoch_model_checkpoint" + "_" + str((trajectory_index + 1) % 2)))
        torch.save(optimizer.state_dict(),
                   os.path.join(FLAGS.checkpoint_dir,
                                "epoch_optimizer_checkpoint" + "_" + str((epoch + 1) % 2) + ".pth"))
        torch.save(scheduler.state_dict(),
                   os.path.join(FLAGS.checkpoint_dir,
                                "epoch_scheduler_checkpoint" + "_" + str((epoch + 1) % 2) + ".pth"))
        if epoch == 20:
            scheduler.step()
        torch.save({'epoch': epoch}, os.path.join(FLAGS.checkpoint_dir, "epoch_checkpoint.pth"))
        epoch_run_times.append(time.time() - hpc_current_time)
    model.save_model(os.path.join(FLAGS.checkpoint_dir, "model_checkpoint"))
    torch.save(optimizer.state_dict(), os.path.join(FLAGS.checkpoint_dir, "optimizer_checkpoint.pth"))
    torch.save(scheduler.state_dict(), os.path.join(FLAGS.checkpoint_dir, "scheduler_checkpoint.pth"))
    loss_record = {}
    loss_record['train_total_loss'] = torch.sum(torch.stack(epoch_training_losses))
    loss_record['train_mean_epoch_loss'] = torch.mean(torch.stack(epoch_training_losses)).item()
    loss_record['train_max_epoch_loss'] = torch.max(torch.stack(epoch_training_losses)).item()
    loss_record['train_min_epoch_loss'] = torch.min(torch.stack(epoch_training_losses)).item()
    loss_record['train_epoch_losses'] = epoch_training_losses
    loss_record['all_trajectory_train_losses'] = all_trajectory_train_losses
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


def evaluator(params, model):
    root_logger = logging.getLogger()
    model_type = FLAGS.model
    """Run a model rollout trajectory."""
    ds_loader = dataset.load_dataset(FLAGS.dataset_dir, FLAGS.rollout_split, add_targets=True)
    ds_iterator = iter(ds_loader)
    trajectories = []

    mse_losses = []
    l1_losses = []
    for index in range(FLAGS.num_rollouts):
        root_logger.info("Evaluating trajectory " + str(index + 1))
        trajectory = next(ds_iterator)
        trajectory = process_trajectory(trajectory, params, model_type, True)
        _, prediction_trajectory = params['evaluator'].evaluate(model, trajectory)
        mse_loss_fn = torch.nn.MSELoss()
        l1_loss_fn = torch.nn.L1Loss()
        if model_type == 'cloth':
            mse_loss = mse_loss_fn(torch.squeeze(trajectory['world_pos'], dim=0), prediction_trajectory['pred_pos'])
            l1_loss = l1_loss_fn(torch.squeeze(trajectory['world_pos'], dim=0), prediction_trajectory['pred_pos'])
        elif model_type == 'cfd':
            mse_loss = mse_loss_fn(torch.squeeze(trajectory['velocity'], dim=0), prediction_trajectory['pred_velocity'])
            l1_loss = l1_loss_fn(torch.squeeze(trajectory['velocity'], dim=0), prediction_trajectory['pred_velocity'])
        mse_losses.append(mse_loss.cpu())
        l1_losses.append(l1_loss.cpu())
        root_logger.info("    trajectory evaluation mse loss")
        root_logger.info("    " + str(mse_loss))
        root_logger.info("    trajectory evaluation l1 loss")
        root_logger.info("    " + str(l1_loss))
        trajectories.append(prediction_trajectory)
        # scalars.append(scalar_data)
    root_logger.info("mean mse loss of " + str(FLAGS.num_rollouts) + " rollout trajectories")
    root_logger.info(torch.mean(torch.stack(mse_losses)))
    root_logger.info("mean l1 loss " + str(FLAGS.num_rollouts) + " rollout trajectories")
    root_logger.info(torch.mean(torch.stack(l1_losses)))
    with open(os.path.join(run_dir, "rollout", "rollout.pkl"), 'wb') as fp:
        pickle.dump(trajectories, fp)
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


def plot_data(data):
    return None


def main(argv):
    global run_dir
    # set log configuration
    is_all = False
    if FLAGS.mode == "all":
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        # save program configuration in file title
        run_config_record = FLAGS.mode + "_epoch" + str(FLAGS.epochs) + "_trajectory" + str(
            FLAGS.trajectories) + "_rollout" + str(FLAGS.num_rollouts)
        Path(os.path.join(run_dir, run_config_record)).touch()
        log_path = os.path.join(FLAGS.logging_dir, "all_log.log")
        is_all = True

    if FLAGS.mode == "train" or is_all:
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        # save program configuration in file title
        if not is_all:
            run_config_record = FLAGS.mode + "_epoch" + str(FLAGS.epochs) + "_trajectory" + str(FLAGS.trajectories)
            log_path = os.path.join(FLAGS.logging_dir, "train_log.log")
        Path(os.path.join(run_dir, run_config_record)).touch()

        # make direcotory for log, checkpoint
        Path(FLAGS.logging_dir).mkdir(parents=True, exist_ok=True)
        Path(FLAGS.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    if FLAGS.mode == "eval" or is_all:
        # find latest directory in output directory
        all_subdirs = os.listdir(output_dir)
        all_subdirs = map(lambda d: os.path.join(output_dir, d), all_subdirs)
        all_subdirs = [d for d in all_subdirs if os.path.isdir(d)]
        latest_subdir = max(all_subdirs, key=os.path.getmtime)
        run_dir = latest_subdir
        # save program configuration in file title
        if not is_all:
            run_config_record = FLAGS.mode + "_rollout" + str(FLAGS.num_rollouts)
            log_path = os.path.join(run_dir, "logs", "eval_log.log")
        Path(os.path.join(run_dir, run_config_record)).touch()

        # make direcotory for rollout
        Path(os.path.join(run_dir, "rollout")).mkdir(parents=True, exist_ok=True)

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

    root_logger.info("Program started at time " + str(start_datetime))
    params = PARAMETERS[FLAGS.model]

    if FLAGS.mode == 'train' or FLAGS.mode == 'all':
        root_logger.info("Start training......")
        model = params['model'].Model(params, FLAGS.core_model, FLAGS.message_passing_aggregator,
                                      FLAGS.message_passing_steps, FLAGS.attention, FLAGS.ripple_used,
                                      FLAGS.ripple_generation, FLAGS.ripple_generation_number,
                                      FLAGS.ripple_node_selection, FLAGS.ripple_node_selection_random_top_n,
                                      FLAGS.ripple_node_connection,
                                      FLAGS.ripple_node_ncross)
        if FLAGS.model_last_checkpoint_dir is not None:
            model.load_model(os.path.join(FLAGS.model_last_checkpoint_dir, "model_checkpoint"))
            root_logger.info(
                "Loaded checkpoint file in " + str(FLAGS.model_last_checkpoint_dir) + "and starting retraining...")
        model.to(device)

        # run summary
        root_logger.info("")
        root_logger.info("=======================Run Summary=======================")
        root_logger.info("Simulation task is " + str(FLAGS.model) + " simulation")
        root_logger.info("Finished FLAGS.mode " + FLAGS.mode)
        if FLAGS.mode == 'eval' or FLAGS.mode == 'all':
            root_logger.info("Evaluation set is " + FLAGS.rollout_split)
        elif FLAGS.mode == 'train':
            root_logger.info("No Evaluation")
        root_logger.info(
            "Train and/or evaluation configuration are " + str(FLAGS.epochs) + " epochs, " + str(
                FLAGS.trajectories) + " trajectories each epoch, number of rollouts is " + str(FLAGS.num_rollouts))
        root_logger.info("Core model is " + FLAGS.core_model)
        root_logger.info("Message passing aggregator is " + FLAGS.message_passing_aggregator)
        root_logger.info("Message passing steps are " + str(FLAGS.message_passing_steps))
        root_logger.info("Attention used is " + str(FLAGS.attention))
        root_logger.info("Ripple used is " + str(FLAGS.ripple_used))
        if FLAGS.ripple_used:
            root_logger.info("  Ripple generation method is " + str(FLAGS.ripple_generation))
            root_logger.info("  Ripple generation number is " + str(FLAGS.ripple_generation_number))
            root_logger.info("  Ripple node selection method is " + str(FLAGS.ripple_node_selection))
            root_logger.info("  Ripple node selection number is " + str(FLAGS.ripple_node_selection_random_top_n))
            root_logger.info("  Ripple node connection method is " + str(FLAGS.ripple_node_connection))
            root_logger.info("  Ripple node ncross number is " + str(FLAGS.ripple_node_ncross))
        root_logger.info("Run output directory is " + run_dir)
        root_logger.info("=======================Run Summary=======================")
        root_logger.info("")

        train_start = time.time()
        train_loss_record = learner(model, params)
        # load train loss
        if FLAGS.model_last_checkpoint_dir is not None:
            with open(os.path.join(Path(FLAGS.model_last_checkpoint_dir).parent, 'logs', 'train_loss.pkl'), 'rb') as pickle_file:
                saved_train_loss_record = pickle.load(pickle_file)
            train_loss_record['train_epoch_losses'] = train_loss_record['train_epoch_losses'] + \
                                                      saved_train_loss_record['train_epoch_losses']
            train_loss_record['train_total_loss'] = torch.sum(torch.stack(train_loss_record['train_epoch_losses']))
            train_loss_record['train_mean_epoch_loss'] = torch.mean(torch.stack(train_loss_record['train_epoch_losses'])).item()
            train_loss_record['train_max_epoch_loss'] = torch.max(torch.stack(train_loss_record['train_epoch_losses'])).item()
            train_loss_record['train_min_epoch_loss'] = torch.min(torch.stack(train_loss_record['train_epoch_losses'])).item()
            train_loss_record['all_trajectory_train_losses'] = train_loss_record['all_trajectory_train_losses'] + \
                                                               saved_train_loss_record['all_trajectory_train_losses']
        train_end = time.time()
        train_elapsed_time_in_second = train_end - train_start
        if FLAGS.model_last_checkpoint_dir is not None:
            with open(os.path.join(Path(FLAGS.model_last_checkpoint_dir).parent, 'logs', 'train_elapsed_time_in_second.pkl'), 'rb') as pickle_file:
                saved_train_elapsed_time_in_second = pickle.load(pickle_file)
            train_elapsed_time_in_second += saved_train_elapsed_time_in_second
        train_elapsed_time_in_second_pkl_file = os.path.join(FLAGS.logging_dir, 'train_elapsed_time_in_second.pkl')
        Path(train_elapsed_time_in_second_pkl_file).touch()
        with open(train_elapsed_time_in_second_pkl_file, 'wb') as f:
            pickle.dump(train_elapsed_time_in_second, f)
        train_mean_elapsed_time = str(
            datetime.timedelta(seconds=train_elapsed_time_in_second // (FLAGS.epochs * FLAGS.trajectories)))
        train_elapsed_time = str(datetime.timedelta(seconds=train_elapsed_time_in_second))

        root_logger.info("Finished training......")
    if FLAGS.mode == 'eval' or FLAGS.mode == 'all':
        root_logger.info("Start evaluating......")
        model = params['model'].Model(params, FLAGS.core_model, FLAGS.message_passing_aggregator,
                                      FLAGS.message_passing_steps, FLAGS.attention, FLAGS.ripple_used,
                                      FLAGS.ripple_generation, FLAGS.ripple_generation_number,
                                      FLAGS.ripple_node_selection, FLAGS.ripple_node_selection_random_top_n,
                                      FLAGS.ripple_node_connection,
                                      FLAGS.ripple_node_ncross)
        if FLAGS.model_last_checkpoint_dir is not None:
            model.load_model(os.path.join(FLAGS.model_last_checkpoint_dir, "model_checkpoint"))
            root_logger.info(
                "Loaded checkpoint file in " + str(FLAGS.model_last_checkpoint_dir) + "and starting retraining...")
        else:
            model.load_model(os.path.join(run_dir, "checkpoint_dir", "model_checkpoint"))
            root_logger.info("Loaded model from " + str(run_dir))
        model.evaluate()
        model.to(device)
        root_logger.info("Core model is " + FLAGS.core_model)
        root_logger.info("Message passing aggregator is " + FLAGS.message_passing_aggregator)
        root_logger.info("Message passing steps are " + str(FLAGS.message_passing_steps))
        root_logger.info("Attention used is " + str(FLAGS.attention))
        root_logger.info("Ripple used is " + str(FLAGS.ripple_used))
        if FLAGS.ripple_used:
            root_logger.info("  Ripple generation method is " + str(FLAGS.ripple_generation))
            root_logger.info("  Ripple generation number is " + str(FLAGS.ripple_generation_number))
            root_logger.info("  Ripple node selection method is " + str(FLAGS.ripple_node_selection))
            root_logger.info("  Ripple node selection number is " + str(FLAGS.ripple_node_selection_random_top_n))
            root_logger.info("  Ripple node connection method is " + str(FLAGS.ripple_node_connection))
            root_logger.info("  Ripple node ncross number is " + str(FLAGS.ripple_node_ncross))
        eval_loss_record = evaluator(params, model)
        root_logger.info("Finished evaluating......")
    end = time.time()
    end_datetime = datetime.datetime.fromtimestamp(end).strftime('%c')
    root_logger.info("Program ended at time " + end_datetime)
    elapsed_time_in_second = end - start
    if FLAGS.model_last_checkpoint_dir is not None:
        with open(
                os.path.join(Path(FLAGS.model_last_checkpoint_dir).parent, 'logs', 'elapsed_time_in_second.pkl'),
                'rb') as pickle_file:
            saved_elapsed_time_in_second = pickle.load(pickle_file)
        elapsed_time_in_second += saved_elapsed_time_in_second
    elapsed_time = str(datetime.timedelta(seconds=elapsed_time_in_second))
    elapsed_time_in_second_pkl_file = os.path.join(FLAGS.logging_dir, 'elapsed_time_in_second.pkl')
    Path(elapsed_time_in_second_pkl_file).touch()
    with open(elapsed_time_in_second_pkl_file, 'wb') as f:
        pickle.dump(elapsed_time_in_second, f)

    # run summary
    root_logger.info("")
    root_logger.info("=======================Run Summary=======================")
    root_logger.info("Simulation model is " + str(FLAGS.model))
    root_logger.info("Finished FLAGS.mode " + FLAGS.mode)
    if FLAGS.mode == 'eval' or FLAGS.mode == 'all':
        root_logger.info("Evaluation set is " + FLAGS.rollout_split)
    elif FLAGS.mode == 'train':
        root_logger.info("No Evaluation")
    root_logger.info("Core model is " + FLAGS.core_model)
    root_logger.info("Message passing aggregator is " + FLAGS.message_passing_aggregator)
    root_logger.info("Message passing steps are " + str(FLAGS.message_passing_steps))
    root_logger.info("Attention used is " + str(FLAGS.attention))
    root_logger.info("Ripple used is " + str(FLAGS.ripple_used))
    if FLAGS.ripple_used:
        root_logger.info("  Ripple generation method is " + str(FLAGS.ripple_generation))
        root_logger.info("  Ripple generation number is " + str(FLAGS.ripple_generation_number))
        root_logger.info("  Ripple node selection method is " + str(FLAGS.ripple_node_selection))
        root_logger.info("  Ripple node selection number is " + str(FLAGS.ripple_node_selection_random_top_n))
        root_logger.info("  Ripple node connection method is " + str(FLAGS.ripple_node_connection))
        root_logger.info("  Ripple node ncross number is " + str(FLAGS.ripple_node_ncross))
    root_logger.info("Elapsed time " + elapsed_time)
    root_logger.info("Run output directory is " + run_dir)
    root_logger.info("=======================Run Summary=======================")
    root_logger.info("")
    root_logger.info("--------------------train loss record--------------------")
    if FLAGS.mode == "train" or FLAGS.mode == "all":
        train_loss_pkl_file = os.path.join(FLAGS.logging_dir, 'train_loss.pkl')
        Path(train_loss_pkl_file).touch()
        with open(train_loss_pkl_file, 'wb') as f:
            pickle.dump(train_loss_record, f)
        for item in train_loss_record.items():
            root_logger.info(item)
    root_logger.info("---------------------------------------------------------")
    root_logger.info("")
    root_logger.info("--------------------eval loss record---------------------")
    if FLAGS.mode == "eval" or FLAGS.mode == "all":
        eval_loss_pkl_file = os.path.join(FLAGS.logging_dir, 'eval_loss.pkl')
        Path(eval_loss_pkl_file).touch()
        with open(eval_loss_pkl_file, 'wb') as f:
            pickle.dump(eval_loss_record, f)
        for item in eval_loss_record.items():
            root_logger.info(item)
    root_logger.info("---------------------------------------------------------")

    # save result in figure
    fig = plt.figure(figsize=(38.4, 21.6), constrained_layout=True)
    gs = fig.add_gridspec(4, 3)
    fig.suptitle('Train and Evaluation Losses', fontsize=32)
    description = []
    description.append("Simulation model is " + str(FLAGS.model) + "\n")
    description.append("Finished FLAGS.mode " + FLAGS.mode + "\n")
    if FLAGS.mode == 'eval' or FLAGS.mode == 'all':
        description.append("Evaluation set is " + FLAGS.rollout_split + "\n")
    elif FLAGS.mode == 'train':
        description.append("No Evaluation" + "\n")
    description.append("Core model is " + FLAGS.core_model + "\n")
    description.append("Message passing aggregator is " + FLAGS.message_passing_aggregator + "\n")
    description.append("Message passing steps are " + str(FLAGS.message_passing_steps) + "\n")
    description.append("Attention used is " + str(FLAGS.attention) + "\n")
    description.append("Ripple used is " + str(FLAGS.ripple_used) + "\n")
    if FLAGS.ripple_used:
        description.append("    Ripple generation method is " + str(FLAGS.ripple_generation) + "\n")
        description.append("    Ripple generation number is " + str(FLAGS.ripple_generation_number) + "\n")
        description.append("    Ripple node selection method is " + str(FLAGS.ripple_node_selection) + "\n")
        description.append(
            "    Ripple node selection number is " + str(FLAGS.ripple_node_selection_random_top_n) + "\n")
        description.append("    Ripple node connection method is " + str(FLAGS.ripple_node_connection) + "\n")
        description.append("    Ripple node ncross number is " + str(FLAGS.ripple_node_ncross) + "\n")
    description.append("Elapsed time " + elapsed_time + "\n")
    description.append("Train mean elapsed time " + train_mean_elapsed_time + "\n")
    description_txt = ""
    for item in description:
        description_txt += item
    plt.figtext(0.5, 0.01, description_txt, wrap=True, horizontalalignment='left', fontsize=22)
    if FLAGS.mode == 'train' or FLAGS.mode == 'all':
        train_loss_ax = fig.add_subplot(gs[0, 0])
        all_trajectory_train_losses_ax = fig.add_subplot(gs[1:3, 0:])

        train_loss_ax.set_title('Train Loss', fontsize=28)
        train_loss_ax.set_xlabel('epoch', fontsize=22)
        train_loss_ax.set_ylabel('loss', fontsize=22)

        all_trajectory_train_losses_ax.set_title('Train trajectory Loss', fontsize=28)
        all_trajectory_train_losses_ax.set_xlabel('trajectory no.', fontsize=22)
        all_trajectory_train_losses_ax.set_ylabel('loss', fontsize=22)

        # train_loss_ax.xaxis.set_major_locator(ticker.AutoLocator())
        # train_loss_ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

        # all_trajectory_train_losses_ax.xaxis.set_major_locator(ticker.AutoLocator())
        # all_trajectory_train_losses_ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

        train_loss_ax.plot(range(1, len(train_loss_record['train_epoch_losses']) + 1),
                           train_loss_record['train_epoch_losses'])
        all_trajectory_train_losses_ax.plot(range(1, len(train_loss_record['all_trajectory_train_losses']) + 1),
                                            train_loss_record['all_trajectory_train_losses'])
    if FLAGS.mode == 'eval' or FLAGS.mode == 'all':
        eval_mse_loss_ax = fig.add_subplot(gs[0, 1])
        eval_l1_loss_ax = fig.add_subplot(gs[0, 2])

        eval_mse_loss_ax.set_title('Eval MSE Loss', fontsize=28)
        eval_mse_loss_ax.set_xlabel('rollout no.', fontsize=22)
        eval_mse_loss_ax.set_ylabel('loss', fontsize=22)

        eval_l1_loss_ax.set_title('Eval L1 Loss', fontsize=28)
        eval_l1_loss_ax.set_xlabel('rollout no.', fontsize=22)
        eval_l1_loss_ax.set_ylabel('loss', fontsize=22)

        '''
        eval_mse_loss_ax.xaxis.set_major_locator(ticker.AutoLocator())
        eval_mse_loss_ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        eval_l1_loss_ax.xaxis.set_major_locator(ticker.AutoLocator())
        eval_l1_loss_ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        '''

        eval_mse_loss_ax.plot(range(1, len(eval_loss_record['eval_mse_losses']) + 1),
                              eval_loss_record['eval_mse_losses'], 'o')
        eval_l1_loss_ax.plot(range(1, len(eval_loss_record['eval_l1_losses']) + 1), eval_loss_record['eval_l1_losses'],
                             'o')
    fig.savefig(os.path.join(run_dir, "logs", "Train_and_Eval_Loss.png"))

    # save max, min and mean value of train and eval losses as csv
    csv_path = os.path.join(FLAGS.logging_dir, 'result.csv')
    try:
        Path(csv_path).touch()
    except FileExistsError:
        csv_path = os.path.join(FLAGS.logging_dir, 'FileExistsErrorHandled_result.csv')
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
