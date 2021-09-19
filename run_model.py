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

import cloth_eval
import cloth_model
import dataset
import normalization
import common
import logging

import time
import datetime

import PyG_GCN
from PyG_GCN import gcn

host_system = 'windows'

device = torch.device('cuda')

FLAGS = flags.FLAGS
flags.DEFINE_enum('mode', 'all', ['train', 'eval', 'all', 'test_gcn'],
                  'Train model, or run evaluation.')
flags.DEFINE_enum('model', 'cloth', ['cfd', 'cloth', 'gcn'],
                  'Select model to run.')
flags.DEFINE_enum('network', 'PyG_GCN', ['mgn', 'PyG_GCN'], 'Select network to train.')

flags.DEFINE_enum('rollout_split', 'valid', ['train', 'test', 'valid'],
                  'Dataset split to use for rollouts.')
flags.DEFINE_integer('epochs', 3, 'No. of training epochs')
flags.DEFINE_integer('trajectories', 4, 'No. of training trajectories')
flags.DEFINE_integer('num_rollouts', 5, 'No. of rollout trajectories')

start = time.time()
start_datetime = datetime.datetime.fromtimestamp(start).strftime('%c')
start_datetime_dash = start_datetime.replace(" ", "-").replace(":", "-")

root_dir = pathlib.Path(__file__).parent.resolve()
dataset_name = 'flag_simple'
dataset_dir = os.path.join(root_dir, 'data', dataset_name)
output_dir = os.path.join(root_dir, 'output')
run_dir = os.path.join(output_dir, start_datetime_dash)
Path(run_dir).mkdir(parents=True, exist_ok=True)

if host_system == 'windows':
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
    flags.DEFINE_string('model_last_checkpoint_file',
                        None,
                        # 'C:\\Users\\Mark\\iCloudDrive\\master_arbeit\\implementation\\meshgraphnets\\checkpoint_dir\\checkpoint.pth',
                        'Path to the checkpoint file of a network that should continue training')
    flags.DEFINE_string('optimizer_last_checkpoint_file',
                        None,
                        # 'C:\\Users\\Mark\\iCloudDrive\\master_arbeit\\implementation\\meshgraphnets\\checkpoint_dir\\checkpoint.pth',
                        'Path to the checkpoint file of a network that should continue training')
    flags.DEFINE_string('last_checkpoint_file',
                        None,
                        # 'C:\\Users\\Mark\\iCloudDrive\\master_arbeit\\implementation\\meshgraphnets\\checkpoint_dir\\checkpoint.pth',
                        'Path to the checkpoint file of a network that should continue training')
elif host_system == 'linux':
    flags.DEFINE_string('dataset_dir',
                        '/home/i53/student/ruoheng_ma/mgn_tmp/data/flag_simple/',
                        'Directory to load dataset from.')
    flags.DEFINE_string('checkpoint_dir',
                        '/home/i53/student/ruoheng_ma/mgn_tmp/windows_code_tmp/checkpoint_dir/',
                        'Directory to save checkpoint')
    flags.DEFINE_string('rollout_path',
                        '/home/i53/student/ruoheng_ma/mgn_tmp/windows_code_tmp/rollout/rollout.pkl',
                        'Pickle file to save eval trajectories')
    flags.DEFINE_string('logging_dir',
                        None,
                        'Log file directory')
    flags.DEFINE_string('model_last_checkpoint_file',
                        None,
                        # '/home/i53/student/ruoheng_ma/mgn_tmp/windows_code_tmp/checkpoint_dir/checkpoint.pth',
                        'Path to the checkpoint file of a network that should continue training')
    flags.DEFINE_string('optimizer_last_checkpoint_file',
                        None,
                        # '/home/i53/student/ruoheng_ma/mgn_tmp/windows_code_tmp/checkpoint_dir/checkpoint.pth',
                        'Path to the checkpoint file of a network that should continue training')
    flags.DEFINE_string('last_checkpoint_file',
                        None,
                        # '/home/i53/student/ruoheng_ma/mgn_tmp/windows_code_tmp/checkpoint_dir/checkpoint.pth',
                        'Path to the checkpoint file of a network that should continue training')

PARAMETERS = {
    # 'cfd': dict(noise=0.02, gamma=1.0, field='velocity', history=False,
    #             size=2, batch=2, model=cfd_model, evaluator=cfd_eval),
    'cloth': dict(noise=0.003, gamma=0.1, field='world_pos', history=True,
                  size=3, batch=1, model=cloth_model, evaluator=cloth_eval),
    'gcn': dict(noise=0.003, gamma=0.1, field='world_pos', history=True,
                size=3, batch=1, model=gcn, evaluator=cloth_eval),
}

output_normalizer = normalization.Normalizer(size=3, name='output_normalizer')


def squeeze_data_frame(data_frame):
    for k, v in data_frame.items():
        data_frame[k] = torch.squeeze(v, 0)
    return data_frame


def learner(params, model):
    # handles dataset preprocessing, model definition, training process definition and model training

    # dataset preprocessing
    # batch size can be defined in load_dataset. Default to 1.

    root_logger = logging.getLogger()

    batch_size = 1
    prefetch_factor = 2
    ds_loader = dataset.load_dataset(FLAGS.dataset_dir, 'train', batch_size=batch_size, prefetch_factor=prefetch_factor,
                                     add_targets=True, split_and_preprocess=True)
    # ds_loader = dataset.load_dataset(FLAGS.dataset_dir, 'train', add_targets=add_targets, split_and_preprocess=split_and_preprocess, batch_size=batch_size)

    # model definition
    # dataset will be passed to model, and some specific size of the dataset will be calculated inside model
    # then networks will be initialized

    # training process definition

    optimizer = torch.optim.Adam(model.parameters(recurse=True), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.8, last_epoch=-1)

    # model training
    is_training = True
    batches_in_dataset = 1000 // batch_size

    epoch_training_losses = []

    for epoch in range(FLAGS.epochs):
        # every time when model.train is called, model will train itself with the whole dataset
        root_logger.info("Epoch " + str(epoch + 1) + "/" + str(FLAGS.epochs))
        epoch_training_loss = 0.0
        ds_iterator = iter(ds_loader)
        for trajectory_index in range(FLAGS.trajectories):
            if host_system == 'linux':
                root_logger.info(
                    "    program started on " + start_datetime + ", now in Epoch" + str(epoch + 1) + "/" + str(
                        FLAGS.epochs))
            root_logger.info("    trajectory index " + str(trajectory_index + 1) + "/" + str(batches_in_dataset))
            data = next(ds_iterator)
            trajectory_loss = 0.0
            for data_frame_index, data_frame in enumerate(data):
                data_frame = squeeze_data_frame(data_frame)
                network_output = model(data_frame, is_training)
                loss = loss_fn(data_frame, network_output)
                trajectory_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            epoch_training_loss += trajectory_loss
            root_logger.info("        trajectory_loss")
            root_logger.info("        " + str(trajectory_loss))
            model.save_model(
                FLAGS.checkpoint_dir + "trajectory_model_checkpoint" + "_" + str((trajectory_index + 1) % 2) + ".pth")
            torch.save(optimizer.state_dict(),
                       FLAGS.checkpoint_dir + "trajectory_optimizer_checkpoint" + "_" + str(
                           (trajectory_index + 1) % 2) + ".pth")
            torch.save(scheduler.state_dict(),
                       FLAGS.checkpoint_dir + "trajectory_scheduler_checkpoint" + "_" + str(
                           (trajectory_index + 1) % 2) + ".pth")
        epoch_training_loss = epoch_training_loss
        epoch_training_losses.append(epoch_training_loss)
        root_logger.info("Current mean of epoch training losses")
        root_logger.info(torch.mean(torch.stack(epoch_training_losses)))
        model.save_model(
            os.path.join(FLAGS.checkpoint_dir, "trajectory_model_checkpoint" + "_" + str((trajectory_index + 1) % 2) + ".pth"))
        torch.save(optimizer.state_dict(),
                   os.path.join(FLAGS.checkpoint_dir, "epoch_optimizer_checkpoint" + "_" + str((epoch + 1) % 2) + ".pth"))
        torch.save(scheduler.state_dict(),
                   os.path.join(FLAGS.checkpoint_dir, "epoch_scheduler_checkpoint" + "_" + str((epoch + 1) % 2) + ".pth"))
        scheduler.step()
    model.save_model(os.path.join(FLAGS.checkpoint_dir, "model_checkpoint.pth"))
    torch.save(optimizer.state_dict(), os.path.join(FLAGS.checkpoint_dir, "optimizer_checkpoint.pth"))
    torch.save(scheduler.state_dict(), os.path.join(FLAGS.checkpoint_dir, "scheduler_checkpoint.pth"))
    return model


def loss_fn(inputs, network_output):
    """L2 loss on position."""
    # build target acceleration
    world_pos = inputs['world_pos']
    prev_world_pos = inputs['prev|world_pos']
    target_world_pos = inputs['target|world_pos']

    cur_position = world_pos
    prev_position = prev_world_pos
    target_position = target_world_pos
    target_acceleration = target_position - 2 * cur_position + prev_position
    # target_normalized = output_normalizer(target_acceleration).to(device)

    # build loss
    node_type = inputs['node_type']
    loss_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.NORMAL.value], device=device).int())
    # error = torch.sum((target_normalized - network_output) ** 2, dim=1)
    error = torch.sum((target_acceleration - network_output) ** 2, dim=1)
    loss = torch.mean(error[loss_mask])
    return loss


def evaluator(params, model):
    root_logger = logging.getLogger()
    """Run a model rollout trajectory."""
    ds_loader = dataset.load_dataset(FLAGS.dataset_dir, FLAGS.rollout_split, add_targets=True)
    ds_iterator = iter(ds_loader)
    trajectories = []

    mse_losses = []
    l1_losses = []
    for index in range(FLAGS.num_rollouts):
        root_logger.info("Evaluating trajectory " + str(index + 1))
        trajectory = next(ds_iterator)
        _, prediction_trajectory = params['evaluator'].evaluate(model, trajectory)
        mse_loss_fn = torch.nn.MSELoss()
        l1_loss_fn = torch.nn.L1Loss()
        mse_loss = mse_loss_fn(torch.squeeze(trajectory['world_pos'], dim=0), prediction_trajectory['pred_pos'])
        l1_loss = l1_loss_fn(torch.squeeze(trajectory['world_pos'], dim=0), prediction_trajectory['pred_pos'])
        mse_losses.append(mse_loss)
        l1_losses.append(l1_loss)
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
    with open(FLAGS.rollout_path, 'wb') as fp:
        pickle.dump(trajectories, fp)


def main(argv):
    run_config_record = FLAGS.mode + "_epoch" + str(FLAGS.epochs) + "_trajectory" + str(FLAGS.trajectories) + "_rollout" + str(FLAGS.num_rollouts)
    Path(os.path.join(run_dir, run_config_record)).touch()
    Path(FLAGS.logging_dir).mkdir(parents=True, exist_ok=True)
    Path(FLAGS.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(Path(FLAGS.rollout_path).parent.absolute()).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(FLAGS.logging_dir, "log.log")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    console_output_handler = logging.StreamHandler(sys.stdout)
    console_output_handler.setLevel(logging.INFO)
    file_log_handler = logging.FileHandler(filename=log_path, mode='w', encoding='utf-8')
    file_log_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s - %(message)s')
    console_output_handler.setFormatter(formatter)
    file_log_handler.setFormatter(formatter)
    root_logger.addHandler(console_output_handler)
    root_logger.addHandler(file_log_handler)

    root_logger.info("Program started at time " + str(start_datetime))
    params = PARAMETERS[FLAGS.model]

    if FLAGS.mode == 'train' or FLAGS.mode == 'all':
        root_logger.info("Start training......")
        if FLAGS.model_last_checkpoint_file is not None:
            root_logger.info("Loaded checkpoint file", FLAGS.model_last_checkpoint_file)
            learned_model = torch.load(FLAGS.checkpoint_dir + "model_checkpoint.pth")
            learned_model.to(device)
        model = params['model'].Model(params, output_normalizer)
        model.to(device)
        learner(params, model)
        root_logger.info("Finished training......")
    if FLAGS.mode == 'eval' or FLAGS.mode == 'all':
        root_logger.info("Start evaluating......")
        model = params['model'].Model(params, output_normalizer)
        model.load_model(os.path.join(FLAGS.checkpoint_dir, "model_checkpoint.pth"))
        model.evaluate()
        model.to(device)
        model.eval()
        evaluator(params, model)
        root_logger.info("Finished evaluating......")
    if FLAGS.mode == 'test_gcn':
        print("Start all of test_gcn......")
        model = PyG_GCN.Model()
        model.to(device)
        learner(params, model)
        model = PyG_GCN.Model()
        model.load_model(FLAGS.checkpoint_dir + "model_checkpoint.pth")
        model.to(device)
        model.evaluate()
        evaluator(params, model)
        print("Finished all......")
    end = time.time()
    end_datetime = datetime.datetime.fromtimestamp(end).strftime('%c')
    root_logger.info("Program ended at time " + end_datetime)
    root_logger.info("Finished FLAGS.mode " + FLAGS.mode)
    elapsed_time_in_second = end - start
    elapsed_time = str(datetime.timedelta(seconds=elapsed_time_in_second))
    root_logger.info("Elapsed time " + elapsed_time)


if __name__ == '__main__':
    app.run(main)
