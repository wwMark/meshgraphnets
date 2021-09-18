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

import pickle
from absl import app
from absl import flags

import torch

import cloth_eval
import cloth_model
import dataset
import normalization
import encode_process_decode
import common
import PyG_GCN
import logging
from statistics import mean

import time
import datetime

host_system = 'windows'
start_datetime_global = None

device = torch.device('cuda')

FLAGS = flags.FLAGS
flags.DEFINE_enum('mode', 'all', ['train', 'eval', 'all', 'test_gcn'],
                  'Train model, or run evaluation.')
flags.DEFINE_enum('model', 'cloth', ['cfd', 'cloth'],
                  'Select model to run.')
flags.DEFINE_enum('network', 'PyG_GCN', ['mgn', 'PyG_GCN'], 'Select network to train.')

flags.DEFINE_enum('rollout_split', 'valid', ['train', 'test', 'valid'],
                  'Dataset split to use for rollouts.')
flags.DEFINE_integer('epochs', 2, 'No. of training epochs')
flags.DEFINE_integer('trajectories', 1, 'No. of training trajectories')
flags.DEFINE_integer('num_rollouts', 2, 'No. of rollout trajectories')

if host_system == 'windows':
    flags.DEFINE_string('dataset_dir',
                        'C:\\Users\\Mark\\iCloudDrive\\master_arbeit\\deepmind-research\\meshgraphnets\\data\\flag_simple\\',
                        'Directory to load dataset from.')
    flags.DEFINE_string('checkpoint_dir',
                        'C:\\Users\\Mark\\iCloudDrive\\master_arbeit\\implementation\\meshgraphnets\\checkpoint_dir\\',
                        'Directory to save checkpoint')
    flags.DEFINE_string('rollout_path',
                        'C:\\Users\\Mark\\iCloudDrive\\master_arbeit\\implementation\\meshgraphnets\\rollout\\rollout.pkl',
                        'Pickle file to save eval trajectories')
    flags.DEFINE_string('logging_dir',
                        'C:\\Users\\Mark\\iCloudDrive\\master_arbeit\\implementation\\meshgraphnets\\output\\logs\\',
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
                  size=3, batch=1, model=cloth_model, evaluator=cloth_eval)
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
                root_logger.info("    program started on " + start_datetime_global + ", now in Epoch" + str(epoch + 1) + "/" + str(FLAGS.epochs))
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
            torch.save(model.learned_model,
                       FLAGS.checkpoint_dir + "trajectory_model_checkpoint" + "_" + str(
                           (trajectory_index + 1) % 2) + ".pth")
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
        torch.save(model.learned_model,
                   FLAGS.checkpoint_dir + "epoch_model_checkpoint" + "_" + str((epoch + 1) % 2) + ".pth")
        torch.save(optimizer.state_dict(),
                   FLAGS.checkpoint_dir + "epoch_optimizer_checkpoint" + "_" + str((epoch + 1) % 2) + ".pth")
        torch.save(scheduler.state_dict(),
                   FLAGS.checkpoint_dir + "epoch_scheduler_checkpoint" + "_" + str((epoch + 1) % 2) + ".pth")
        scheduler.step()
    torch.save(model.learned_model, FLAGS.checkpoint_dir + "model_checkpoint.pth")
    torch.save(optimizer.state_dict(), FLAGS.checkpoint_dir + "optimizer_checkpoint.pth")
    torch.save(scheduler.state_dict(), FLAGS.checkpoint_dir + "scheduler_checkpoint.pth")
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
    start = time.time()
    start_datetime = datetime.datetime.fromtimestamp(start).strftime('%c')
    start_datetime_dash = start_datetime.replace(" ", "-").replace(":", "-")

    log_path = FLAGS.logging_dir + start_datetime_dash + "_" + FLAGS.mode + "_epoch" + str(FLAGS.epochs) + "_trajectory" + str(FLAGS.trajectories) + "_rollout" + str(FLAGS.num_rollouts) + ".log"
    # logging.basicConfig(encoding='utf-8', level=logging.INFO)
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

    # print("Program started at time", start_datetime)
    root_logger.info("Program started at time " + str(start_datetime))
    global start_datetime_global
    start_datetime_global = start_datetime
    params = PARAMETERS[FLAGS.model]

    if FLAGS.mode == 'train':
        root_logger.info("Start training......")
        if FLAGS.model_last_checkpoint_file is not None:
            root_logger.info("Loaded checkpoint file", FLAGS.model_last_checkpoint_file)
            learned_model = torch.load(FLAGS.checkpoint_dir + "model_checkpoint.pth")
            learned_model.to(device)
        model = params['model'].Model(params, output_normalizer)
        model.to(device)
        learner(params, model)
        root_logger.info("Finished training......")
    elif FLAGS.mode == 'eval':
        root_logger.info("Start evaluating......")
        learned_model = torch.load(FLAGS.checkpoint_dir + "model_checkpoint.pth")
        learned_model.to(device)
        learned_model.eval()
        model = params['model'].Model(params, learned_model, output_normalizer)
        model.to(device)
        model.eval()
        evaluator(params, model)
        '''
        prefix = "learned_model."
        n_clip = len(prefix)
        adapted_dict = {k[n_clip]: v for k, v in torch.load(FLAGS.checkpoint_dir).items() if k.startswith(prefix)}
        model.load_state_dict(adapted_dict)
        model.eval()
        evaluator(params, model)
        '''
        root_logger.info("Finished evaluating......")
    elif FLAGS.mode == 'all':
        root_logger.info("Start all......")
        root_logger.info("Start training......")
        if FLAGS.model_last_checkpoint_file is not None:
            learned_model = torch.load(FLAGS.checkpoint_dir + "model_checkpoint.pth")
            learned_model.to(device)
        model = params['model'].Model(params, output_normalizer)
        model.to(device)
        learner(params, model)
        root_logger.info("Finished training......")
        root_logger.info("Start evaluating......")
        learned_model = torch.load(FLAGS.checkpoint_dir + "model_checkpoint.pth")
        learned_model.to(device)
        learned_model.eval()
        model = params['model'].Model(params, learned_model, output_normalizer)
        model.eval()
        model.to(device)
        evaluator(params, model)
        root_logger.info("Finished evaluating......")
        root_logger.info("Finished all......")
    elif FLAGS.mode == 'test_gcn':
        root_logger.info("Start all of test_gcn......")
        model = params['network'].Model()
        model.to(device)
        learner(params, model)
        learned_model = torch.load(FLAGS.checkpoint_dir + "model_checkpoint.pth")
        learned_model.to(device)
        learned_model.eval()
        model = params['model'].Model(params, learned_model, output_normalizer)
        model.eval()
        model.to(device)
        evaluator(params, model)
        root_logger.info("Finished all......")
    end = time.time()
    end_datetime = datetime.datetime.fromtimestamp(end).strftime('%c')
    root_logger.info("Program ended at time " + end_datetime)
    root_logger.info("Finished FLAGS.mode " + FLAGS.mode)
    elapsed_time_in_second = end - start
    elapsed_time = str(datetime.timedelta(seconds=elapsed_time_in_second))
    root_logger.info("Elapsed time " + elapsed_time)


if __name__ == '__main__':
    app.run(main)
