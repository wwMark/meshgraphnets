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
import common
import logging

import numpy as np
import json
from common import NodeType

import time
import datetime

import PyG_GCN
from PyG_GCN import gcn

device = torch.device('cuda')

FLAGS = flags.FLAGS
flags.DEFINE_enum('mode', 'train', ['train', 'eval', 'all', 'test_gcn'],
                  'Train model, or run evaluation.')
flags.DEFINE_enum('model', 'cloth', ['cfd', 'cloth', 'gcn'],
                  'Select model to run.')
flags.DEFINE_enum('network', 'PyG_GCN', ['mgn', 'PyG_GCN'], 'Select network to train.')

flags.DEFINE_enum('rollout_split', 'valid', ['train', 'test', 'valid'],
                  'Dataset split to use for rollouts.')
flags.DEFINE_integer('epochs', 2, 'No. of training epochs')
flags.DEFINE_integer('trajectories', 2, 'No. of training trajectories')
flags.DEFINE_integer('num_rollouts', 2, 'No. of rollout trajectories')

start = time.time()
start_datetime = datetime.datetime.fromtimestamp(start).strftime('%c')
start_datetime_dash = start_datetime.replace(" ", "-").replace(":", "-")

root_dir = pathlib.Path(__file__).parent.resolve()
dataset_name = 'flag_simple'
dataset_dir = os.path.join(root_dir, 'data', dataset_name)
output_dir = os.path.join(root_dir, 'output')
run_dir = os.path.join(output_dir, start_datetime_dash)

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
                    # None,
                    os.path.join('C:\\Users\\Mark\\iCloudDrive\\master_arbeit\\implementation\\meshgraphnets\\output\\Fri-Oct--8-20-25-39-2021', 'checkpoint_dir'),
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
    # 'cfd': dict(noise=0.02, gamma=1.0, field='velocity', history=False,
    #             size=2, batch=2, model=cfd_model, evaluator=cfd_eval),
    'cloth': dict(noise=0.003, gamma=0.1, field='world_pos', history=True,
                  size=3, batch=1, model=cloth_model, evaluator=cloth_eval),
    'gcn': dict(noise=0.003, gamma=0.1, field='world_pos', history=True,
                size=3, batch=1, model=gcn, evaluator=cloth_eval),
}

# output_normalizer = normalization.Normalizer(size=3, name='output_normalizer')
loaded_meta = False
shapes = {}
dtypes = {}
types = {}

def squeeze_data_frame(data_frame):
    for k, v in data_frame.items():
        data_frame[k] = torch.squeeze(v, 0)
    return data_frame

def add_targets():
    """Adds target and optionally history fields to dataframe."""
    fields = 'world_pos'
    add_history = True

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

def split_and_preprocess():
    """Splits trajectories into frames, and adds training noise."""
    noise_field = 'world_pos'
    noise_scale = 0.003
    noise_gamma = 0.1

    def add_noise(frame):
        zero_size = torch.zeros(frame[noise_field].size(), dtype=torch.float32).to(device)
        noise = torch.normal(zero_size, std=noise_scale).to(device)
        other = torch.Tensor([NodeType.NORMAL.value]).to(device)
        mask = torch.eq(frame['node_type'], other.int())[:, 0]
        mask = torch.stack((mask, mask, mask), dim=1)
        noise = torch.where(mask, noise, torch.zeros_like(noise))
        frame[noise_field] += noise
        frame['target|' + noise_field] += (1.0 - noise_gamma) * noise
        return frame

    def element_operation(trajectory):
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
    return element_operation

def process_trajectory(trajectory_data, add_targets_bool=False, split_and_preprocess_bool=False):
    global loaded_meta
    global shapes
    global dtypes
    global types
    if not loaded_meta:
        try:
            with open(os.path.join(FLAGS.dataset_dir, 'meta.json'), 'r') as fp:
                meta = json.loads(fp.read())
            shapes = {}
            dtypes = {}
            types = {}
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

    '''
    if self.add_targets is not None:
        trajectory = self.add_targets(trajectory)
    if self.split_and_preprocess is not None:
        trajectory = self.split_and_preprocess(trajectory)
    '''
    if add_targets_bool:
        trajectory = add_targets()(trajectory)
    if split_and_preprocess_bool:
        trajectory = split_and_preprocess()(trajectory)
    return trajectory

def learner(model):
    # handles dataset preprocessing, model definition, training process definition and model training

    # dataset preprocessing
    # batch size can be defined in load_dataset. Default to 1.

    root_logger = logging.getLogger()

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
    for epoch in range(FLAGS.epochs - trained_epoch):
        ds_loader = dataset.load_dataset(FLAGS.dataset_dir, 'train', batch_size=batch_size, prefetch_factor=prefetch_factor,
                                     add_targets=True, split_and_preprocess=True)
        # every time when model.train is called, model will train itself with the whole dataset
        root_logger.info("Epoch " + str(epoch + 1) + "/" + str(FLAGS.epochs))
        epoch_training_loss = 0.0
        ds_iterator = iter(ds_loader)
        for trajectory_index in range(FLAGS.trajectories):
            root_logger.info("    trajectory index " + str(trajectory_index + 1) + "/" + str(FLAGS.trajectories))
            trajectory = next(ds_iterator)
            trajectory = process_trajectory(trajectory, True, True)
            trajectory_loss = 0.0
            for data_frame_index, data_frame in enumerate(trajectory):
                count += 1
                data_frame = squeeze_data_frame(data_frame)
                network_output = model(data_frame, is_training)
                loss = loss_fn(data_frame, network_output, model)
                if count % 1000 == 0:
                    root_logger.info("    1000 step loss " + str(loss))
                if pass_count > 0:
                    pass_count -= 1
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    trajectory_loss += loss.detach()
                    optimizer.step()
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
                   os.path.join(FLAGS.checkpoint_dir, "epoch_optimizer_checkpoint" + "_" + str((epoch + 1) % 2) + ".pth"))
        torch.save(scheduler.state_dict(),
                   os.path.join(FLAGS.checkpoint_dir, "epoch_scheduler_checkpoint" + "_" + str((epoch + 1) % 2) + ".pth"))
        if epoch == (FLAGS.epochs // 2):
            scheduler.step()
        torch.save({'epoch': epoch}, os.path.join(FLAGS.checkpoint_dir, "epoch_checkpoint.pth"))
    model.save_model(os.path.join(FLAGS.checkpoint_dir, "model_checkpoint"))
    torch.save(optimizer.state_dict(), os.path.join(FLAGS.checkpoint_dir, "optimizer_checkpoint.pth"))
    torch.save(scheduler.state_dict(), os.path.join(FLAGS.checkpoint_dir, "scheduler_checkpoint.pth"))
    loss_record = {}
    loss_record['train_total_loss'] = torch.sum(torch.stack(epoch_training_losses))
    loss_record['train_mean_epoch_loss'] = torch.mean(torch.stack(epoch_training_losses))
    loss_record['train_epoch_losses'] = epoch_training_losses
    return loss_record

def loss_fn(inputs, network_output, model):
    """L2 loss on position."""
    # build target acceleration
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
        trajectory = process_trajectory(trajectory, True)
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
    with open(os.path.join(run_dir, "rollout", "rollout.pkl"), 'wb') as fp:
        pickle.dump(trajectories, fp)
    loss_record = {}
    loss_record['eval_total_mse_loss'] = torch.sum(torch.stack(mse_losses))
    loss_record['eval_total_l1_loss'] = torch.sum(torch.stack(l1_losses))
    loss_record['eval_mean_mse_loss'] = torch.mean(torch.stack(mse_losses))
    loss_record['eval_mean_l1_loss'] = torch.mean(torch.stack(l1_losses))
    loss_record['eval_mse_losses'] = mse_losses
    loss_record['eval_l1_losses'] = l1_losses
    return loss_record


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
        # all_subdirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
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
        model = params['model'].Model(params)
        if FLAGS.model_last_checkpoint_dir is not None:
            model.load_model(os.path.join(FLAGS.model_last_checkpoint_dir, "model_checkpoint"))
            root_logger.info("Loaded checkpoint file in " + str(FLAGS.model_last_checkpoint_dir) + "and starting retraining...")
        model.to(device)
        train_loss_record = learner(model)
        root_logger.info("Finished training......")
    if FLAGS.mode == 'eval' or FLAGS.mode == 'all':
        root_logger.info("Start evaluating......")
        model = params['model'].Model(params)
        model.load_model(os.path.join(run_dir, "checkpoint_dir", "model_checkpoint"))
        root_logger.info("Loaded model from " + str(run_dir))
        model.evaluate()
        model.to(device)
        eval_loss_record = evaluator(params, model)
        root_logger.info("Finished evaluating......")
    if FLAGS.mode == 'test_gcn':
        print("Start all of test_gcn......")
        model = PyG_GCN.Model()
        model.to(device)
        learner(model)
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
    root_logger.info("--------------------train loss record--------------------")
    if FLAGS.mode == "train" or FLAGS.mode == "all":
        for item in train_loss_record.items():
            root_logger.info(item)
    root_logger.info("---------------------------------------------------------")
    root_logger.info("--------------------eval loss record---------------------")
    if FLAGS.mode == "eval" or FLAGS.mode == "all":
        for item in eval_loss_record.items():
            root_logger.info(item)
    root_logger.info("---------------------------------------------------------")


if __name__ == '__main__':
    app.run(main)
