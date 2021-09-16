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

import time
import datetime

host_system = 'linux'
start_datetime_global = None

device = torch.device('cuda')

FLAGS = flags.FLAGS
flags.DEFINE_enum('mode', 'all', ['train', 'eval', 'all'],
                  'Train model, or run evaluation.')
flags.DEFINE_enum('model', 'cloth', ['cfd', 'cloth'],
                  'Select model to run.')
flags.DEFINE_enum('rollout_split', 'valid', ['train', 'test', 'valid'],
                  'Dataset split to use for rollouts.')
flags.DEFINE_integer('epochs', 3, 'No. of training epochs')
flags.DEFINE_integer('trajectories', 1000, 'No. of training trajectories')
flags.DEFINE_integer('num_rollouts', 100, 'No. of rollout trajectories')

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


def learner(params, model):
    # handles dataset preprocessing, model definition, training process definition and model training

    # dataset preprocessing
    # batch size can be defined in load_dataset. Default to 1.

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

    for epoch in range(FLAGS.epochs):
        # every time when model.train is called, model will train itself with the whole dataset
        print("Epoch", epoch + 1, "/", FLAGS.epochs)
        ds_iterator = iter(ds_loader)
        for trajectory_index in range(FLAGS.trajectories):
            if host_system == 'linux':
                print("    program started on", start_datetime_global, ", now in Epoch", epoch)
            print("    trajectory index", trajectory_index + 1, "/", batches_in_dataset)
            data = next(ds_iterator)
            trajectory_loss = 0.0
            for data_frame_index, data_frame in enumerate(data):
                network_output = model(data_frame, is_training)
                # print("        Finished", data_frame_index + 1, "frame.")
                loss = loss_fn(data_frame, network_output)
                trajectory_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print("        trajectory_loss")
            print("       ", trajectory_loss)
            torch.save(model.learned_model,
                       FLAGS.checkpoint_dir + "trajectory_model_checkpoint" + "_" + str((trajectory_index + 1) % 2) + ".pth")
            torch.save(optimizer.state_dict(),
                       FLAGS.checkpoint_dir + "trajectory_optimizer_checkpoint" + "_" + str((trajectory_index + 1) % 2) + ".pth")
            torch.save(scheduler.state_dict(),
                       FLAGS.checkpoint_dir + "trajectory_scheduler_checkpoint" + "_" + str((trajectory_index + 1) % 2) + ".pth")
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
    world_pos = torch.squeeze(inputs['world_pos'], dim=0)
    prev_world_pos = torch.squeeze(inputs['prev|world_pos'], dim=0)
    target_world_pos = torch.squeeze(inputs['target|world_pos'], dim=0)

    cur_position = world_pos
    prev_position = prev_world_pos
    target_position = target_world_pos
    target_acceleration = target_position - 2 * cur_position + prev_position
    # target_normalized = output_normalizer(target_acceleration).to(device)

    # build loss
    node_type = torch.squeeze(inputs['node_type'], dim=0)
    loss_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.NORMAL.value], device=device).int())
    # error = torch.sum((target_normalized - network_output) ** 2, dim=1)
    error = torch.sum((target_acceleration - network_output) ** 2, dim=1)
    loss = torch.mean(error[loss_mask])
    return loss


def evaluator(params, model):
    """Run a model rollout trajectory."""
    ds_loader = dataset.load_dataset(FLAGS.dataset_dir, FLAGS.rollout_split, add_targets=True)
    ds_iterator = iter(ds_loader)
    trajectories = []

    # scalars = []
    for index in range(FLAGS.num_rollouts):
        print("Evaluating trajectory", index + 1)
        trajectory = next(ds_iterator)
        _, prediction_trajectory = params['evaluator'].evaluate(model, trajectory)
        mse_loss_fn = torch.nn.MSELoss()
        l1_loss_fn = torch.nn.L1Loss()
        mse_loss = mse_loss_fn(torch.squeeze(trajectory['world_pos'], dim=0), prediction_trajectory['pred_pos'])
        l1_loss = l1_loss_fn(torch.squeeze(trajectory['world_pos'], dim=0), prediction_trajectory['pred_pos'])
        print("    evaluation mse loss")
        print("   ", mse_loss)
        print("    evaluation l1 loss")
        print("   ", l1_loss)
        trajectories.append(prediction_trajectory)
        # scalars.append(scalar_data)
    with open(FLAGS.rollout_path, 'wb') as fp:
        pickle.dump(trajectories, fp)


def main(argv):
    start = time.time()
    start_datetime = datetime.datetime.fromtimestamp(start).strftime('%c')
    print("Program started at time", start_datetime)
    global start_datetime_global
    start_datetime_global = start_datetime
    params = PARAMETERS[FLAGS.model]

    if FLAGS.mode == 'train':
        print("Start training......")
        if FLAGS.model_last_checkpoint_file is not None:
            print("Loaded checkpoint file", FLAGS.model_last_checkpoint_file)
            learned_model = torch.load(FLAGS.checkpoint_dir + "model_checkpoint.pth")
            learned_model.to(device)
        else:
            learned_model = encode_process_decode.EncodeProcessDecode(
                output_size=params['size'],
                latent_size=128,
                num_layers=2,
                message_passing_steps=15)
            learned_model.to(device)
        model = params['model'].Model(params, learned_model, output_normalizer)
        model.to(device)
        # wandb.watch(model, log_freq=100)
        learner(params, model)
        print("Finished training......")
    elif FLAGS.mode == 'eval':
        print("Start evaluating......")
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
        print("Finished evaluating......")
    elif FLAGS.mode == 'all':
        print("Start all......")
        if FLAGS.model_last_checkpoint_file is not None:
            learned_model = torch.load(FLAGS.checkpoint_dir + "model_checkpoint.pth")
            learned_model.to(device)
        else:
            learned_model = encode_process_decode.EncodeProcessDecode(
                output_size=params['size'],
                latent_size=128,
                num_layers=2,
                message_passing_steps=15)
            learned_model.to(device)
        model = params['model'].Model(params, learned_model, output_normalizer)
        model.to(device)
        learner(params, model)
        learned_model = torch.load(FLAGS.checkpoint_dir + "model_checkpoint.pth")
        learned_model.to(device)
        learned_model.eval()
        model = params['model'].Model(params, learned_model, output_normalizer)
        model.eval()
        model.to(device)
        evaluator(params, model)
        print("Finished all......")
    end = time.time()
    end_datetime = datetime.datetime.fromtimestamp(end).strftime('%c')
    print("Program ended at time", end_datetime)
    print("Finished FLAGS.mode", FLAGS.mode)
    elapsed_time_in_second = end - start
    elapsed_time = str(datetime.timedelta(seconds=elapsed_time_in_second))
    print("Elapsed time", elapsed_time)


if __name__ == '__main__':
    app.run(main)
