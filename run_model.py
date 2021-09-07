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
# python -m meshgraphnets.run_model --model=cloth --mode=train --checkpoint_dir="C:\Users\Mark\iCloudDrive\master_arbeit\implementation\meshgraphnets\tmp\checkpoint\" --num_training_steps=1000
import pickle
from absl import app
from absl import flags
from absl import logging
import numpy as np
# import tensorflow.compat.v1 as tf
import torch

from test import encode_process_decode

# from meshgraphnets import cfd_eval
# from meshgraphnets import cfd_model
'''
from meshgraphnets import cloth_eval
from meshgraphnets import cloth_model
# from meshgraphnets import core_model
from meshgraphnets import dataset
from meshgraphnets import normalization, common
'''
import cloth_eval
import cloth_model
# from meshgraphnets import core_model
import dataset
import normalization, common
from test import encode_process_decode

import time

# from torchsummary import summary

device = torch.device('cuda')

FLAGS = flags.FLAGS
flags.DEFINE_enum('mode', 'eval', ['train', 'eval', 'all'],
                  'Train model, or run evaluation.')
flags.DEFINE_enum('model', 'cloth', ['cfd', 'cloth'],
                  'Select model to run.')
flags.DEFINE_string('checkpoint_dir', 'C:\\Users\\Mark\\iCloudDrive\\master_arbeit\\implementation\\meshgraphnets\\checkpoint_dir\\checkpoint.pth', 'Directory to save checkpoint')
flags.DEFINE_string('dataset_dir',
                    'C:\\Users\\Mark\\iCloudDrive\\master_arbeit\\deepmind-research\\meshgraphnets\\data\\flag_simple\\',
                    'Directory to load dataset from.')
flags.DEFINE_string('rollout_path', 'C:\\Users\\Mark\\iCloudDrive\\master_arbeit\\implementation\\meshgraphnets\\rollout\\rollout.pkl',
                    'Pickle file to save eval trajectories')
flags.DEFINE_enum('rollout_split', 'valid', ['train', 'test', 'valid'],
                  'Dataset split to use for rollouts.')
flags.DEFINE_integer('num_rollouts',2, 'No. of rollout trajectories')
flags.DEFINE_integer('num_training_steps', int(10e2), 'No. of training steps')

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
    '''
    add_targets = dataset.add_targets([params['field']], add_history=params['history'])
    split_and_preprocess = dataset.split_and_preprocess(noise_field=params['field'],
                                      noise_scale=params['noise'],
                                      noise_gamma=params['gamma'])
    '''
    batch_size = 1
    prefetch_factor = 2
    ds_loader = dataset.load_dataset(FLAGS.dataset_dir, 'train', batch_size=batch_size, prefetch_factor=prefetch_factor, add_targets=True, split_and_preprocess=True)
    # ds_loader = dataset.load_dataset(FLAGS.dataset_dir, 'train', add_targets=add_targets, split_and_preprocess=split_and_preprocess, batch_size=batch_size)

    # model definition
    # dataset will be passed to model, and some specific size of the dataset will be calculated inside model
    # then networks will be initialized
    '''
    print("type of ds loader", type(ds_loader))
    print("ds loader iter", iter(ds_loader))
    print("type of ds loader iter next", type(next(iter(ds_loader))))
    '''
    trajectory_state_for_init = next(iter(ds_loader))

    # training process definition
    optimizer = torch.optim.Adam(model.learned_model.parameters(recurse=True), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1 + 1e-6, last_epoch=-1)

    # model training
    is_training = True
    ds_iterator = iter(ds_loader)
    batches_in_dataset = 1000 // batch_size
    total_epoch = FLAGS.num_training_steps // batches_in_dataset
    for epoch in range(FLAGS.num_training_steps // batches_in_dataset):
        # every time when model.train is called, model will train itself with the whole dataset
        print("Epoch", epoch + 1, "/", total_epoch)
        # for batch_index in range(batches_in_dataset):
        for batch_index in [0]:
            print("    Batch index", batch_index + 1, "/", batches_in_dataset)
            data = next(ds_iterator)
            # print(data[0]['world_pos'].shape)
            # quit()
            # data = torch.squeeze(data, dim=-1)
            # for trajectory in data:
            optimizer.zero_grad()
            loss_fn = torch.nn.MSELoss()
            count = 0
            for data_frame_index, data_frame in enumerate(data):
                count += 1
                if count == 2:
                    break
                network_output = model(data_frame, is_training)
                print("        Finished", data_frame_index + 1, " frame.")
                loss = loss_fn(network_output, target(data_frame))
                # loss_mask = torch.equal(input['node_type'][:, 0], common.NodeType.NORMAL)
                loss.backward()
            optimizer.step()
            scheduler.step()
    # torch.save(model.learned_model.state_dict(), FLAGS.checkpoint_dir)
    torch.save(model.learned_model, FLAGS.checkpoint_dir)
    # torch.save(model, FLAGS.checkpoint_dir)
    return model

def target(inputs):
    """L2 loss on position."""
    # build target acceleration
    cur_position = inputs['world_pos']
    prev_position = inputs['prev|world_pos']
    target_position = inputs['target|world_pos']
    target_acceleration = target_position - 2 * cur_position + prev_position
    target_normalized = output_normalizer(target_acceleration)
    '''
    # build loss
    loss_mask = torch.equal(inputs['node_type'][:, 0], common.NodeType.NORMAL)
    error = torch.sum((target_normalized - network_output)**2, dim=1)
    loss = torch.mean(error[loss_mask])
    return loss
    '''
    return target_normalized.to(device)


def evaluator(params, model):
  """Run a model rollout trajectory."""
  ds_loader = dataset.load_dataset(FLAGS.dataset_dir, FLAGS.rollout_split, add_targets=True)
  ds_iterator = iter(ds_loader)
  trajectories = []
  # scalars = []
  for index in range(FLAGS.num_rollouts):
    print("Evaluating trajectory", index + 1)
    trajectory = ds_iterator.next()
    _, prediction_trajectory = params['evaluator'].evaluate(model, trajectory)
    trajectories.append(prediction_trajectory)
    # scalars.append(scalar_data)
  print("trajectories length", len(trajectories))
  with open(FLAGS.rollout_path, 'wb') as fp:
    pickle.dump(trajectories, fp)



def main(argv):
    del argv
    start = time.time()
    params = PARAMETERS[FLAGS.model]
    learned_model = encode_process_decode.EncodeProcessDecode(
        output_size=params['size'],
        latent_size=128,
        num_layers=2,
        message_passing_steps=15)
    model = params['model'].Model(params, learned_model)
    model.to(device)
    if FLAGS.mode == 'train':
        print("Start training......")
        learner(params, model)
        print("Finished training......")
    elif FLAGS.mode == 'eval':
        print("Start evaluating......")
        # learned_model.load_state_dict(torch.load(FLAGS.checkpoint_dir))
        learned_model = torch.load(FLAGS.checkpoint_dir)
        learned_model.to(device)
        learned_model.eval()
        model = params['model'].Model(params, learned_model)
        model.eval()
        model.to(device)
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
        learner(params, model)
        learned_model = torch.load(FLAGS.checkpoint_dir)
        learned_model.eval()
        model = params['model'].Model(params, learned_model)
        model.eval()
        model.to(device)
        evaluator(params, model)
        print("Finished all......")
    end = time.time()
    print("Finished FLAGS.mode", FLAGS.mode)
    print("Elapsed time", end - start)


if __name__ == '__main__':
    app.run(main)
