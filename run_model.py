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
# from meshgraphnets import cfd_eval
# from meshgraphnets import cfd_model
from meshgraphnets import cloth_eval
from meshgraphnets import cloth_model
# from meshgraphnets import core_model
from meshgraphnets import dataset
from meshgraphnets import normalization, common
from meshgraphnets.test import encode_process_decode

# from torchsummary import summary


FLAGS = flags.FLAGS
flags.DEFINE_enum('mode', 'train', ['train', 'eval'],
                  'Train model, or run evaluation.')
flags.DEFINE_enum('model', None, ['cfd', 'cloth'],
                  'Select model to run.')
flags.DEFINE_string('checkpoint_dir', None, 'Directory to save checkpoint')
flags.DEFINE_string('dataset_dir', 'C:\\Users\\Mark\\iCloudDrive\\master_arbeit\\deepmind-research\\meshgraphnets\\data\\flag_simple\\', 'Directory to load dataset from.')
flags.DEFINE_string('rollout_path', None,
                    'Pickle file to save eval trajectories')
flags.DEFINE_enum('rollout_split', 'valid', ['train', 'test', 'valid'],
                  'Dataset split to use for rollouts.')
flags.DEFINE_integer('num_rollouts', 10, 'No. of rollout trajectories')
flags.DEFINE_integer('num_training_steps', int(10e6), 'No. of training steps')

PARAMETERS = {
    # 'cfd': dict(noise=0.02, gamma=1.0, field='velocity', history=False,
    #             size=2, batch=2, model=cfd_model, evaluator=cfd_eval),
    'cloth': dict(noise=0.003, gamma=0.1, field='world_pos', history=True,
                  size=3, batch=1, model=cloth_model, evaluator=cloth_eval)
}

output_normalizer = normalization.Normalizer(size=3, name='output_normalizer')

def learner(params):
  # handles dataset preprocessing, model definition, training process definition and model training

  # dataset preprocessing
  # batch size can be defined in load_dataset. Default to 1.
  add_targets = dataset.add_targets([params['field']], add_history=params['history'])
  split_and_preprocess = dataset.split_and_preprocess(noise_field=params['field'],
                                    noise_scale=params['noise'],
                                    noise_gamma=params['gamma'])
  batch_size = 1
  ds_loader = dataset.load_dataset(FLAGS.dataset_dir, 'train', add_targets=add_targets, split_and_preprocess=split_and_preprocess, batch_size=batch_size)
  
  # model definition
  # dataset will be passed to model, and some specific size of the dataset will be calculated inside model
  # then networks will be initialized
  '''
  print("type of ds loader", type(ds_loader))
  print("ds loader iter", iter(ds_loader))
  print("type of ds loader iter next", type(next(iter(ds_loader))))
  '''
  trajectory_state_for_init = next(iter(ds_loader))[0]
  model = params['model'].Model(trajectory_state_for_init, params)
  model.cuda()

  # training process definition 
  optimizer = torch.optim.Adam(model.learned_model.parameters(recurse=True), lr=1e-4)
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1 + 1e-6, last_epoch=-1)

  # model training
  is_training = True
  ds_iterator = iter(ds_loader)
  batches_in_dataset = 1000 // batch_size
  for epoch in range(FLAGS.num_training_steps // batches_in_dataset):
      # every time when model.train is called, model will train itself with the whole dataset
      print("Epoch", epoch)
      for batch_index in range(batches_in_dataset):
        data = next(ds_iterator)
        print("    Batch index is", batch_index)
        for data_frame in data:
          network_output = model(data_frame, is_training)
          print("        Finished one frame.")
          optimizer.zero_grad()
          loss = torch.nn.MSELoss()(target(inputs=data_frame), network_output)
          # loss_mask = torch.equal(input['node_type'][:, 0], common.NodeType.NORMAL)
          loss.backward()
          optimizer.step()
        scheduler.step()

def target(inputs):
    """L2 loss on position."""
    # build target acceleration
    cur_position = inputs['world_pos']
    prev_position = inputs['prev|world_pos']
    target_position = inputs['target|world_pos']
    target_acceleration = target_position - 2*cur_position + prev_position
    target_normalized = output_normalizer(target_acceleration)
    '''
    # build loss
    loss_mask = torch.equal(inputs['node_type'][:, 0], common.NodeType.NORMAL)
    error = torch.sum((target_normalized - network_output)**2, dim=1)
    loss = torch.mean(error[loss_mask])
    return loss
    '''
    return target_normalized.cuda()


  
'''
def evaluator(model, params):
  """Run a model rollout trajectory."""
  ds = dataset.load_dataset(FLAGS.dataset_dir, FLAGS.rollout_split)
  ds = dataset.add_targets(ds, [params['field']], add_history=params['history'])
  inputs = tf.data.make_one_shot_iterator(ds).get_next()
  scalar_op, traj_ops = params['evaluator'].evaluate(model, inputs)
  tf.train.create_global_step()

  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=FLAGS.checkpoint_dir,
      save_checkpoint_secs=None,
      save_checkpoint_steps=None) as sess:
    trajectories = []
    scalars = []
    for traj_idx in range(FLAGS.num_rollouts):
      logging.info('Rollout trajectory %d', traj_idx)
      scalar_data, traj_data = sess.run([scalar_op, traj_ops])
      trajectories.append(traj_data)
      scalars.append(scalar_data)
    for key in scalars[0]:
      logging.info('%s: %g', key, np.mean([x[key] for x in scalars]))
    with open(FLAGS.rollout_path, 'wb') as fp:
      pickle.dump(trajectories, fp)
'''

def main(argv):
  del argv
  params = PARAMETERS[FLAGS.model]
  '''
  learned_model = core_model.EncodeProcessDecode(
      output_size=params['size'],
      latent_size=128,
      num_layers=2,
      message_passing_steps=15)
  '''
  if FLAGS.mode == 'train':
    learner(params)
  elif FLAGS.mode == 'eval':
    # evaluator(params)
    pass

if __name__ == '__main__':
  app.run(main)
