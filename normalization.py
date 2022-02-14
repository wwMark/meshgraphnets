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
"""Online data normalization."""

# import sonnet as snt
# import tensorflow.compat.v1 as tf
import torch
import torch.nn as nn

device = torch.device('cuda')


# class Normalizer(snt.AbstractModule):
class Normalizer(nn.Module):
    """Feature normalizer that accumulates statistics online."""

    def __init__(self, size, name, max_accumulations=10 ** 6, std_epsilon=1e-8, ):
        super(Normalizer, self).__init__()
        self._name = name
        self._max_accumulations = max_accumulations
        self._std_epsilon = torch.tensor([std_epsilon], requires_grad=False).to(device)

        self._acc_count = torch.zeros(1, dtype=torch.float32, requires_grad=False).to(device)
        self._num_accumulations = torch.zeros(1, dtype=torch.float32, requires_grad=False).to(device)
        self._acc_sum = torch.zeros(size, dtype=torch.float32, requires_grad=False).to(device)
        self._acc_sum_squared = torch.zeros(size, dtype=torch.float32, requires_grad=False).to(device)

    def forward(self, batched_data, node_num=None, accumulate=True):
        """Normalizes input data and accumulates statistics."""
        if accumulate and self._num_accumulations < self._max_accumulations:
            # stop accumulating after a million updates, to prevent accuracy issues
            self._accumulate(batched_data)
        return (batched_data - self._mean()) / self._std_with_epsilon()

    def inverse(self, normalized_batch_data):
        """Inverse transformation of the normalizer."""
        return normalized_batch_data * self._std_with_epsilon() + self._mean()

    def _accumulate(self, batched_data, node_num=None):
        """Function to perform the accumulation of the batch_data statistics."""
        count = torch.tensor(batched_data.shape[0], dtype=torch.float32, device=device)

        data_sum = torch.sum(batched_data, dim=0)
        squared_data_sum = torch.sum(batched_data ** 2, dim=0)
        self._acc_sum = self._acc_sum.add(data_sum)
        self._acc_sum_squared = self._acc_sum_squared.add(squared_data_sum)
        self._acc_count = self._acc_count.add(count)
        self._num_accumulations = self._num_accumulations.add(1.)

    def _mean(self):
        safe_count = torch.maximum(self._acc_count, torch.tensor([1.], device=device))
        return self._acc_sum / safe_count

    def _std_with_epsilon(self):
        safe_count = torch.maximum(self._acc_count, torch.tensor([1.], device=device))
        std = torch.sqrt(self._acc_sum_squared / safe_count - self._mean() ** 2)
        return torch.maximum(std, self._std_epsilon)

    def get_acc_sum(self):
        return self._acc_sum
