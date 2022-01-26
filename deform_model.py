"""Model for DeformingPlate."""

import torch
from torch import nn as nn
import torch.nn.functional as F

import common
import normalization
import encode_process_decode
import encode_process_decode_max_pooling
import encode_process_decode_lstm
import encode_process_decode_graph_structure_watcher

import torch_scatter

device = torch.device('cuda')


class Model(nn.Module):
    """Model for static cloth simulation."""

    def __init__(self, params, core_model_name=encode_process_decode, message_passing_aggregator='sum',
                 message_passing_steps=15, attention=False, ripple_used=False, ripple_generation=None,
                 ripple_generation_number=None,
                 ripple_node_selection=None, ripple_node_selection_random_top_n=None, ripple_node_connection=None,
                 ripple_node_ncross=None):
        super(Model, self).__init__()
        self._params = params
        self._output_normalizer = normalization.Normalizer(size=3, name='output_normalizer')
        self._node_normalizer = normalization.Normalizer(
            size=3 + common.NodeType.SIZE, name='node_normalizer')
        self._edge_normalizer = normalization.Normalizer(
            size=8, name='edge_normalizer')  # 3D coord + 3D coord + 2*length = 7
        self._model_type = params['model'].__name__
        self._displacement_base = None

        # for stochastic message passing
        '''
        self.random_walk_generation_interval = 399
        self.input_count = 0
        self.sto_mat = None
        self.normalized_adj_mat = None
        '''

        self.core_model_name = core_model_name
        self.core_model = self.select_core_model(core_model_name)
        self.message_passing_steps = message_passing_steps
        self.message_passing_aggregator = message_passing_aggregator
        self._attention = attention
        self._ripple_used = ripple_used
        if self._ripple_used:
            self._ripple_generation = ripple_generation
            self._ripple_generation_number = ripple_generation_number
            self._ripple_node_selection = ripple_node_selection
            self._ripple_node_selection_random_top_n = ripple_node_selection_random_top_n
            self._ripple_node_connection = ripple_node_connection
            self._ripple_node_ncross = ripple_node_ncross
        # self.stochastic_message_passing_used = False
        if self._ripple_used:
            self.learned_model = self.core_model.EncodeProcessDecode(
                output_size=params['size'],
                latent_size=128,
                num_layers=2,
                message_passing_steps=self.message_passing_steps,
                message_passing_aggregator=self.message_passing_aggregator, attention=self._attention,
                ripple_used=self._ripple_used,
                ripple_generation=self._ripple_generation, ripple_generation_number=self._ripple_generation_number,
                ripple_node_selection=self._ripple_node_selection,
                ripple_node_selection_random_top_n=self._ripple_node_selection_random_top_n,
                ripple_node_connection=self._ripple_node_connection,
                ripple_node_ncross=self._ripple_node_ncross)
        else:
            self.learned_model = self.core_model.EncodeProcessDecode(
                output_size=params['size'],
                latent_size=128,
                num_layers=2,
                message_passing_steps=self.message_passing_steps,
                message_passing_aggregator=self.message_passing_aggregator, attention=self._attention,
                ripple_used=self._ripple_used)

    def select_core_model(self, core_model_name):
        return {
            'encode_process_decode': encode_process_decode,
            'encode_process_decode_graph_structure_watcher': encode_process_decode_graph_structure_watcher,
            'encode_process_decode_max_pooling': encode_process_decode_max_pooling,
            'encode_process_decode_lstm': encode_process_decode_lstm,
        }.get(core_model_name, encode_process_decode)

    def unsorted_segment_operation(self, data, segment_ids, num_segments, operation):
        """
        Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

        :param data: A tensor whose segments are to be summed.
        :param segment_ids: The segment indices tensor.
        :param num_segments: The number of segments.
        :return: A tensor of same data type as the data argument.
        """
        assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

        # segment_ids is a 1-D tensor repeat it to have the same shape as data
        if len(segment_ids.shape) == 1:
            s = torch.prod(torch.tensor(data.shape[1:])).long().to(device)
            segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:]).to(device)

        assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

        shape = [num_segments] + list(data.shape[1:])
        result = torch.zeros(*shape)
        if operation == 'sum':
            result = torch_scatter.scatter_add(data.float(), segment_ids, dim=0, dim_size=num_segments)
            print(result)
        elif operation == 'max':
            result, _ = torch_scatter.scatter_max(data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'mean':
            result = torch_scatter.scatter_mean(data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'min':
            result, _ = torch_scatter.scatter_min(data.float(), segment_ids, dim=0, dim_size=num_segments)
        else:
            raise Exception('Invalid operation type!')
        result = result.type(data.dtype)
        return result

    def _build_graph(self, inputs, is_training):
        """Builds input graph."""
        world_pos = inputs['world_pos']
        node_type = inputs['node_type']

        # mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.NORMAL.value], device=device))
        mask = None

        one_hot_node_type = F.one_hot(node_type[:, 0].to(torch.int64), common.NodeType.SIZE).float()

        cells = inputs['cells']
        decomposed_cells = common.triangles_to_edges(cells, deform=True)
        senders, receivers = decomposed_cells['two_way_connectivity']

        mesh_pos = inputs['mesh_pos']
        relative_world_pos = (torch.index_select(input=world_pos, dim=0, index=senders) -
                              torch.index_select(input=world_pos, dim=0, index=receivers))
        relative_mesh_pos = (torch.index_select(mesh_pos, 0, senders) -
                             torch.index_select(mesh_pos, 0, receivers))
        edge_features = torch.cat((
            relative_world_pos,
            torch.norm(relative_world_pos, dim=-1, keepdim=True),
            relative_mesh_pos,
            torch.norm(relative_mesh_pos, dim=-1, keepdim=True)), dim=-1)

        mesh_edges = self.core_model.EdgeSet(
            name='mesh_edges',
            features=self._edge_normalizer(edge_features, is_training),
            receivers=receivers,
            senders=senders)

        if self._ripple_used:
            num_nodes = node_type.shape[0]
            node_dynamic = self.unsorted_segment_operation(torch.abs(relative_world_pos), receivers, num_nodes,
                                                                operation='sum').to(device)
            if self._displacement_base is None or self._displacement_base.shape[0] != node_dynamic.shape[0]:
                self._displacement_base = node_dynamic
            node_displacement = torch.abs(node_dynamic - self._displacement_base)
            node_features = torch.cat((node_displacement, one_hot_node_type), dim=-1)
        else:
            node_features = one_hot_node_type

        if self.core_model == encode_process_decode and self._ripple_used == True:

            return (self.core_model.MultiGraphWithPos(node_features=self._node_normalizer(node_features, is_training),
                                                     edge_sets=[mesh_edges], target_feature=world_pos,
                                                     mesh_pos=mesh_pos, model_type=self._model_type), mask)
        else:

            return (self.core_model.MultiGraph(node_features=self._node_normalizer(node_features, is_training),
                                              edge_sets=[mesh_edges]), mask)

    def forward(self, inputs, is_training):
        graph, mask = self._build_graph(inputs, is_training=is_training)
        if is_training:
            return self.learned_model(graph, self._edge_normalizer, is_training=is_training, mask=mask)
        else:
            return self._update(inputs, self.learned_model(graph, self._edge_normalizer, is_training=is_training, mask=mask))

    def _update(self, inputs, per_node_network_output):
        """Integrate model outputs."""
        velocity = self._output_normalizer.inverse(per_node_network_output)
        # print(velocity[187])

        # integrate forward
        cur_position = inputs['world_pos']
        position = cur_position + velocity
        return (position, cur_position, velocity)
        # return inputs['world_pos']

    def get_output_normalizer(self):
        return self._output_normalizer

    def save_model(self, path):
        torch.save(self.learned_model, path + "_learned_model.pth")
        torch.save(self._output_normalizer, path + "_output_normalizer.pth")
        torch.save(self._edge_normalizer, path + "_edge_normalizer.pth")
        torch.save(self._node_normalizer, path + "_node_normalizer.pth")

    def load_model(self, path):
        self.learned_model = torch.load(path + "_learned_model.pth")
        self._output_normalizer = torch.load(path + "_output_normalizer.pth")
        self._edge_normalizer = torch.load(path + "_edge_normalizer.pth")
        self._node_normalizer = torch.load(path + "_node_normalizer.pth")

    def evaluate(self):
        self.eval()
        self.learned_model.eval()
