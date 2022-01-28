import torch
import collections
import find_influential_nodes

device = torch.device('cuda')

EdgeSet = collections.namedtuple('EdgeSet', ['name', 'features', 'senders',
                                             'receivers'])
MultiGraph = collections.namedtuple('Graph', ['node_features', 'edge_sets'])

# aggregate nodes into ripples
# returns node index in each ripple
class RippleGenerator():

    def __init__(self, ripple_generation, ripple_generation_number):
        self._ripple_generation_method = ripple_generation
        self._ripple_generation_number = ripple_generation_number

    def generate_ripple(self, graph):
        ripple_indices = []
        if self._ripple_generation_method == 'equal_size':
            ripple_number = self._ripple_generation_number
            target_feature_matrix = graph.target_feature
            num_nodes = target_feature_matrix.shape[0]
            ripple_size = num_nodes // ripple_number
            ripple_size_rest = num_nodes % ripple_number
            assert ripple_size > 0
            for i in range(ripple_number - 1):
                start_index = i * ripple_size
                end_index = (i + 1) * ripple_size
                ripple_indices.append((start_index, end_index))
            ripple_indices.append(((ripple_number - 1) * ripple_size, ripple_number * ripple_size + ripple_size_rest))
            return (ripple_indices, None)
        elif self._ripple_generation_method == 'gradient':
            # bins should be set as small as possible to ensure the nodes inside a bin has the greatest similarity and
            # as big as possible to ensure the similar nodes are assign to same group
            target_feature_matrix = graph.target_feature
            num_nodes = target_feature_matrix.shape[0]
            bins = 100
            take_n_bins = self._ripple_generation_number - 1
            velocity_matrix = graph.node_features[:, 0:3]
            norm = torch.linalg.vector_norm(velocity_matrix, dim=1)
            histogram = torch.histc(norm, bins=bins)
            values, indices = torch.topk(histogram, take_n_bins)
            for i in range(take_n_bins):
                '''
                start_index = torch.sum(histogram[:indices[i]])
                print(start_index.data.cpu().numpy() + values[i].data.cpu().numpy())
                print(type(start_index.data.cpu().numpy() + values[i].data.cpu().numpy()))
                quit()
                '''
                start_index = torch.sum(histogram[:indices[i]]).to(torch.int32)
                end_index = start_index + values[i]
                ripple_indices.append((start_index.item(), end_index.to(torch.int32).item()))
                ripple_indices.sort(key=lambda x: x[0])
            selected_nodes_concat = []
            for start_index, end_index in ripple_indices:
                selected_nodes = list(range(start_index, end_index))
                selected_nodes_concat.extend(selected_nodes)
            rest_nodes = list(range(0, num_nodes))
            rest_nodes = set(rest_nodes) - set(selected_nodes_concat)
            return (ripple_indices, rest_nodes)
        elif self._ripple_generation_method == 'exponential_size':
            base = self._ripple_generation_number
            exponential = 1
            target_feature_matrix = graph.target_feature
            num_nodes = target_feature_matrix.shape[0]
            start_index = 0
            while True:
                end_index = start_index + base ** exponential
                if end_index >= num_nodes:
                    end_index = num_nodes
                    ripple_indices.append((start_index, end_index))
                    return (ripple_indices, None)
                ripple_indices.append((start_index, end_index))
                exponential += 1
                start_index = end_index


# select node from ripple that will be connected with nodes from other ripples
# takes output of ripple generator as input, and output a list of list of indices which contains the selected nodes of each ripple
class RippleNodeSelector():

    def __init__(self, ripple_node_selection, ripple_node_selection_random_top_n):
        self._ripple_node_selection = ripple_node_selection
        self._ripple_node_selection_random_top_n = ripple_node_selection_random_top_n

    def select_nodes(self, ripple_tuple):
        selected_nodes = []
        ripple_list = ripple_tuple[0]
        ripple_rest = ripple_tuple[1]
        if self._ripple_node_selection == 'random':
            for ripple in ripple_list:
                ripple_size = ripple[1] - ripple[0]
                ripple_select_size = self._ripple_node_selection_random_top_n if self._ripple_node_selection_random_top_n <= ripple_size else ripple_size
                random_select_mask = torch.randperm(n=ripple_size)[0:ripple_select_size]
                selected_nodes.append(random_select_mask)
            if ripple_rest is not None:
                ripple_size = len(ripple_rest)
                ripple_select_size = self._ripple_node_selection_random_top_n if self._ripple_node_selection_random_top_n <= ripple_size else ripple_size
                random_select_mask = torch.randperm(n=ripple_size)[0:ripple_select_size]
                selected_nodes.append(random_select_mask)
        elif self._ripple_node_selection == 'top':
            for ripple in ripple_list:
                ripple_size = ripple[1] - ripple[0]
                ripple_select_size = self._ripple_node_selection_random_top_n if self._ripple_node_selection_random_top_n <= ripple_size else ripple_size
                selected_nodes.append(range(ripple_size)[:ripple_select_size])
            if ripple_rest is not None:
                ripple_size = len(ripple_rest)
                ripple_select_size = self._ripple_node_selection_random_top_n if self._ripple_node_selection_random_top_n <= ripple_size else ripple_size
                selected_nodes.append(range(ripple_size)[:ripple_select_size])
        elif self._ripple_node_selection == 'all':
            for ripple in ripple_list:
                ripple_size = ripple[1] - ripple[0]
                selected_nodes.append(range(ripple_size))
            if ripple_rest is not None:
                ripple_size = len(ripple_rest)
                selected_nodes.append(range(ripple_size))
        return selected_nodes

# connect the selected nodes
class RippleNodeConnector():
    def __init__(self, ripple_node_connection, ripple_node_ncross):
        self._ripple_node_connection = ripple_node_connection
        self._ripple_node_ncross = ripple_node_ncross

    def connect(self, graph, ripple_tuple, node_selections, edge_normalizer, is_training):
        model_type = graph.model_type
        velocity_matrix = graph.node_features[:, 0:3]

        velocity_norm = torch.norm(velocity_matrix, dim=1)
        _, sort_indices = torch.sort(velocity_norm, dim=0, descending=True)

        selected_nodes = []
        ripples = ripple_tuple[0]
        ripple_rest = ripple_tuple[1]
        for (start_index, end_index), node_mask in zip(ripples, node_selections):
            if end_index > start_index:
                ripple = sort_indices[start_index:end_index]
                selected_nodes.append(ripple[node_mask])
        if ripple_rest is not None:
            ripple = sort_indices[ripple_rest]
            selected_nodes.append(ripple[node_selections[-1]])

        if self._ripple_node_connection == 'most_influential':
            target_feature = graph.target_feature
            mesh_pos = graph.mesh_pos
            receivers_list = [index for sub_selected_nodes in selected_nodes for index in sub_selected_nodes]
            receivers_list.pop(0)
            senders_list = []
            senders_list.extend([sort_indices[0]] * len(receivers_list))
            senders = torch.cat(
                (torch.tensor(senders_list, device=device), torch.tensor(receivers_list, device=device)), dim=0)
            receivers = torch.cat(
                (torch.tensor(receivers_list, device=device), torch.tensor(senders_list, device=device)), dim=0)
            if model_type == 'cloth_model' or model_type == 'deform_model':
                relative_target_feature = (torch.index_select(input=target_feature, dim=0, index=senders) -
                                      torch.index_select(input=target_feature, dim=0, index=receivers))
                relative_mesh_pos = (torch.index_select(mesh_pos, 0, senders) -
                                     torch.index_select(mesh_pos, 0, receivers))
                edge_features = torch.cat((
                    relative_target_feature,
                    torch.norm(relative_target_feature, dim=-1, keepdim=True),
                    relative_mesh_pos,
                    torch.norm(relative_mesh_pos, dim=-1, keepdim=True)), dim=-1)
            elif model_type == 'cfd_model':
                relative_mesh_pos = (torch.index_select(mesh_pos, 0, senders) -
                                     torch.index_select(mesh_pos, 0, receivers))
                edge_features = torch.cat((
                    relative_mesh_pos,
                    torch.norm(relative_mesh_pos, dim=-1, keepdim=True)), dim=-1)
            else:
                raise Exception("Model type is not specified in RippleNodeConnector.")
            edge_features = edge_normalizer(edge_features, is_training)

            mesh_edges = graph.edge_sets[0]
            mesh_edges = mesh_edges._replace(features=torch.cat((mesh_edges.features, edge_features), dim=0))
            mesh_edges = mesh_edges._replace(senders=torch.cat((mesh_edges.senders, senders), dim=0))
            mesh_edges = mesh_edges._replace(receivers=torch.cat((mesh_edges.receivers, receivers), dim=0))
            new_graph = MultiGraph(node_features=graph.node_features, edge_sets=[mesh_edges])

        elif self._ripple_node_connection == 'fully_connected':
            target_feature = graph.target_feature
            mesh_pos = graph.mesh_pos
            for ripple_selected_nodes in selected_nodes:
                receivers_list = ripple_selected_nodes
                senders_list = ripple_selected_nodes
                senders = torch.cat(
                    (torch.tensor(senders_list, device=device), torch.tensor(receivers_list, device=device)), dim=0)
                receivers = torch.cat(
                    (torch.tensor(receivers_list, device=device), torch.tensor(senders_list, device=device)), dim=0)
                if model_type == 'cloth_model' or model_type == 'deform_model':
                    relative_target_feature = (torch.index_select(input=target_feature, dim=0, index=senders) -
                                               torch.index_select(input=target_feature, dim=0, index=receivers))
                    relative_mesh_pos = (torch.index_select(mesh_pos, 0, senders) -
                                         torch.index_select(mesh_pos, 0, receivers))
                    edge_features = torch.cat((
                        relative_target_feature,
                        torch.norm(relative_target_feature, dim=-1, keepdim=True),
                        relative_mesh_pos,
                        torch.norm(relative_mesh_pos, dim=-1, keepdim=True)), dim=-1)
                elif model_type == 'cfd_model':
                    relative_mesh_pos = (torch.index_select(mesh_pos, 0, senders) -
                                         torch.index_select(mesh_pos, 0, receivers))
                    edge_features = torch.cat((
                        relative_mesh_pos,
                        torch.norm(relative_mesh_pos, dim=-1, keepdim=True)), dim=-1)
                else:
                    raise Exception("Model type is not specified in RippleNodeConnector.")
                edge_features = edge_normalizer(edge_features, is_training)

                mesh_edges = graph.edge_sets[0]
                mesh_edges = mesh_edges._replace(features=torch.cat((mesh_edges.features, edge_features), dim=0))
                mesh_edges = mesh_edges._replace(senders=torch.cat((mesh_edges.senders, senders), dim=0))
                mesh_edges = mesh_edges._replace(receivers=torch.cat((mesh_edges.receivers, receivers), dim=0))
                new_graph = MultiGraph(node_features=graph.node_features, edge_sets=[mesh_edges])

        elif self._ripple_node_connection == 'fully_ncross_connected':
            target_feature = graph.target_feature
            mesh_pos = graph.mesh_pos
            cross_nodes = []
            for ripple_selected_nodes in selected_nodes:
                # select cross nodes
                # print(self._ripple_node_ncross)
                # print(ripple_selected_nodes.shape[0])
                assert self._ripple_node_ncross <= ripple_selected_nodes.shape[0]
                mask = torch.randperm(n=len(ripple_selected_nodes))[:self._ripple_node_ncross]
                for index in ripple_selected_nodes[mask]:
                    cross_nodes.append(index)

                receivers_list = ripple_selected_nodes
                senders_list = ripple_selected_nodes
                senders = torch.cat(
                    (torch.tensor(senders_list, device=device), torch.tensor(receivers_list, device=device)), dim=0)
                receivers = torch.cat(
                    (torch.tensor(receivers_list, device=device), torch.tensor(senders_list, device=device)), dim=0)
                if model_type == 'cloth_model' or model_type == 'deform_model':
                    relative_target_feature = (torch.index_select(input=target_feature, dim=0, index=senders) -
                                               torch.index_select(input=target_feature, dim=0, index=receivers))
                    relative_mesh_pos = (torch.index_select(mesh_pos, 0, senders) -
                                         torch.index_select(mesh_pos, 0, receivers))
                    edge_features = torch.cat((
                        relative_target_feature,
                        torch.norm(relative_target_feature, dim=-1, keepdim=True),
                        relative_mesh_pos,
                        torch.norm(relative_mesh_pos, dim=-1, keepdim=True)), dim=-1)
                elif model_type == 'cfd_model':
                    relative_mesh_pos = (torch.index_select(mesh_pos, 0, senders) -
                                         torch.index_select(mesh_pos, 0, receivers))
                    edge_features = torch.cat((
                        relative_mesh_pos,
                        torch.norm(relative_mesh_pos, dim=-1, keepdim=True)), dim=-1)
                else:
                    raise Exception("Model type is not specified in RippleNodeConnector.")
                edge_features = edge_normalizer(edge_features, is_training)

                mesh_edges = graph.edge_sets[0]
                mesh_edges = mesh_edges._replace(features=torch.cat((mesh_edges.features, edge_features), dim=0))
                mesh_edges = mesh_edges._replace(senders=torch.cat((mesh_edges.senders, senders), dim=0))
                mesh_edges = mesh_edges._replace(receivers=torch.cat((mesh_edges.receivers, receivers), dim=0))
                new_graph = MultiGraph(node_features=graph.node_features, edge_sets=[mesh_edges])

            # fully connect cross nodes
            receivers_list = cross_nodes
            senders_list = cross_nodes
            senders = torch.cat(
                (torch.tensor(senders_list, device=device, dtype=torch.int32), torch.tensor(receivers_list, device=device, dtype=torch.int32)), dim=0)
            receivers = torch.cat(
                (torch.tensor(receivers_list, device=device), torch.tensor(senders_list, device=device)), dim=0)
            if model_type == 'cloth_model' or model_type == 'deform_model':
                relative_target_feature = (torch.index_select(input=target_feature, dim=0, index=senders) -
                                           torch.index_select(input=target_feature, dim=0, index=receivers))
                relative_mesh_pos = (torch.index_select(mesh_pos, 0, senders) -
                                     torch.index_select(mesh_pos, 0, receivers))
                edge_features = torch.cat((
                    relative_target_feature,
                    torch.norm(relative_target_feature, dim=-1, keepdim=True),
                    relative_mesh_pos,
                    torch.norm(relative_mesh_pos, dim=-1, keepdim=True)), dim=-1)
            elif model_type == 'cfd_model':
                relative_mesh_pos = (torch.index_select(mesh_pos, 0, senders) -
                                     torch.index_select(mesh_pos, 0, receivers))
                edge_features = torch.cat((
                    relative_mesh_pos,
                    torch.norm(relative_mesh_pos, dim=-1, keepdim=True)), dim=-1)
            else:
                raise Exception("Model type is not specified in RippleNodeConnector.")
            edge_features = edge_normalizer(edge_features, is_training)

            mesh_edges = graph.edge_sets[0]
            mesh_edges = mesh_edges._replace(features=torch.cat((mesh_edges.features, edge_features), dim=0))
            mesh_edges = mesh_edges._replace(senders=torch.cat((mesh_edges.senders, senders), dim=0))
            mesh_edges = mesh_edges._replace(receivers=torch.cat((mesh_edges.receivers, receivers), dim=0))
            new_graph = MultiGraph(node_features=graph.node_features, edge_sets=[mesh_edges])
        return new_graph


# class that aggregates ripple generator, ripple node selector and ripple node connector
class RippleMachine():
    def __init__(self, ripple_generation, ripple_generation_number, ripple_node_selection,
                 ripple_node_selection_random_top_n, ripple_node_connection, ripple_node_ncross):
        self._ripple_generation = ripple_generation
        self._ripple_generation_number = ripple_generation_number
        self._radius = 0.01
        self._topk = 10
        if self._ripple_generation != 'random_nodes' and self._ripple_generation != 'distance_density':
            self._ripple_generator = RippleGenerator(ripple_generation, ripple_generation_number)
            self._ripple_node_selector = RippleNodeSelector(ripple_node_selection, ripple_node_selection_random_top_n)
            self._ripple_node_connector = RippleNodeConnector(ripple_node_connection, ripple_node_ncross)

    def add_meta_edges(self, graph, edge_normalizer, is_training):
        if self._ripple_generation == 'random_nodes' or self._ripple_generation == 'distance_density':
            target_feature = graph.target_feature
            mesh_pos = graph.mesh_pos
            if self._ripple_generation == 'random_nodes':
                selected_nodes = torch.randperm(n=target_feature.shape[0])[0:self._ripple_generation_number]
            if self._ripple_generation == 'distance_density':
                selected_nodes = find_influential_nodes.find_influential_nodes(graph.target_feature, self._radius, self._topk)
            receivers_list = selected_nodes
            senders_list = selected_nodes
            senders = torch.cat(
                (torch.tensor(senders_list, device=device), torch.tensor(receivers_list, device=device)), dim=0)
            receivers = torch.cat(
                (torch.tensor(receivers_list, device=device), torch.tensor(senders_list, device=device)), dim=0)

            model_type = graph.model_type
            if model_type == 'cloth_model' or model_type == 'deform_model':
                relative_target_feature = (torch.index_select(input=target_feature, dim=0, index=senders) -
                                           torch.index_select(input=target_feature, dim=0, index=receivers))
                relative_mesh_pos = (torch.index_select(mesh_pos, 0, senders) -
                                     torch.index_select(mesh_pos, 0, receivers))
                edge_features = torch.cat((
                    relative_target_feature,
                    torch.norm(relative_target_feature, dim=-1, keepdim=True),
                    relative_mesh_pos,
                    torch.norm(relative_mesh_pos, dim=-1, keepdim=True)), dim=-1)
            else:
                raise Exception("Model type is not specified in RippleNodeConnector.")
            edge_features = edge_normalizer(edge_features, is_training)

            mesh_edges = graph.edge_sets[0]
            mesh_edges = mesh_edges._replace(features=torch.cat((mesh_edges.features, edge_features), dim=0))
            mesh_edges = mesh_edges._replace(senders=torch.cat((mesh_edges.senders, senders), dim=0))
            mesh_edges = mesh_edges._replace(receivers=torch.cat((mesh_edges.receivers, receivers), dim=0))
            new_graph = MultiGraph(node_features=graph.node_features, edge_sets=[mesh_edges])
        else:
            ripple_indices = self._ripple_generator.generate_ripple(graph)
            selected_nodes = self._ripple_node_selector.select_nodes(ripple_indices)
            new_graph = self._ripple_node_connector.connect(graph, ripple_indices, selected_nodes, edge_normalizer, is_training)
        return new_graph

'''
class RippleConnectionGenerator(nn.Module):
    
    # ripple_size_generator will generate for each ripple a node size according to some math equation

    def __init__(self, make_mlp, output_size):
        super().__init__()
        self._ripple_model = self._ripple_params['model']

        # ripple model will learn ripple influence based on source feature and source-destination distance
        self.ripple_model = make_mlp(output_size)
        if self._normalize_connection:
            self.ripple_model = nn.Sequential(nn.LayerNorm(normalized_shape=17), self.ripple_model)

    def generate_ripple_sample_size(self, ripple_order, ripple_size, ripple_sample_size_generator):
        # Currently assume that exponential
        if ripple_sample_size_generator == 'equal':
            equal_generator_sample_size = self._equal_generator_sample_size
            if equal_generator_sample_size > ripple_size:
                print(equal_generator_sample_size)
                print(ripple_size)
                raise Exception(
                    'Equal ripple sample size generator does not work properly. Sample number of nodes in ripple are greater than number of all ripple nodes!')
            return equal_generator_sample_size
        elif ripple_sample_size_generator == 'exponential':
            sample_size = (ripple_order + 1) * (ripple_order + 1)
            if sample_size > ripple_size:
                raise Exception(
                    'Exponential ripple size generator does not work properly. Sample number of nodes in ripple are greater than number of all ripple nodes!')
            return sample_size

    def forward(self, latent_graph, graph):
        # get graph node features with the highest velocity
        world_pos_matrix = graph.world_pos
        mesh_pos_matrix = graph.mesh_pos
        velocity_matrix = graph.node_features[:, 0:3]
        num_nodes = world_pos_matrix.shape[0]
        num_influential_nodes = ceil(world_pos_matrix.shape[
                                         0] * self._num_or_percentage_value) if self._num_or_percentage == 'percentage' else self._num_or_percentage_value
        sort_by_velocity = torch.square(velocity_matrix)
        sort_by_velocity = torch.sum(sort_by_velocity, dim=-1)
        _, sort_indices = torch.sort(sort_by_velocity, dim=0, descending=True)

        # virtual highest node is the sum of a number of the highest node
        highest_velocity_node_feature = torch.sum(velocity_matrix[sort_indices[0:num_influential_nodes]], dim=0)
        highest_velocity_node_world_pos = torch.sum(world_pos_matrix[sort_indices[0:num_influential_nodes]], dim=0)
        highest_velocity_node_mesh_pos = torch.sum(mesh_pos_matrix[sort_indices[0:num_influential_nodes]], dim=0)

        # get all nodes that need to establish a connection with highest_velocity_node from velocity_matrix
        ripple_size = num_nodes // self._num_ripples
        ripple_size_rest = num_nodes % self._num_ripples
        ripple_nodes_feature = []
        ripple_nodes_index = []
        ripple_nodes_world_pos = []
        ripple_nodes_mesh_pos = []
        for i in range(self._num_ripples):
            start_index = i * ripple_size
            ripple_actual_size = ripple_size if i < self._num_ripples - 1 else ripple_size + ripple_size_rest
            ripple_sample_size = self.generate_ripple_sample_size(i, ripple_actual_size,
                                                                  ripple_sample_size_generator='equal')
            end_index = start_index + ripple_actual_size

            random_select_mask = torch.randperm(n=ripple_actual_size)[0:ripple_sample_size]
            random_select_mask = random_select_mask[0:ripple_sample_size]

            info_of_a_ripple = velocity_matrix[start_index:end_index]
            info_of_a_ripple = info_of_a_ripple[random_select_mask]
            ripple_nodes_feature.append(info_of_a_ripple)
            index = random_select_mask + start_index
            ripple_nodes_index.append(index)
            info_of_a_ripple = world_pos_matrix[start_index:end_index]
            info_of_a_ripple = info_of_a_ripple[random_select_mask]
            ripple_nodes_world_pos.append(info_of_a_ripple)
            info_of_a_ripple = mesh_pos_matrix[start_index:end_index]
            info_of_a_ripple = info_of_a_ripple[random_select_mask]
            ripple_nodes_mesh_pos.append(info_of_a_ripple)
        ripple_nodes_feature = torch.cat(ripple_nodes_feature, dim=0)
        ripple_nodes_index = torch.cat(ripple_nodes_index, dim=0)
        ripple_nodes_world_pos = torch.cat(ripple_nodes_world_pos, dim=0)
        ripple_nodes_mesh_pos = torch.cat(ripple_nodes_mesh_pos, dim=0)

        relative_world_pos = torch.sub(ripple_nodes_world_pos, highest_velocity_node_world_pos)
        relative_mesh_pos = torch.sub(ripple_nodes_mesh_pos, highest_velocity_node_mesh_pos)

        ripple_and_highest_info = torch.cat((highest_velocity_node_feature.repeat(ripple_nodes_feature.shape[0],
                                                                                  highest_velocity_node_feature.shape[
                                                                                      0]), ripple_nodes_feature,
                                             relative_world_pos, relative_mesh_pos), dim=-1)
        ripple_and_highest_result = self.ripple_model(ripple_and_highest_info)
        latent_graph.node_features[ripple_nodes_index] += ripple_and_highest_result
        return latent_graph
'''
