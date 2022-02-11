import torch
import collections
import find_influential_nodes

device = torch.device('cuda')

EdgeSet = collections.namedtuple('EdgeSet', ['name', 'features', 'senders',
                                             'receivers'])
MultiGraph = collections.namedtuple('Graph', ['node_features', 'edge_sets'])

MultiGraphWithPos = collections.namedtuple('Graph', ['node_features', 'edge_sets', 'target_feature', 'model_type', 'node_dynamic'])

# aggregate nodes into ripples
# returns node index in each ripple
class RippleGenerator():

    def __init__(self, ripple_generation, ripple_generation_number):
        self._ripple_generation_method = ripple_generation
        self._ripple_generation_number = ripple_generation_number

    def generate_ripple(self, graph):
        ripple_indices = []
        is_gradient = False
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
            return (ripple_indices, None, is_gradient)
        elif self._ripple_generation_method == 'gradient':
            # bins should be set as small as possible to ensure the nodes inside a bin has the greatest similarity and
            # as big as possible to ensure the similar nodes are assign to same group
            is_gradient = True
            target_feature_matrix = graph.node_dynamic
            num_nodes = target_feature_matrix.shape[0]
            bins = 100
            take_n_bins = self._ripple_generation_number - 1
            # velocity_matrix = graph.node_features[:, 0:3]
            # norm = torch.linalg.vector_norm(velocity_matrix, dim=1)
            histogram = torch.histc(target_feature_matrix, bins=bins)
            values, indices = torch.topk(histogram, take_n_bins)
            for i in range(take_n_bins):
                start_index = torch.sum(histogram[:indices[i]]).to(torch.int32)
                end_index = start_index + values[i]
                ripple_indices.append((start_index.item(), end_index.to(torch.int32).item()))
                ripple_indices.sort(key=lambda x: x[0])
            selected_nodes_concat = []
            for start_index, end_index in ripple_indices:
                selected_nodes = list(range(start_index, end_index))
                selected_nodes_concat.append(selected_nodes)
            flattened_list = [item for sublist in selected_nodes_concat for item in sublist]
            rest_nodes = list(range(0, num_nodes))
            rest_nodes = list(set(rest_nodes) - set(flattened_list))
            return (ripple_indices, rest_nodes, is_gradient)
        elif self._ripple_generation_method == 'exponential_size':
            is_gradient = False
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
                    return (ripple_indices, None, is_gradient)
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
        is_gradient = ripple_tuple[2]
        if self._ripple_node_selection == 'random':
            for ripple in ripple_list:
                ripple_size = ripple[1] - ripple[0]
                ripple_select_size = self._ripple_node_selection_random_top_n if self._ripple_node_selection_random_top_n <= ripple_size else ripple_size
                random_select_mask = torch.randperm(n=ripple_size)[0:ripple_select_size]
                selected_nodes.append(random_select_mask)
            if is_gradient:
                ripple_size = len(ripple_rest)
                ripple_select_size = self._ripple_node_selection_random_top_n if self._ripple_node_selection_random_top_n <= ripple_size else ripple_size
                random_select_mask = torch.randperm(n=ripple_size)[0:ripple_select_size]
                selected_nodes.append(random_select_mask)
        elif self._ripple_node_selection == 'top':
            for ripple in ripple_list:
                ripple_size = ripple[1] - ripple[0]
                ripple_select_size = self._ripple_node_selection_random_top_n if self._ripple_node_selection_random_top_n <= ripple_size else ripple_size
                selected_nodes.append(range(ripple_size)[:ripple_select_size])
            if is_gradient:
                ripple_size = len(ripple_rest)
                ripple_select_size = self._ripple_node_selection_random_top_n if self._ripple_node_selection_random_top_n <= ripple_size else ripple_size
                selected_nodes.append(range(ripple_size)[:ripple_select_size])
        elif self._ripple_node_selection == 'all':
            for ripple in ripple_list:
                ripple_size = ripple[1] - ripple[0]
                selected_nodes.append(range(ripple_size))
            if is_gradient:
                ripple_size = len(ripple_rest)
                selected_nodes.append(range(ripple_size))
        return selected_nodes

# connect the selected nodes
class RippleNodeConnector():
    def __init__(self, ripple_node_connection, ripple_node_ncross):
        self._ripple_node_connection = ripple_node_connection
        self._ripple_node_ncross = ripple_node_ncross

    def connect(self, graph, ripple_tuple, node_selections, world_edge_normalizer, is_training):
        model_type = graph.model_type
        node_dynamic = graph.node_dynamic

        _, sort_indices = torch.sort(node_dynamic, dim=0, descending=True)

        selected_nodes = []
        ripples = ripple_tuple[0]
        ripple_rest = ripple_tuple[1]
        for (start_index, end_index), node_mask in zip(ripples, node_selections):
            if end_index > start_index:
                ripple = sort_indices[start_index:end_index]
                selected_nodes.append(ripple[node_mask])
        if ripple_rest is not None:
            ripple = sort_indices[list(ripple_rest)]
            selected_nodes.append(ripple[node_selections[-1]])

        ripple_edges = []
        if self._ripple_node_connection == 'most_influential':
            target_feature = graph.target_feature
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
                edge_features = torch.cat((relative_target_feature, torch.norm(relative_target_feature, dim=-1, keepdim=True)), dim=-1)
            else:
                raise Exception("Model type is not specified in RippleNodeConnector.")
            edge_features = world_edge_normalizer(edge_features)

            world_edges = EdgeSet(
                name='ripple_edges',
                features=world_edge_normalizer(edge_features, None, is_training),
                receivers=receivers,
                senders=senders)

            ripple_edges.append(world_edges)

        elif self._ripple_node_connection == 'fully_connected':
            target_feature = graph.target_feature
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
                    edge_features = torch.cat(
                        (relative_target_feature, torch.norm(relative_target_feature, dim=-1, keepdim=True)), dim=-1)
                else:
                    raise Exception("Model type is not specified in RippleNodeConnector.")
                edge_features = world_edge_normalizer(edge_features)

                world_edges = EdgeSet(
                    name='ripple_edges',
                    features=world_edge_normalizer(edge_features, None, is_training),
                    receivers=receivers,
                    senders=senders)

                ripple_edges.append(world_edges)

        elif self._ripple_node_connection == 'fully_ncross_connected':
            target_feature = graph.target_feature
            cross_nodes = []
            for ripple_selected_nodes in selected_nodes:
                if len(ripple_selected_nodes) == 0:
                    world_edges = EdgeSet(
                        name='ripple_edges',
                        features=[],
                        receivers=[],
                        senders=[])
                    ripple_edges.append(world_edges)
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
                    edge_features = torch.cat(
                        (relative_target_feature, torch.norm(relative_target_feature, dim=-1, keepdim=True)), dim=-1)
                else:
                    raise Exception("Model type is not specified in RippleNodeConnector.")
                edge_features = world_edge_normalizer(edge_features)

                world_edges = EdgeSet(
                    name='ripple_edges',
                    features=world_edge_normalizer(edge_features, None, is_training),
                    receivers=receivers,
                    senders=senders)

                ripple_edges.append(world_edges)

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
                edge_features = torch.cat(
                    (relative_target_feature, torch.norm(relative_target_feature, dim=-1, keepdim=True)), dim=-1)
            else:
                raise Exception("Model type is not specified in RippleNodeConnector.")
            edge_features = world_edge_normalizer(edge_features)

            world_edges = EdgeSet(
                name='ripple_edges',
                features=world_edge_normalizer(edge_features, None, is_training),
                receivers=receivers,
                senders=senders)

            ripple_edges.append(world_edges)

        edge_sets = graph.edge_sets
        edge_sets.extend(ripple_edges)

        return MultiGraphWithPos(node_features=graph.node_features,
                                 edge_sets=edge_sets, target_feature=graph.target_feature,
                                 model_type=graph.model_type, node_dynamic=graph.node_dynamic)

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

    def add_meta_edges(self, graph, world_edge_normalizer, is_training):
        if self._ripple_generation == 'random_nodes' or self._ripple_generation == 'distance_density':
            target_feature = graph.target_feature
            selected_nodes = None
            if self._ripple_generation == 'random_nodes':
                selected_nodes = torch.randperm(n=target_feature.shape[0])[0:self._ripple_generation_number]
            if self._ripple_generation == 'distance_density':
                selected_nodes = find_influential_nodes.find_influential_nodes(target_feature, self._radius, self._topk)
            reverse_selected_nodes = torch.flip(selected_nodes, [-1])
            edges = torch.cat((torch.combinations(selected_nodes, with_replacement=True), torch.combinations(reverse_selected_nodes, with_replacement=True)), dim=0)
            senders, receivers = torch.unbind(edges, dim=-1)

            model_type = graph.model_type
            if model_type == 'cloth_model' or model_type == 'deform_model':
                relative_world_pos = (torch.index_select(input=target_feature.to(device), dim=0, index=senders.to(device)) -
                                           torch.index_select(input=target_feature.to(device), dim=0, index=receivers.to(device)))
                world_edge_features = torch.cat((
                    relative_world_pos,
                    torch.norm(relative_world_pos, dim=-1, keepdim=True)), dim=-1)
            else:
                raise Exception("Model type is not specified in RippleNodeConnector.")
            world_edges = EdgeSet(
                name='ripple_edges',
                features=world_edge_normalizer(world_edge_features, None, is_training),
                receivers=receivers,
                senders=senders)

            edge_sets = graph.edge_sets
            edge_sets.append(world_edges)
            return MultiGraphWithPos(node_features=graph.node_features,
                              edge_sets=edge_sets, target_feature=graph.target_feature,
                              model_type=graph.model_type, node_dynamic=graph.node_dynamic)
        else:
            ripple_indices = self._ripple_generator.generate_ripple(graph)
            selected_nodes = self._ripple_node_selector.select_nodes(ripple_indices)
            new_graph = self._ripple_node_connector.connect(graph, ripple_indices, selected_nodes, world_edge_normalizer, is_training)
        return new_graph
