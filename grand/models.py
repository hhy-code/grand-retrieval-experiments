"""Model architectures used by the GRAND AIDS experiments.

The module contains the embedding-based GCN/GEM models and the matching-based
GMN/GraphSim models selected by the experiment configuration.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def _aggregate(nodes, edge_index):
    out = torch.zeros_like(nodes)
    degree = torch.zeros(nodes.shape[0], device=nodes.device, dtype=nodes.dtype)
    if edge_index.numel():
        source, target = edge_index
        out.index_add_(0, target, nodes[source])
        degree.index_add_(0, target, torch.ones_like(target, dtype=nodes.dtype))
    return out / degree.clamp_min(1).sqrt().unsqueeze(-1)


class GCNLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, nodes, edge_index):
        return self.linear(_aggregate(nodes, edge_index) + nodes)


class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, dropout):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([GCNLayer(hidden_dim) for _ in range(layers)])
        self.dropout = dropout

    def forward(self, graph):
        nodes = F.relu(self.input(graph["x"]))
        for layer in self.layers:
            nodes = F.dropout(F.relu(layer(nodes, graph["edge_index"])), self.dropout, self.training)
        return nodes, nodes.mean(dim=0)


GEM_DEFAULTS = {
    "node_state_dim": 32,
    "edge_feature_dim": 3,
    "edge_state_dim": 16,
    "message_hidden_dim": 64,
    "propagation_steps": 5,
    "graph_rep_dim": 128,
}


def _gem_config(values):
    config = GEM_DEFAULTS.copy()
    config.update(values or {})
    for key, value in config.items():
        if not isinstance(value, int) or value <= 0:
            raise ValueError("GEM setting '{}' must be a positive integer".format(key))
    return config


def _undirected_edges(edge_index, edge_features):
    """Collapse stored bidirectional edges before GEM propagates both ways."""
    if not edge_index.numel():
        return edge_index, edge_features
    source, target = edge_index
    pairs = torch.stack((torch.minimum(source, target), torch.maximum(source, target)), dim=1)
    pairs, inverse = torch.unique(pairs, dim=0, return_inverse=True)
    merged_features = edge_features.new_zeros((pairs.shape[0], edge_features.shape[1]))
    merged_features.index_add_(0, inverse, edge_features)
    counts = edge_features.new_zeros(pairs.shape[0])
    counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=edge_features.dtype))
    merged_features = merged_features / counts.clamp_min(1).unsqueeze(-1)
    return pairs.t().contiguous(), merged_features


def _graphsim_bfs_order(graph):
    """Return the stable BFS permutation used by the reference GraphSim code.

    The reference implementation starts at the highest-degree node (ties are
    broken by node type and then node id), and sorts each BFS frontier by node
    type and id.  AIDS node features are one-hot atom types, so ``argmax`` is
    the corresponding deterministic type key here.
    """
    node_count = int(graph["x"].shape[0])
    if node_count <= 1:
        return torch.arange(node_count, device=graph["x"].device), torch.arange(node_count, device=graph["x"].device)
    edge_index = graph["edge_index"].detach().cpu()
    neighbors = [[] for _ in range(node_count)]
    for source, target in edge_index.t().tolist():
        source, target = int(source), int(target)
        if 0 <= source < node_count and 0 <= target < node_count and source != target:
            neighbors[source].append(target)
            neighbors[target].append(source)
    node_types = graph["x"].detach().argmax(dim=1).cpu().tolist()
    degree = [len(set(items)) for items in neighbors]
    key = lambda node: (-degree[node], node_types[node], node)
    start = min(range(node_count), key=key)
    order, seen = [], {start}
    queue = [start]
    while queue:
        current = queue.pop(0)
        order.append(current)
        frontier = sorted((node for node in set(neighbors[current]) if node not in seen), key=lambda node: (node_types[node], node))
        seen.update(frontier)
        queue.extend(frontier)
    # The reference data are connected; append disconnected components
    # deterministically instead of failing on an unexpected input graph.
    for component_start in sorted((node for node in range(node_count) if node not in seen), key=key):
        seen.add(component_start)
        queue = [component_start]
        while queue:
            current = queue.pop(0)
            order.append(current)
            frontier = sorted((node for node in set(neighbors[current]) if node not in seen), key=lambda node: (node_types[node], node))
            seen.update(frontier)
            queue.extend(frontier)
    permutation = torch.tensor(order, dtype=torch.long, device=graph["x"].device)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(node_count, device=permutation.device)
    return permutation, inverse


def _reorder_graph(graph, permutation):
    """Reindex node and edge tensors without mutating the dataset bundle."""
    node_count = graph["x"].shape[0]
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(node_count, device=permutation.device)
    edge_index = inverse[graph["edge_index"]]
    return {
        "id": graph.get("id"),
        "x": graph["x"][permutation],
        "edge_index": edge_index,
        "edge_attr": graph["edge_attr"],
    }


class GEMFeatureEncoder(nn.Module):
    """Project raw node and edge features to GEM's recommended state sizes."""

    def __init__(self, node_feature_dim, edge_feature_dim, node_state_dim, edge_state_dim):
        super().__init__()
        self.node = nn.Linear(node_feature_dim, node_state_dim)
        self.edge = nn.Linear(edge_feature_dim, edge_state_dim)

    def forward(self, node_features, edge_features):
        return self.node(node_features), self.edge(edge_features)


class GEMPropagation(nn.Module):
    """One shared, bidirectional GEM propagation layer."""

    def __init__(self, node_state_dim, edge_state_dim, message_hidden_dim):
        super().__init__()
        message_dim = node_state_dim * 2
        self.message = nn.Sequential(
            nn.Linear(node_state_dim * 2 + edge_state_dim, message_hidden_dim),
            nn.ReLU(),
            nn.Linear(message_hidden_dim, message_dim),
        )
        self.update = nn.GRUCell(message_dim, node_state_dim)

    def forward(self, nodes, edge_index, edge_features):
        messages = nodes.new_zeros((nodes.shape[0], self.update.input_size))
        if edge_index.numel():
            source, target = edge_index
            forward_inputs = torch.cat((nodes[source], nodes[target], edge_features), dim=-1)
            reverse_inputs = torch.cat((nodes[target], nodes[source], edge_features), dim=-1)
            messages.index_add_(0, target, self.message(forward_inputs))
            messages.index_add_(0, source, self.message(reverse_inputs))
        return self.update(messages, nodes)


class GEMGraphAggregator(nn.Module):
    """Gated MLP, sum pooling, and graph MLP from the GEM architecture."""

    def __init__(self, node_state_dim, graph_rep_dim):
        super().__init__()
        self.graph_rep_dim = graph_rep_dim
        self.gated_node_transform = nn.Linear(node_state_dim, graph_rep_dim * 2)
        self.graph_transform = nn.Linear(graph_rep_dim, graph_rep_dim)

    def forward(self, nodes):
        gates, values = self.gated_node_transform(nodes).chunk(2, dim=-1)
        return self.graph_transform((torch.sigmoid(gates) * values).sum(dim=0))


class GEMEncoder(nn.Module):
    """GEM graph embedding network adapted from the reference implementation."""

    def __init__(self, input_dim, gem_config=None, dropout=0.0):
        super().__init__()
        config = _gem_config(gem_config)
        self.node_state_dim = config["node_state_dim"]
        self.edge_feature_dim = config["edge_feature_dim"]
        self.propagation_steps = config["propagation_steps"]
        self.feature_encoder = GEMFeatureEncoder(
            input_dim,
            self.edge_feature_dim,
            self.node_state_dim,
            config["edge_state_dim"],
        )
        self.propagation = GEMPropagation(
            self.node_state_dim,
            config["edge_state_dim"],
            config["message_hidden_dim"],
        )
        self.aggregator = GEMGraphAggregator(self.node_state_dim, config["graph_rep_dim"])
        self.dropout = dropout

    def encode_layers(self, graph):
        """Return every GEM node state plus the final graph representation."""
        edge_features = graph.get("edge_attr")
        if edge_features is None:
            raise ValueError("GEM requires edge_attr; rebuild the AIDS bundle with scripts/prepare_aids.py")
        if (
            edge_features.ndim != 2
            or edge_features.shape[0] != graph["edge_index"].shape[1]
            or edge_features.shape[1] != self.edge_feature_dim
        ):
            raise ValueError(
                "GEM expected edge_attr with shape [{}, {}], got {}".format(
                    graph["edge_index"].shape[1], self.edge_feature_dim, tuple(edge_features.shape)
                )
            )
        edge_index, edge_features = _undirected_edges(graph["edge_index"], edge_features)
        nodes, edge_features = self.feature_encoder(graph["x"], edge_features)
        layer_outputs = [nodes]
        for _ in range(self.propagation_steps):
            nodes = self.propagation(nodes, edge_index, edge_features)
            nodes = F.dropout(nodes, self.dropout, self.training)
            layer_outputs.append(nodes)
        return layer_outputs, self.aggregator(nodes)

    def forward(self, graph):
        layer_outputs, embedding = self.encode_layers(graph)
        return layer_outputs[-1], embedding


class Ebp(nn.Module):
    """Embedding-based protocol: encode graphs independently, then compare."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def encode(self, graph):
        return self.encoder(graph)

    @staticmethod
    def score_embeddings(left_embedding, right_embedding):
        """GRAND Eq. 5: negative squared Euclidean distance."""
        return -torch.sum((left_embedding - right_embedding) ** 2, dim=-1)

    def score_pair(self, left, right):
        left_nodes, left_embedding = self.encode(left)
        right_nodes, right_embedding = self.encode(right)
        return self.score_embeddings(left_embedding, right_embedding), (left_nodes, right_nodes)
"""  """

class GMN(nn.Module):
    """Pair-specific matching network with cross-graph attention at each layer."""

    def __init__(self, input_dim, hidden_dim, layers, dropout):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.messages = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(layers)])
        self.updates = nn.ModuleList([nn.GRUCell(hidden_dim * 2, hidden_dim) for _ in range(layers)])
        self.scorer = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.dropout = dropout

    def score_pair(self, left, right):
        left_nodes, right_nodes = F.relu(self.input(left["x"])), F.relu(self.input(right["x"]))
        for message, update in zip(self.messages, self.updates):
            left_messages = _aggregate(F.relu(message(left_nodes)), left["edge_index"])
            right_messages = _aggregate(F.relu(message(right_nodes)), right["edge_index"])
            affinity = left_nodes @ right_nodes.t() / math.sqrt(left_nodes.shape[-1])
            left_cross = F.softmax(affinity, dim=1) @ right_nodes
            right_cross = F.softmax(affinity.t(), dim=1) @ left_nodes
            left_nodes = F.dropout(update(torch.cat((left_messages, left_cross), dim=-1), left_nodes), self.dropout, self.training)
            right_nodes = F.dropout(update(torch.cat((right_messages, right_cross), dim=-1), right_nodes), self.dropout, self.training)
        left_graph, right_graph = left_nodes.mean(dim=0), right_nodes.mean(dim=0)
        score = self.scorer(torch.cat((left_graph, right_graph, (left_graph - right_graph).abs()))).squeeze(-1)
        return score, (left_nodes, right_nodes)


class GraphSim(nn.Module):
    """GEM multi-scale node matching head for the AIDS experiment."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        layers,
        dropout,
        cnn_layers=5,
        mlp_layers=5,
        gem_config=None,
        max_nodes=10,
        scales=3,
        ordering=None,
    ):
        super().__init__()
        if cnn_layers <= 0 or mlp_layers <= 0 or max_nodes <= 0 or scales <= 0:
            raise ValueError("GraphSim layer counts, max_nodes, and scales must be positive")
        if ordering not in (None, "bfs"):
            raise ValueError("GraphSim ordering must be None or 'bfs'")
        self.encoder = GEMEncoder(input_dim, gem_config, dropout)
        self.max_nodes = max_nodes
        self.scales = scales
        self.ordering = ordering
        self._ordering_cache = {}
        channel_schedule = [16, 32, 64, 128, 128]
        if cnn_layers > len(channel_schedule):
            channel_schedule.extend([channel_schedule[-1]] * (cnn_layers - len(channel_schedule)))
        channels = [scales] + channel_schedule[:cnn_layers]
        self.cnn = nn.ModuleList(
            [nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1) for i in range(cnn_layers)]
        )
        self.pool = nn.ModuleList(
            [nn.MaxPool2d(2, stride=2, ceil_mode=True) if i < min(3, cnn_layers) else nn.Identity() for i in range(cnn_layers)]
        )
        mlp = []
        for index in range(mlp_layers):
            output_dim = 1 if index == mlp_layers - 1 else hidden_dim
            mlp.append(nn.Linear(channels[-1] if index == 0 else hidden_dim, output_dim))
            if index != mlp_layers - 1:
                mlp.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp)

    def _pad_similarity(self, left_nodes, right_nodes):
        if left_nodes.shape[0] > self.max_nodes or right_nodes.shape[0] > self.max_nodes:
            raise ValueError("GraphSim received a graph larger than max_nodes={}".format(self.max_nodes))
        matrix = left_nodes.new_zeros((self.max_nodes, self.max_nodes))
        matrix[: left_nodes.shape[0], : right_nodes.shape[0]] = left_nodes @ right_nodes.t()
        return matrix

    def score_pair(self, left, right):
        left_permutation = right_permutation = None
        if self.ordering == "bfs":
            left_permutation = self._cached_order(left)
            right_permutation = self._cached_order(right)
            left = _reorder_graph(left, left_permutation)
            right = _reorder_graph(right, right_permutation)
        left_layers, _ = self.encoder.encode_layers(left)
        right_layers, _ = self.encoder.encode_layers(right)
        left_nodes, right_nodes = left_layers[-1], right_layers[-1]
        if len(left_layers) < self.scales or len(right_layers) < self.scales:
            raise ValueError("GraphSim scales exceed available GEM propagation layers")
        matrices = [
            self._pad_similarity(left_layer, right_layer)
            for left_layer, right_layer in zip(left_layers[-self.scales :], right_layers[-self.scales :])
        ]
        matrix = torch.stack(matrices, dim=0).unsqueeze(0)
        for layer, pool in zip(self.cnn, self.pool):
            matrix = F.relu(layer(matrix))
            matrix = pool(matrix)
        vector = F.adaptive_max_pool2d(matrix, (1, 1)).flatten(1).squeeze(0)
        if self.ordering == "bfs":
            left_nodes = left_nodes[torch.argsort(left_permutation)]
            right_nodes = right_nodes[torch.argsort(right_permutation)]
        return self.mlp(vector).squeeze(-1), (left_nodes, right_nodes)

    def _cached_order(self, graph):
        key = graph.get("id")
        if key is None:
            key = (id(graph), graph["x"].shape[0])
        device_key = str(graph["x"].device)
        cache_key = (int(key) if isinstance(key, (int,)) else key, device_key)
        permutation = self._ordering_cache.get(cache_key)
        if permutation is None:
            permutation, _ = _graphsim_bfs_order(graph)
            self._ordering_cache[cache_key] = permutation
        return permutation


def build_model(
    name,
    input_dim,
    hidden_dim=128,
    layers=3,
    dropout=0.0,
    cnn_layers=5,
    mlp_layers=5,
    gem_config=None,
    graphsim_max_nodes=10,
    graphsim_scales=3,
    graphsim_ordering=None,
):
    """Build one architecture named in the GRAND experimental configurations."""
    name = name.lower()
    if name == "gcn":
        return Ebp(GCNEncoder(input_dim, hidden_dim, layers, dropout))
    if name == "gem":
        return Ebp(GEMEncoder(input_dim, gem_config, dropout))
    if name == "gmn":
        return GMN(input_dim, hidden_dim, layers, dropout)
    if name == "graphsim":
        return GraphSim(
            input_dim,
            hidden_dim,
            layers,
            dropout,
            cnn_layers,
            mlp_layers,
            gem_config,
            graphsim_max_nodes,
            graphsim_scales,
            graphsim_ordering,
        )
    raise ValueError("Unknown paper model: {}".format(name))
