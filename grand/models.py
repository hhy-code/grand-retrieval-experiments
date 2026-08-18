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

    def forward(self, graph):
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
        for _ in range(self.propagation_steps):
            nodes = self.propagation(nodes, edge_index, edge_features)
            nodes = F.dropout(nodes, self.dropout, self.training)
        return nodes, self.aggregator(nodes)


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
    """GEM node encoder plus five CNN/five MLP layers for AIDS."""

    def __init__(self, input_dim, hidden_dim, layers, dropout, cnn_layers=5, mlp_layers=5, gem_config=None):
        super().__init__()
        self.encoder = GEMEncoder(input_dim, gem_config, dropout)
        channels = [1] + [32] * cnn_layers
        self.cnn = nn.ModuleList([nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1) for i in range(cnn_layers)])
        mlp = []
        for index in range(mlp_layers):
            output_dim = 1 if index == mlp_layers - 1 else hidden_dim
            mlp.append(nn.Linear(32 if index == 0 else hidden_dim, output_dim))
            if index != mlp_layers - 1:
                mlp.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp)

    def score_pair(self, left, right):
        left_nodes, _ = self.encoder(left)
        right_nodes, _ = self.encoder(right)
        matrix = (left_nodes @ right_nodes.t()).unsqueeze(0).unsqueeze(0)
        for layer in self.cnn:
            matrix = F.relu(layer(matrix))
        vector = F.adaptive_max_pool2d(matrix, (1, 1)).flatten(1).squeeze(0)
        return self.mlp(vector).squeeze(-1), (left_nodes, right_nodes)


def build_model(name, input_dim, hidden_dim=128, layers=3, dropout=0.0, cnn_layers=5, mlp_layers=5, gem_config=None):
    """Build one architecture named in the GRAND experimental configurations."""
    name = name.lower()
    if name == "gcn":
        return Ebp(GCNEncoder(input_dim, hidden_dim, layers, dropout))
    if name == "gem":
        return Ebp(GEMEncoder(input_dim, gem_config, dropout))
    if name == "gmn":
        return GMN(input_dim, hidden_dim, layers, dropout)
    if name == "graphsim":
        return GraphSim(input_dim, hidden_dim, layers, dropout, cnn_layers, mlp_layers, gem_config)
    raise ValueError("Unknown paper model: {}".format(name))
