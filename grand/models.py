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


def _undirected_edges(edge_index):
    """Match GEM's reverse-direction propagation for an undirected AIDS graph."""
    if not edge_index.numel():
        return edge_index
    source, target = edge_index
    pairs = torch.stack((torch.minimum(source, target), torch.maximum(source, target)), dim=1)
    return torch.unique(pairs, dim=0).t().contiguous()


class GEMPropagation(nn.Module):
    """One shared GEM propagation layer based on Graph Matching Networks."""

    def __init__(self, node_state_dim, message_hidden_dim):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(node_state_dim * 2, message_hidden_dim),
            nn.ReLU(),
            nn.Linear(message_hidden_dim, message_hidden_dim),
        )
        self.update = nn.GRUCell(message_hidden_dim, node_state_dim)

    def forward(self, nodes, edge_index):
        messages = nodes.new_zeros((nodes.shape[0], self.update.input_size))
        if edge_index.numel():
            source, target = edge_index
            messages.index_add_(0, target, self.message(torch.cat((nodes[source], nodes[target]), dim=-1)))
            messages.index_add_(0, source, self.message(torch.cat((nodes[target], nodes[source]), dim=-1)))
        return self.update(messages, nodes)


class GEMEncoder(nn.Module):
    """GEM node encoder with gated-sum graph aggregation."""

    def __init__(self, input_dim, gem_config=None, dropout=0.0):
        super().__init__()
        config = _gem_config(gem_config)
        self.node_state_dim = config["node_state_dim"]
        self.propagation_steps = config["propagation_steps"]
        self.input = nn.Linear(input_dim, self.node_state_dim)
        self.propagation = GEMPropagation(self.node_state_dim, config["message_hidden_dim"])
        self.gated_sum = nn.Linear(self.node_state_dim, config["graph_rep_dim"] * 2)
        self.graph_transform = nn.Linear(config["graph_rep_dim"], config["graph_rep_dim"])
        self.dropout = dropout

    def forward(self, graph):
        nodes = self.input(graph["x"])
        edge_index = _undirected_edges(graph["edge_index"])
        for _ in range(self.propagation_steps):
            nodes = self.propagation(nodes, edge_index)
            nodes = F.dropout(nodes, self.dropout, self.training)
        pooled = self.gated_sum(nodes)
        gates, values = pooled.chunk(2, dim=-1)
        embedding = self.graph_transform((torch.sigmoid(gates) * values).sum(dim=0))
        return nodes, embedding


class Ebp(nn.Module):
    """Embedding-based protocol: encode graphs independently, then compare."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def encode(self, graph):
        nodes, embedding = self.encoder(graph)
        return nodes, F.normalize(embedding, dim=0)

    def score_pair(self, left, right):
        left_nodes, left_embedding = self.encode(left)
        right_nodes, right_embedding = self.encode(right)
        return (left_embedding * right_embedding).sum(), (left_nodes, right_nodes)


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
