"""Subgraph partitioning helpers used by GRAND distillation."""
from __future__ import annotations

import math

import networkx as nx
import torch


def fluidc_partition(edge_index, num_nodes, repeats=3):
    """Run FluidC with k approximately sqrt(|V|), retaining best modularity."""
    if num_nodes < 2:
        return [torch.arange(num_nodes, dtype=torch.long)]
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from((int(left), int(right)) for left, right in edge_index.t().tolist())
    if not nx.is_connected(graph):
        return [torch.tensor(sorted(group), dtype=torch.long) for group in nx.connected_components(graph)]
    communities = max(2, min(int(round(math.sqrt(num_nodes))), num_nodes - 1))
    best, best_modularity = None, float("-inf")
    for seed in range(repeats):
        groups = list(nx.community.asyn_fluidc(graph, communities, seed=seed))
        modularity = nx.community.modularity(graph, groups)
        if modularity > best_modularity:
            best, best_modularity = groups, modularity
    return [torch.tensor(sorted(group), dtype=torch.long) for group in best]
