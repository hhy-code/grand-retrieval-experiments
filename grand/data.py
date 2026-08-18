from __future__ import annotations

import random

import torch


def load_bundle(path):
    bundle = torch.load(path, map_location="cpu")
    missing = {"graphs", "splits", "positives"}.difference(bundle)
    if missing:
        raise ValueError("Dataset is missing keys: {}".format(sorted(missing)))
    for graph in bundle["graphs"]:
        if not {"id", "x", "edge_index", "edge_attr"}.issubset(graph):
            raise ValueError("Every graph requires id, x, edge_index, and edge_attr; rebuild the AIDS bundle")
        graph["x"] = torch.as_tensor(graph["x"], dtype=torch.float32)
        graph["edge_index"] = torch.as_tensor(graph["edge_index"], dtype=torch.long)
        graph["edge_attr"] = torch.as_tensor(graph["edge_attr"], dtype=torch.float32)
        if graph["edge_index"].ndim != 2 or graph["edge_index"].shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if graph["edge_attr"].ndim != 2 or graph["edge_attr"].shape[0] != graph["edge_index"].shape[1]:
            raise ValueError("edge_attr must have shape [num_edges, edge_feature_dim]")
    for split in ("train", "val", "test"):
        if split not in bundle["splits"]:
            raise ValueError("Missing split: {}".format(split))
    return bundle


def graph_map(bundle):
    return {int(graph["id"]): graph for graph in bundle["graphs"]}


def candidate_ids(bundle, split):
    candidates = bundle.get("candidates", {})
    return [int(item) for item in candidates.get(split, bundle["splits"][split])]


def split_positives(bundle, split):
    candidate_set = set(candidate_ids(bundle, split))
    queries = [int(item) for item in bundle["splits"][split]]
    result = {}
    positives = bundle["positives"]
    for query in queries:
        raw = positives.get(query, positives.get(str(query), []))
        relevant = [int(item) for item in raw if int(item) in candidate_set and int(item) != query]
        if relevant:
            result[query] = relevant
    return result


def sample_triplets(bundle, split, count, rng):
    ids = candidate_ids(bundle, split)
    positives = split_positives(bundle, split)
    queries = list(positives)
    if not queries:
        raise ValueError("No queries with positive examples in '{}' split".format(split))
    for _ in range(count):
        query = rng.choice(queries)
        positive = rng.choice(positives[query])
        negative = rng.choice(ids)
        while negative == query or negative in positives[query]:
            negative = rng.choice(ids)
        yield query, positive, negative
