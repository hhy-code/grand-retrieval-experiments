from __future__ import annotations

import argparse
import io
import pickle
import zipfile
from pathlib import Path

import networkx as nx
import torch


def load_graphs(zip_path):
    graphs, raw_splits = [], {"train": [], "test": []}
    with zipfile.ZipFile(zip_path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".gexf")]
        for name in names:
            parts = name.split("/")
            split = parts[1].lower()
            identifier = int(Path(parts[-1]).stem)
            graph = nx.read_gexf(io.BytesIO(archive.read(name)), node_type=str)
            graphs.append((identifier, split, graph))
            raw_splits.setdefault(split, []).append(identifier)
    if len(graphs) != 700 or len(raw_splits.get("train", [])) != 560 or len(raw_splits.get("test", [])) != 140:
        raise ValueError("Expected 700 graphs split as 560 train and 140 test in AIDS700nef")
    return graphs, raw_splits


def atom_type_mapping(records, feature_dim=29):
    """Build one stable atom-type vocabulary across the complete dataset."""
    atom_types = sorted(
        {
            str(graph.nodes[node]["type"])
            for _, _, graph in records
            for node in graph.nodes()
        }
    )
    if len(atom_types) != feature_dim:
        raise ValueError(
            "Expected {} AIDS atom types, found {}: {}".format(
                feature_dim, len(atom_types), atom_types
            )
        )
    return {atom_type: index for index, atom_type in enumerate(atom_types)}


def graph_tensor(graph, atom_types):
    nodes = list(graph.nodes())
    position = {node: index for index, node in enumerate(nodes)}
    features = torch.zeros(len(nodes), len(atom_types), dtype=torch.float32)
    for index, node in enumerate(nodes):
        atom_type = str(graph.nodes[node].get("type", ""))
        if atom_type not in atom_types:
            raise ValueError("Unexpected AIDS atom type: {!r}".format(atom_type))
        features[index, atom_types[atom_type]] = 1.0
    edges = []
    for source, target in graph.edges():
        left, right = position[source], position[target]
        edges.extend(((left, right), (right, left)))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    return features, edge_index


def distance_map(path):
    with open(path, "rb") as handle:
        distances = pickle.load(handle)
    return {(int(left), int(right)): float(value) for (left, right), value in distances.items()}


def build_positives(distances, query_ids, candidate_ids):
    """Return every candidate tied for a query's minimum GED.

    GRAND Section 5.1.1 defines a similar AIDS graph as one with the minimum
    GED to the query. This is deliberately not a GED top-k conversion.
    """
    result = {}
    for query in query_ids:
        pairs = []
        for candidate in candidate_ids:
            if candidate == query:
                continue
            value = distances.get((query, candidate), distances.get((candidate, query)))
            if value is not None:
                pairs.append((candidate, value))
        if not pairs:
            raise ValueError("No GED entries found for query {}".format(query))
        threshold = min(value for _, value in pairs)
        result[query] = [candidate for candidate, value in pairs if value <= threshold]
    return result


def main():
    parser = argparse.ArgumentParser(description="Convert GraphSim AIDS700nef files to GRAND format")
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--ged", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records, raw_splits = load_graphs(args.graphs)
    atom_types = atom_type_mapping(records)
    raw_train = sorted(raw_splits["train"])
    # GraphSim uses the sorted original training list and holds out its last 25%.
    train_ids, val_ids = raw_train[:420], raw_train[420:]
    test_ids = sorted(raw_splits["test"])
    candidate_train = sorted(train_ids)
    candidate_test = raw_train
    distances = distance_map(args.ged)
    positives = {}
    positives.update(build_positives(distances, train_ids, candidate_train))
    positives.update(build_positives(distances, val_ids, candidate_train))
    positives.update(build_positives(distances, test_ids, candidate_test))
    graphs = []
    for identifier, _, graph in records:
        features, edge_index = graph_tensor(graph, atom_types)
        graphs.append({"id": identifier, "x": features, "edge_index": edge_index})
    bundle = {"graphs": graphs, "splits": {"train": train_ids, "val": val_ids, "test": test_ids}, "candidates": {"train": candidate_train, "val": candidate_train, "test": candidate_test}, "positives": positives, "metadata": {"dataset": "AIDS700nef", "feature_dim": 29, "atom_type_mapping": atom_types, "relevance_definition": "all candidates with minimum A* GED to each query", "split_protocol": "GraphSim sorted 75/25 train/validation; test queries retrieve from all 560 original train graphs", "source": "GraphSim AIDS700nef + A* GED pickle"}}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output)
    print("wrote {} graphs to {}".format(len(graphs), output))
    print("query_splits=train:{} val:{} test:{} candidates=train/val:{} test:{} relevance=minimum_ged".format(len(train_ids), len(val_ids), len(test_ids), len(candidate_train), len(candidate_test)))


if __name__ == "__main__":
    main()
