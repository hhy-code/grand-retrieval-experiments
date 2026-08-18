from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from grand.data import candidate_ids, graph_map, load_bundle, split_positives
from grand.metrics import ranking_metrics
from grand.models import build_model


def move(graph, device):
    return {
        "id": graph["id"],
        "x": graph["x"].to(device),
        "edge_index": graph["edge_index"].to(device),
        "edge_attr": graph["edge_attr"].to(device),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=("ebp", "mbp"), default="ebp")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device("cuda" if config["training"]["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    bundle = load_bundle(config["data"]["path"])
    graphs = graph_map(bundle)
    candidates = candidate_ids(bundle, "test")
    positives = split_positives(bundle, "test")
    queries = list(positives)
    settings = config["model"]
    name = settings["student"] if args.mode == "ebp" else settings["teacher"]
    model = build_model(name, settings["input_dim"], settings["hidden_dim"], settings["layers"], settings["dropout"], settings["graphsim_cnn_layers"], settings["graphsim_mlp_layers"], settings.get("gem")).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    rankings, times = [], []
    with torch.no_grad():
        if args.mode == "ebp":
            vectors = torch.stack([model.encode(move(graphs[item], device))[1] for item in candidates])
            for query in queries:
                started = time.perf_counter()
                _, vector = model.encode(move(graphs[query], device))
                scores = model.score_embeddings(vectors, vector)
                order = torch.argsort(scores, descending=True).tolist()
                rankings.append([candidates[index] for index in order if candidates[index] != query])
                times.append((time.perf_counter() - started) * 1000)
        else:
            for query in queries:
                started = time.perf_counter()
                scores = [(candidate, model.score_pair(move(graphs[query], device), move(graphs[candidate], device))[0].item()) for candidate in candidates if candidate != query]
                rankings.append([candidate for candidate, _ in sorted(scores, key=lambda item: item[1], reverse=True)])
                times.append((time.perf_counter() - started) * 1000)
    evaluation = config["evaluation"]
    result = ranking_metrics(rankings, positives, queries, evaluation["recall_k"], evaluation["ndcg_k"])
    result.update({"mode": args.mode, "queries": len(queries), "device": str(device), "query_latency_ms_p50": float(np.percentile(times, 50)), "query_latency_ms_p95": float(np.percentile(times, 95))})
    target = Path(args.checkpoint).parent / "metrics_{}.json".format(args.mode)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
