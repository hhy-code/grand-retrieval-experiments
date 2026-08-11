from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Convert normalized JSON graphs to GRAND .pt format")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source = json.loads(Path(args.input).read_text(encoding="utf-8"))
    graphs = [{"id": int(item["id"]), "x": torch.tensor(item["x"], dtype=torch.float32), "edge_index": torch.tensor(item["edge_index"], dtype=torch.long)} for item in source["graphs"]]
    if "splits" in source:
        splits = {name: [int(value) for value in values] for name, values in source["splits"].items()}
    else:
        ids = [item["id"] for item in graphs]
        random.Random(args.seed).shuffle(ids)
        first, second = int(len(ids) * 0.6), int(len(ids) * 0.8)
        splits = {"train": ids[:first], "val": ids[first:second], "test": ids[second:]}
    positives = {int(key): [int(value) for value in values] for key, values in source["positives"].items()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"graphs": graphs, "splits": splits, "positives": positives}, output)
    print("wrote {} graphs to {}".format(len(graphs), output))


if __name__ == "__main__":
    main()
