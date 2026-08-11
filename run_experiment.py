"""Run one configured dataset experiment by its dataset name."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def find_config(method, dataset):
    method = method.lower().strip().replace("-", "_")
    normalized = dataset.lower().strip()
    if method == "grand":
        raise SystemExit(
            "GRAND needs an explicit Ebp/Mbp pair. Choose GRAND-GCN-GMN, "
            "GRAND-GCN-GraphSim, GRAND-GEM-GMN, or GRAND-GEM-GraphSim."
        )
    matches = sorted(Path("configs").glob("{}_{}.yaml".format(normalized, method)))
    if not matches:
        available = sorted({path.name.split("_")[0] for path in Path("configs").glob("*.yaml")})
        raise SystemExit("No configuration for '{}'. Available datasets: {}".format(dataset, ", ".join(available)))
    if len(matches) > 1:
        raise SystemExit("Multiple configurations for '{}': {}. Use --config to choose one.".format(dataset, ", ".join(str(path) for path in matches)))
    return matches[0]


def run(command):
    print("\n>> {}\n".format(" ".join(str(item) for item in command)), flush=True)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description="Train then evaluate a dataset experiment")
    parser.add_argument("method", nargs="?", help="Method: GCN, GEM, GMN, GraphSim, or an explicit GRAND pair")
    parser.add_argument("dataset", nargs="?", help="Dataset name, e.g. AIDS")
    parser.add_argument("--config", help="Explicit configuration file; overrides dataset lookup")
    args = parser.parse_args()
    if not args.dataset and not args.config:
        parser.error("provide a method and dataset, for example: python train GMN AIDS")
    config_path = Path(args.config) if args.config else find_config(args.method, args.dataset)
    if not config_path.is_file():
        raise SystemExit("Configuration not found: {}".format(config_path))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = Path("outputs") / config["experiment"]["name"]
    checkpoint = output / "student_best.pt"
    try:
        run([sys.executable, "train_paper_aids.py", "--config", str(config_path)])
    except subprocess.CalledProcessError as error:
        raise SystemExit("Training stopped or failed (exit code {}). Final evaluation was not started.".format(error.returncode))
    if not checkpoint.is_file():
        raise SystemExit("Training finished without {}. Final evaluation was not started.".format(checkpoint))
    evaluation_mode = config.get("evaluation", {}).get("mode", "ebp")
    run([sys.executable, "evaluate_paper_aids.py", "--config", str(config_path), "--checkpoint", str(checkpoint), "--mode", evaluation_mode])
    print("\nExperiment complete. Results: {}".format(output / "metrics_{}.json".format(evaluation_mode)))


if __name__ == "__main__":
    main()
