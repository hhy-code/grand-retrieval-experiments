"""Iteration-based AIDS trainer for the paper-style GCN/GEM/GMN/GraphSim models."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

from grand.data import graph_map, load_bundle, sample_triplets
from grand.losses import node_distillation, ranking_loss, score_distillation, subgraph_distillation
from grand.models import build_model
from grand.subgraphs import fluidc_partition


def move(graph, device):
    moved = {"id": graph["id"], "x": graph["x"].to(device), "edge_index": graph["edge_index"].to(device)}
    if "subgraph_groups" in graph:
        moved["subgraph_groups"] = [group.to(device) for group in graph["subgraph_groups"]]
    return moved


def triplet_loss(student, teacher, graphs, triples, device, kd):
    losses = []
    for query_id, positive_id, negative_id in triples:
        query, positive, negative = [move(graphs[item], device) for item in (query_id, positive_id, negative_id)]
        teacher_pos, teacher_pos_nodes = teacher.score_pair(query, positive)
        teacher_neg, teacher_neg_nodes = teacher.score_pair(query, negative)
        student_pos, student_pos_nodes = student.score_pair(query, positive)
        student_neg, student_neg_nodes = student.score_pair(query, negative)
        rank = ranking_loss(student_pos, student_neg)
        loss = kd["alpha"] * rank
        if kd["score"]:
            loss = loss + (1.0 - kd["alpha"]) * score_distillation(student_pos, student_neg, teacher_pos, teacher_neg, kd["temperature"])
        if kd["node"]:
            node_loss = node_distillation(student_pos_nodes[0], student_pos_nodes[1], teacher_pos_nodes[0], teacher_pos_nodes[1], kd["node_temperature"])
            node_loss = node_loss + node_distillation(student_neg_nodes[0], student_neg_nodes[1], teacher_neg_nodes[0], teacher_neg_nodes[1], kd["node_temperature"])
            loss = loss + kd["beta"] * node_loss
        if kd["subgraph"]:
            subgraph_loss = subgraph_distillation(student_pos_nodes[0], student_pos_nodes[1], teacher_pos_nodes[0], teacher_pos_nodes[1], query["edge_index"], positive["edge_index"], kd["subgraph_temperature"], query.get("subgraph_groups"), positive.get("subgraph_groups"))
            subgraph_loss = subgraph_loss + subgraph_distillation(student_neg_nodes[0], student_neg_nodes[1], teacher_neg_nodes[0], teacher_neg_nodes[1], query["edge_index"], negative["edge_index"], kd["subgraph_temperature"], query.get("subgraph_groups"), negative.get("subgraph_groups"))
            loss = loss + kd["gamma"] * subgraph_loss
        losses.append(loss)
    return torch.stack(losses).mean()


def reverse_teacher_loss(student, teacher, graphs, triples, device, kd):
    """Paper Eq. 17: Ebp is the detached target when updating Mbp."""
    losses = []
    for query_id, positive_id, negative_id in triples:
        query, positive, negative = [move(graphs[item], device) for item in (query_id, positive_id, negative_id)]
        teacher_pos, teacher_pos_nodes = teacher.score_pair(query, positive)
        teacher_neg, teacher_neg_nodes = teacher.score_pair(query, negative)
        with torch.no_grad():
            student_pos, student_pos_nodes = student.score_pair(query, positive)
            student_neg, student_neg_nodes = student.score_pair(query, negative)
        rank = ranking_loss(teacher_pos, teacher_neg)
        loss = kd["alpha"] * rank
        if kd["score"]:
            loss = loss + (1.0 - kd["alpha"]) * score_distillation(teacher_pos, teacher_neg, student_pos, student_neg, kd["temperature"])
        if kd["node"]:
            node_loss = node_distillation(teacher_pos_nodes[0], teacher_pos_nodes[1], student_pos_nodes[0], student_pos_nodes[1], kd["node_temperature"])
            node_loss = node_loss + node_distillation(teacher_neg_nodes[0], teacher_neg_nodes[1], student_neg_nodes[0], student_neg_nodes[1], kd["node_temperature"])
            loss = loss + kd["beta"] * node_loss
        if kd["subgraph"]:
            subgraph_loss = subgraph_distillation(teacher_pos_nodes[0], teacher_pos_nodes[1], student_pos_nodes[0], student_pos_nodes[1], query["edge_index"], positive["edge_index"], kd["subgraph_temperature"], query.get("subgraph_groups"), positive.get("subgraph_groups"))
            subgraph_loss = subgraph_loss + subgraph_distillation(teacher_neg_nodes[0], teacher_neg_nodes[1], student_neg_nodes[0], student_neg_nodes[1], query["edge_index"], negative["edge_index"], kd["subgraph_temperature"], query.get("subgraph_groups"), negative.get("subgraph_groups"))
            loss = loss + kd["gamma"] * subgraph_loss
        losses.append(loss)
    return torch.stack(losses).mean()


def baseline_loss(model, graphs, triples, device):
    losses = []
    for query_id, positive_id, negative_id in triples:
        query, positive, negative = [move(graphs[item], device) for item in (query_id, positive_id, negative_id)]
        positive_score, _ = model.score_pair(query, positive)
        negative_score, _ = model.score_pair(query, negative)
        losses.append(ranking_loss(positive_score, negative_score))
    return torch.stack(losses).mean()


def train_baseline(model, graphs, bundle, device, train, seed, output):
    """Train one Ebp/Mbp baseline with the paper's ranking objective."""
    optimizer = torch.optim.Adam(model.parameters(), lr=train["learning_rate"])
    best, stale, history = float("inf"), 0, []
    bar = tqdm(range(1, train["iterations"] + 1), desc="Baseline training", unit="iter")
    for iteration in bar:
        model.train()
        triples = list(sample_triplets(bundle, "train", train["batch_size"], random.Random(seed + iteration)))
        loss = baseline_loss(model, graphs, triples, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bar.set_postfix(train="{:.4f}".format(loss.item()))
        if iteration % train["validation_interval"] != 0:
            continue
        value = validate(model, graphs, bundle, device, seed + iteration)
        history.append({"iteration": iteration, "train_loss": loss.item(), "validation_loss": value})
        bar.set_postfix(train="{:.4f}".format(loss.item()), validation="{:.4f}".format(value))
        if value < best:
            best, stale = value, 0
            torch.save(model.state_dict(), output / "student_best.pt")
        else:
            stale += 1
            if stale >= train["early_stopping_patience"]:
                print("early stopping at iteration={}".format(iteration))
                break
    if not (output / "student_best.pt").is_file():
        torch.save(model.state_dict(), output / "student_best.pt")
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("Training finished. Best baseline checkpoint: {}".format(output / "student_best.pt"))


def validate(model, graphs, bundle, device, seed, count=320):
    model.eval()
    with torch.no_grad():
        triples = list(sample_triplets(bundle, "val", count, random.Random(seed)))
        return baseline_loss(model, graphs, triples, device).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if config["training"]["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    bundle = load_bundle(config["data"]["path"])
    graphs = graph_map(bundle)
    distillation = config.get("distillation", {})
    if distillation.get("subgraph", False):
        for graph in graphs.values():
            graph["subgraph_groups"] = fluidc_partition(graph["edge_index"], graph["x"].shape[0])
    model = config["model"]
    gem_config = model.get("gem")
    student = build_model(model["student"], model["input_dim"], model["hidden_dim"], model["layers"], model["dropout"], model["graphsim_cnn_layers"], model["graphsim_mlp_layers"], gem_config).to(device)
    output = Path("outputs") / config["experiment"]["name"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    train = config["training"]
    if config["experiment"].get("mode", "grand") == "baseline":
        train_baseline(student, graphs, bundle, device, train, seed, output)
        return

    teacher = build_model(model["teacher"], model["input_dim"], model["hidden_dim"], model["layers"], model["dropout"], model["graphsim_cnn_layers"], model["graphsim_mlp_layers"], gem_config).to(device)
    kd = distillation
    teacher_optimizer = torch.optim.Adam(teacher.parameters(), lr=train["learning_rate"])
    student_optimizer = torch.optim.Adam(student.parameters(), lr=train["learning_rate"])
    bidirectional = bool(kd.get("bidirectional", False))
    if not bidirectional:
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        teacher.eval()

    best, stale, history = float("inf"), 0, []
    student_bar = tqdm(range(1, train["iterations"] + 1), desc="Mutual KD", unit="iter")
    for iteration in student_bar:
        student.train()
        triples = list(sample_triplets(bundle, "train", train["batch_size"], random.Random(seed + 100000 + iteration)))
        loss = triplet_loss(student, teacher, graphs, triples, device, kd)
        student_optimizer.zero_grad()
        loss.backward()
        student_optimizer.step()
        if bidirectional:
            teacher.train()
            teacher_loss = reverse_teacher_loss(student, teacher, graphs, triples, device, kd)
            teacher_optimizer.zero_grad()
            teacher_loss.backward()
            teacher_optimizer.step()
        student_bar.set_postfix(train="{:.4f}".format(loss.item()))
        if iteration % train["validation_interval"] != 0:
            continue
        value = validate(student, graphs, bundle, device, seed + iteration)
        history.append({"iteration": iteration, "train_loss": loss.item(), "validation_loss": value})
        student_bar.set_postfix(train="{:.4f}".format(loss.item()), validation="{:.4f}".format(value))
        if value < best:
            best, stale = value, 0
            torch.save(student.state_dict(), output / "student_best.pt")
        else:
            stale += 1
            if stale >= train["early_stopping_patience"]:
                print("early stopping at iteration={}".format(iteration))
                break
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    torch.save(teacher.state_dict(), output / "teacher_final.pt")
    print("Training finished. Best student checkpoint: {}".format(output / "student_best.pt"))


if __name__ == "__main__":
    main()
