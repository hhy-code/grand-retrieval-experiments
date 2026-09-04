# GRAND Retrieval Experiments

PyTorch experiments for the AIDS700nef setting described in GRAND. The
repository contains standalone GCN, GEM, GMN, and GraphSim baselines together
with the four configured GRAND Ebp/Mbp combinations.

## Project Layout

```text
configs/                 Experiment configurations
grand/models.py          GCN, GEM, GMN, GraphSim, and Ebp architectures
grand/losses.py          Ranking and knowledge-distillation objectives
grand/subgraphs.py       FluidC subgraph partitioning
grand/data.py            Dataset loading and triplet sampling
grand/metrics.py         NDCG@5 and Recall@5
scripts/prepare_aids.py  AIDS700nef preprocessing
train_paper_aids.py      Training and early stopping
evaluate_paper_aids.py   Retrieval evaluation
run_experiment.py        Train-then-evaluate command dispatcher
```

## Environment

Create and activate a Python 3.13 virtual environment, then install the pinned
dependencies:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Data

The training configurations expect the prepared bundle at:

```text
data/processed/aids700nef.pt
```

Raw datasets and third-party source repositories are not project code and
should not be committed. Use `scripts/prepare_aids.py --help` for the required
preprocessing arguments, and document the source URLs and revisions when
publishing results.

Prepare the checked-in experiment schema from the local GraphSim data files:

```powershell
python scripts/prepare_aids.py `
  --graphs data/raw/aids/AIDS700nef.zip `
  --ged data/raw/aids/aids700nef_ged_astar_gidpair_dist_map.pickle `
  --output data/processed/aids700nef.pt
```

The prepared graphs contain 29-dimensional one-hot atom features and
3-dimensional one-hot bond-valence features for valences 1, 2, and 3.

## GEM Architecture

The GEM implementation follows the architecture recommended by the reference
Graph Matching Networks implementation: node and edge feature encoders, five
shared bidirectional message-passing steps with GRU node updates, and a gated
MLP followed by sum pooling and a graph MLP. The configured dimensions are 32
for node states, 16 for edge states, 64 for messages, and 128 for graph
representations. Ebp scores graph pairs with the negative squared Euclidean
distance specified by GRAND Equation 5. GraphSim reuses the same GEM node
encoder as required by GRAND Section 5.1.3.

For the AIDS GraphSim adaptation, the final three GEM propagation states are
stacked as a three-channel node-similarity tensor, padded to the dataset's
maximum of 10 nodes, and passed through five convolutional layers with
progressively wider channels and intermediate pooling, followed by a
five-linear-layer MLP. The paper specifies the 5+5 depth but does not publish
the complete GraphSim kernel and pooling details; these choices are explicit in
`grand/models.py` and the configuration.

The GraphSim configurations enable `graphsim_ordering: bfs`. This follows the
optional stable BFS ordering implemented by the reference GraphSim repository:
the highest-degree node is used as the start node, and each frontier is sorted
by atom type and node index. The permutation is applied only while constructing
the GraphSim matching tensor; node states returned for GRAND distillation are
restored to the original indexing. Set this option to `null` for the
non-ordered ablation. GRAND does not explicitly state whether BFS ordering was
used in its AIDS experiment, so this setting is recorded as an implementation
choice rather than a paper-confirmed fact.

GRAND does not explicitly state whether AIDS bond valence is passed to GEM.
This implementation uses it because the reference GEM encoder accepts edge
features and the supplied AIDS graphs contain complete valence labels. The
choice is explicit in the data metadata and YAML settings so it can be ablated
without changing the dataset source.

Checkpoints created before the edge encoder and Euclidean scoring change are
not compatible with the current model. Rebuild the processed data and retrain
instead of loading those checkpoints.

## Baselines

```powershell
python train GCN AIDS
python train GEM AIDS
python train GMN AIDS
python train GraphSim AIDS
```

## GRAND

```powershell
python train GRAND-GCN-GMN AIDS
python train GRAND-GCN-GraphSim AIDS
python train GRAND-GEM-GMN AIDS
python train GRAND-GEM-GraphSim AIDS
```

### GRAND hyperparameter validation

These single-seed configurations keep the data split, validation triplets, BFS
ordering, temperatures, and optimizer fixed. The first three compare `alpha`
with `beta=gamma=0.1`; the final two isolate a smaller node- or subgraph-level
weight.

```powershell
python train --config configs/aids_grand_gem_graphsim_a025_b01_g01.yaml
python train --config configs/aids_grand_gem_graphsim_a050_b01_g01.yaml
python train --config configs/aids_grand_gem_graphsim_a075_b01_g01.yaml已有结果
python train --config configs/aids_grand_gem_graphsim_a075_b001_g01.yaml
python train --config configs/aids_grand_gem_graphsim_a075_b01_g001.yaml
```

Suffixes encode the values: `a025` is `alpha=0.25`, `b01` is `beta=0.1`,
and `g01` is `gamma=0.1`; `b001` and `g001` mean `0.01`. Choose the setting
with the lowest `validation_loss` in its `history.json`; use its test
`metrics_ebp.json` only for final reporting.

Each command trains with its YAML configuration, restores the best Ebp or Mbp
checkpoint selected on the validation split, and writes test metrics under
`outputs/<experiment-name>_<YYYYMMDD_HHMMSS_microseconds>/`. Every invocation
gets a new timestamped directory, so runs never overwrite one another. Each
run directory contains its configuration snapshot, `run.json`, training
history, checkpoints, and final metrics.

## Reproducibility Scope

The repository follows the data split, losses, optimization settings, and model
layer counts stated in GRAND. Some GraphSim CNN details and the adaptation of
GEM intermediate representations are not specified by the paper; choices made
by this implementation are documented in the source and configuration files.
Validation triplets are sampled once from a fixed seed per run and reused for
early stopping, so checkpoints are compared on the same validation examples.
