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

Each command trains with its YAML configuration, restores the best Ebp or Mbp
checkpoint selected on the validation split, and writes test metrics under
`outputs/<experiment-name>/`.

## Reproducibility Scope

The repository follows the data split, losses, optimization settings, and model
layer counts stated in GRAND. Some GraphSim CNN details and the adaptation of
GEM intermediate representations are not specified by the paper; choices made
by this implementation are documented in the source and configuration files.
