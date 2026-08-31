# GRAND 实验运行指南

> 本文重点说明一次实验启动后，程序如何在各文件之间流转；命令只是触发入口。

本项目复现 GRAND 在 AIDS700nef 图检索任务上的实验。支持四个单独基线：GCN、GEM、GMN、GraphSim，以及四个 GRAND 组合：GCN/GEM 作为 Ebp，GMN/GraphSim 作为 Mbp。

## 1. 环境准备

项目面向 Windows、Python 3.13 和 CUDA 12.4 兼容的 NVIDIA 驱动。PowerShell 中执行：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

检查 PyTorch 能否使用 GPU：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

若输出 `False`，配置中的 `device: cuda` 会自动回退到 CPU，但 GraphSim 尤其会较慢。

## 2. 准备 AIDS700nef 数据

先将原始文件放到以下位置，或在命令中传入实际路径：

```text
data/raw/aids/AIDS700nef.zip
data/raw/aids/aids700nef_ged_astar_gidpair_dist_map.pickle
```

运行预处理：

```powershell
python scripts/prepare_aids.py `
  --graphs data/raw/aids/AIDS700nef.zip `
  --ged data/raw/aids/aids700nef_ged_astar_gidpair_dist_map.pickle `
  --output data/processed/aids700nef.pt
```

此命令会生成训练所需的数据 bundle。它包含 700 个图、节点与边特征、train/val/test 划分、候选集，以及由最小 GED 定义的正样本。未生成 `data/processed/aids700nef.pt` 前，训练无法开始。

## 3. 运行基线

每条命令会自动完成训练和最终测试评估：

```powershell
python train GCN AIDS
python train GEM AIDS
python train GMN AIDS
python train GraphSim AIDS
```

它们分别加载：

```text
configs/aids_gcn.yaml
configs/aids_gem.yaml
configs/aids_gmn.yaml
configs/aids_graphsim.yaml
```

GCN、GEM 是 Ebp：候选图可以独立编码后批量比较，检索较快。GMN、GraphSim 是 Mbp：每个 query 和 candidate 图对都需单独匹配，评估更慢。

## 4. 运行 GRAND

GRAND 组合通过 Ebp 学生模型和 Mbp 教师模型进行蒸馏：

```powershell
python train GRAND-GCN-GMN AIDS
python train GRAND-GCN-GraphSim AIDS
python train GRAND-GEM-GMN AIDS
python train GRAND-GEM-GraphSim AIDS
```

对应配置为：

```text
configs/aids_grand_gcn_gmn.yaml
configs/aids_grand_gcn_graphsim.yaml
configs/aids_grand_gem_gmn.yaml
configs/aids_grand_gem_graphsim.yaml
```

当前 GRAND 配置启用 `bidirectional: true`、分数蒸馏、节点级蒸馏和子图级蒸馏。每个 iteration 先用 Mbp 更新 Ebp，再把刚更新的 Ebp 作为固定目标更新 Mbp。

## 5. 一次实验实际做了什么

例如执行：

```powershell
python train GEM AIDS
```

执行链路如下：

```text
train
  -> run_experiment.py
  -> configs/aids_gem.yaml
  -> train_paper_aids.py
  -> outputs/aids_gem_ebp_<timestamp>/student_best.pt
  -> evaluate_paper_aids.py
  -> outputs/.../metrics_ebp.json
```

训练按 `(query, positive, negative)` 三元组进行。正样本是候选集中 GED 最小的图，负样本是非正样本；模型优化排序损失。GRAND 额外加入教师与学生之间的图分数、节点匹配关系、子图匹配关系的蒸馏损失。

## 5.1 文件协作时序（以 `python train GRAND-GEM-GraphSim AIDS` 为例）

### 第一步：入口和配置

`train` 文件本身几乎没有业务逻辑，只把命令转给 `run_experiment.py`。后者完成三件事：

1. 将方法名规范化，并找到 `configs/aids_grand_gem_graphsim.yaml`；
2. 读取 YAML，生成本次独立的 `outputs/<实验名>_<时间戳>/` 目录；
3. 先启动训练脚本，训练成功后再启动评估脚本。

因此，YAML 是本次实验的“控制面”：它决定数据路径、学生/教师模型、训练轮数、蒸馏开关和评估模式。

### 第二步：训练脚本加载数据

`train_paper_aids.py` 读取 YAML 后调用 `grand.data.load_bundle()` 加载 `data/processed/aids700nef.pt`，再用 `graph_map()` 建立 `graph_id -> graph` 的索引。每个图包含节点特征 `x`、边索引 `edge_index` 和边特征 `edge_attr`。

如果启用了 `subgraph: true`，训练开始前还会调用 `grand.subgraphs.fluidc_partition()`，为每张图生成 `subgraph_groups`；这些分组会随图对象传入后续蒸馏损失。

### 第三步：按配置创建学生和教师

训练脚本调用 `grand.models.build_model()`：

```text
student: gem      -> Ebp(GEMEncoder)
teacher: graphsim -> GraphSim（内部复用 GEMEncoder）
```

`GEMEncoder` 对单张图进行节点/边编码和多轮消息传递，产生节点状态与图 embedding。`Ebp` 用两个图 embedding 的负平方欧氏距离打分。`GraphSim` 则取 GEM 的多层节点状态，构造节点相似度矩阵，再交给 CNN/MLP 对图对打分。

### 第四步：每个 iteration 的前向、损失和更新

`grand.data.sample_triplets()` 从训练 split 随机产生 `(query, positive, negative)`。随后 `train_paper_aids.py` 对三张图执行 `score_pair()`，得到正/负图对分数以及节点表示。

`grand.losses` 只负责把这些输出转换成损失：

```text
ranking_loss              基本正负排序
score_distillation        图对分数分布
node_distillation         节点到节点匹配分布
subgraph_distillation     子图到子图匹配分布
```

先调用 `triplet_loss()` 更新学生；若 `bidirectional: true`，再调用 `reverse_teacher_loss()` 更新教师。每次更新的实际顺序都是：

```text
前向计算 -> 计算 loss -> loss.backward() -> optimizer.step()
```

`loss.backward()` 计算参数梯度，`optimizer.step()` 才真正修改模型参数。验证时使用固定随机种子采样的验证三元组；验证损失改善就覆盖保存 `student_best.pt`，连续多次未改善则 early stopping。

### 第五步：训练完成后的评估

`run_experiment.py` 确认 `student_best.pt` 存在后，启动 `evaluate_paper_aids.py`。评估脚本重新加载同一 bundle 和 YAML，恢复模型参数，然后：

- Ebp：先独立编码所有候选图，再批量与 query embedding 比较；
- Mbp：对 query 和每个候选图逐对调用 `score_pair()`。

得到候选排序后，`grand.metrics.ranking_metrics()` 使用 bundle 中由 GED 定义的真实正样本计算 `NDCG@5` 与 `Recall@5`，最后写入本次输出目录的 `metrics_ebp.json` 或 `metrics_mbp.json`。

### 文件职责速查

```text
scripts/prepare_aids.py   原始 GEXF/GED -> .pt 数据 bundle
configs/*.yaml            实验参数和模型组合
train                    命令包装入口
run_experiment.py         选择配置、创建输出、串联训练和评估
train_paper_aids.py       训练循环、采样、反向传播、保存 checkpoint
grand/data.py             加载 bundle、划分候选集、采样三元组
grand/models.py           GCN/GEM/GMN/GraphSim 前向和打分
grand/losses.py           ranking 与三种蒸馏损失公式
grand/subgraphs.py        FluidC 子图划分
evaluate_paper_aids.py    加载 checkpoint、检索排序、写评估结果
grand/metrics.py          NDCG/Recall 计算
```

## 6. 查看实验结果

每次运行都会生成独立、带时间戳的目录，避免覆盖旧结果：

```text
outputs/<experiment-name>_<YYYYMMDD_HHMMSS_microseconds>/
  config.yaml           本次实际使用的配置副本
  run.json              启动信息
  history.json          每次验证的训练与验证 ranking loss
  student_best.pt       验证集上选出的最佳 checkpoint
  metrics_ebp.json      Ebp 测试指标，若使用 Ebp 评估
  metrics_mbp.json      Mbp 测试指标，若使用 Mbp 评估
```

例如查看 GEM 最终指标：

```powershell
Get-Content outputs/<你的运行目录>/metrics_ebp.json
```

其中：

- `NDCG@5`：前五个结果中，相关图是否排得靠前。
- `Recall@5`：所有 GED 最小的相关图中，有多少被前五个结果检索到。
- `query_latency_ms_p50/p95`：单个 query 的中位数和 P95 检索耗时。

## 7. 使用自定义配置

可以复制一个 YAML，再仅修改必要字段，例如实验名、随机种子、训练参数或蒸馏开关。之后显式指定配置：

```powershell
python run_experiment.py --config configs/aids_gem.yaml
```

注意：`--config` 模式仍会训练并评估；`experiment.name` 应改为新的名字，方便区分输出目录。

常见配置字段：

```yaml
experiment:
  name: aids_gem_ebp
  seed: 42
  mode: baseline             # baseline 或 grand
training:
  device: cuda
  batch_size: 32
  learning_rate: 0.001
  iterations: 30000
  validation_interval: 500
  early_stopping_patience: 10
distillation:
  bidirectional: true        # 仅 GRAND 使用
  score: true
  node: true
  subgraph: true
```

项目当前配置的 `learning_rate: 0.001`、Adam 和 early stopping 与 GRAND 论文的通用训练描述一致。改动超参数时，应使用多个随机种子比较，并保持数据划分、候选集和 GED 正样本定义不变。

## 8. 常见问题

### 找不到数据 bundle

报错通常表示缺少：

```text
data/processed/aids700nef.pt
```

重新执行第 2 节的预处理命令，并确认 YAML 中的 `data.path` 与输出路径一致。

### CUDA 不可用

确认虚拟环境已激活、NVIDIA 驱动与 CUDA PyTorch wheel 兼容。也可临时将 YAML 的 `training.device` 改为 `cpu`，但训练和尤其是 Mbp 评估会慢很多。

### GraphSim 报图节点数超过上限

GraphSim 配置中的 `graphsim_max_nodes: 10` 与 AIDS700nef 当前数据集相匹配。若换用节点更多的数据，必须同时提高该值，并重新考虑 CNN 输入尺寸和池化配置。

### 想单独训练或单独评估

训练：

```powershell
python train_paper_aids.py --config configs/aids_gem.yaml --output outputs/manual_gem
```

评估：

```powershell
python evaluate_paper_aids.py `
  --config configs/aids_gem.yaml `
  --checkpoint outputs/manual_gem/student_best.pt `
  --mode ebp
```

对 GMN、GraphSim 使用 `--mode mbp`。
