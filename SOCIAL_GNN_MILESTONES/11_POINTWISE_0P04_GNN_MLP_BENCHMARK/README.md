# 11 — 0.04 s pointwise GNN/MLP 全基准

时间：2026-10-22 至 2026-10-23。

## 开发跨步

- 放弃旧的 0.16 s 四样本 patch 平均；一个 social timestep 现在对应一个
  0.04 s、25 Hz 的上游 latent 时点。
- node 与 edge 使用完全相同的权威物理时间格。缺失区间保留在完整 trial
  clock 上并通过 mask 排除，不裁短、不插值，也不跨 gap 学习转移。
- 重新生成五场 9D node package 和同时间格 edge traits。
- 在相同数据划分下重训 `graph_full`、`graph_hybrid_p2`、
  `invariant_mlp_full`、`invariant_mlp_hybrid_p2`，每种 3 个 encoder seeds、
  100 epochs、64D `g_t`。
- 对四种 frozen representation 重新拟合 K=6、PCA16 的 sticky HMM 与
  HSMM Dmax=200；共 72 个正式 downstream runs。

## 核心结果

Frozen probe 四项指标依次为 node mean、node absolute difference、all edge、
directed edge R2：

| encoder | 四项 R2 均值 | node mean | node difference | edge | directed edge |
| --- | ---: | ---: | ---: | ---: | ---: |
| Invariant MLP Full | **0.7857** | 0.9877 | 0.8731 | 0.7160 | 0.5660 |
| Graph Full | 0.7648 | 0.9836 | 0.8613 | 0.6857 | 0.5285 |
| Invariant MLP Hybrid P2 | 0.7630 | 0.9839 | 0.7886 | 0.7143 | 0.5651 |
| Graph Hybrid P2 | 0.7420 | 0.9812 | 0.7789 | 0.6816 | 0.5263 |

- `Invariant MLP Full` 是当前两只动物任务的信息保留冠军；`Graph Full` 是
  最强的实际 message-passing 模型。
- Sticky HMM 的 overall ARI 为 Graph Full 0.456、Graph Hybrid P2 0.413、
  MLP Full 0.542、MLP Hybrid P2 0.546。
- MLP Hybrid P2 的 fixed-encoder HMM ARI 最高：0.835 +/- 0.119；但
  MLP Full 的 frozen retention 更高，两者回答的是不同问题。
- HMM 在四种 representation 上都比当前 HSMM 更少碎片、较少 one-point
  bouts，并有更好的同空间 development likelihood。
- 72/72 downstream runs 全部收敛，全部占用 K=6；MLP 的 occupancy 较不
  均衡，但未发生 state collapse。

这些结果不证明 MLP state 更具生物学意义。A4_C4 是重复使用的开发数据，
不同 encoder/PCA 空间的原始 Gaussian likelihood 也不能直接横向比较。

## 本目录快照

- [0.04 s 迁移说明](records/POINTWISE_0P04_MIGRATION.md)
- [完整 pointwise benchmark 报告](records/POINTWISE_0P04_BENCHMARK_REPORT.md)
- `configs/social_gnn_pointwise_9dim_v2.yaml`
- `configs/social_gnn_invariant_mlp_9dim_0p04_v2.yaml`
- `configs/social_gnn_downstream_benchmark_9dim_0p04_v2.yaml`

## 关键实现与本地产物

- node conversion：`social_gnn/latent_node_converter.py`
- edge reconstruction：`social_gnn/edge_extraction.py`
- encoder runner：`social_gnn/experiment.py`
- downstream benchmark：`social_gnn/downstream/benchmark.py`
- 全部本地产物：`evaluation_runs/gnn_9dim_0p04_v2/` 与
  `evaluation_runs/gnn_9dim_0p04_downstream_v2/`
- MLP 汇总：
  `evaluation_runs/gnn_9dim_0p04_downstream_v2/aggregate_invariant_mlp/`

大型 node/edge NPZ、checkpoint、embedding、assignment、CSV 和图片没有复制
进里程碑目录。
