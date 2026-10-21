# 08 — 9D Social-GNN 全量 benchmark

时间：2026-10-19。

## 实验规模

- 21 个非冗余 GNN 条件 × 3 seeds = 63 个正式 100-epoch runs。
- 输入：`node [T,2,9]`，`edge values/confidence [T,2,2,8]`，模型入口 edge 为 16 channels。
- frozen retention probes、role binding、permutation audit、representation geometry 全部统一执行。
- VAE 从活动实现中移除。

## 核心结论

- `graph_full` 是四项 retention 指标的平衡冠军：score 0.784，且动物同时置换误差为零。
- `graph_hybrid_p2` 是更偏鲁棒性的候选：full preservation 权重 2 加 masked reconstruction。
- 32D `g_t` effective rank 与 retention 显著下降；默认保持 64D。
- node-only 与 time-shuffled-edge controls 的 edge R² 接近零，进一步确认模型使用对齐 edge。
- retention 不是行为可分性；最终选择仍必须由 frozen downstream 和独立生物学验证决定。

## 本目录快照

- [完整 9D GNN benchmark 报告](records/GNN_9DIM_FULL_BENCHMARK_REPORT.md)
- `configs/social_gnn_full_benchmark_9dim_v1.yaml`

本地产物：`evaluation_runs/gnn_9dim_full_v1/`。
