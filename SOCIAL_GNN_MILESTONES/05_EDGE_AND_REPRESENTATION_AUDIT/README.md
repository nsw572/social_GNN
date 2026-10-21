# 05 — Edge 使用与表示保留审计

时间：2026-10-13 至 2026-10-14。

## 实验问题

1. GNN 是否真的响应视频 edge，而不是只使用 node？
2. `g_t` 中 node、edge、directed role 信息能否被 frozen probe 恢复？
3. mean pooling、role-aware readout、coupled masking 和 relational objective 谁更可靠？
4. 把 `g_t` 从 64D 降到 32D 是否划算？

## 核心结论

- edge zero/shuffle/node-matched-edge-different 干预都会显著改变表示，说明真实 edge 被使用。
- 但当时没有证据表明复杂 relational candidate 稳定优于 strict baseline；不能把 edge sensitivity 夸大为已学到生物学关系。
- 32D bottleneck 几乎没有节省参数，却明显损伤 node/edge retention，因此否决为默认值。
- directed pose-dependent traits 是最难保留的部分；mean pooling 的置换不变性也限制了 identity-ordered target。

## 本目录快照

- [完整实验报告](records/EDGE_EVALUATION_REPORT.md)
- `configs/social_gnn_strict_baseline_6dim_v1.yaml`
- `configs/social_gnn_strict_baseline_g32_6dim_v1.yaml`
- `configs/social_gnn_coupled_mask_6dim_v1.yaml`

## 关键实现

- `social_gnn/evaluate_edge_sensitivity.py`
- `social_gnn/evaluate_information_retention.py`
- `social_gnn/evaluate_role_binding.py`
- `social_gnn/evaluate_representation_geometry.py`

本地产物见 `evaluation_runs/final_strict_baseline/`、`baseline_edge_sensitivity_v1/`、`baseline_information_retention_v*/` 和 `g32_repeated_benchmark/`。
