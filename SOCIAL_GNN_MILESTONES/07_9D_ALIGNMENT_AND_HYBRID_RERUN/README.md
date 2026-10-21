# 07 — 新 9D latent 对齐与 Hybrid 重跑

时间：2026-10-16；对应大提交：`3ef6dca` 中的 9D/temporal 部分。

## 这一步解决了什么

- 接受上游 batch-3 9D latent，并让 node dimension 动态化，不再把 6 写死。
- 通过 `--authoritative-clock-npz` 继承 canonical clock、trial ID 和 identity order。
- 每个 0.16 s patch 对齐/聚合四个 25 Hz latent sample；边界覆盖不足用 mask 表示。
- 五场 package 全部通过 `SocialTrialPackage` 的身份、时钟、shape 与 finite validation。
- 在 9D node 上重训三 seed Hybrid SSL，并重新比较 HMM/HSMM。

## 当时结果

Hybrid 表示能同时保留 node mean、两鼠 node difference 与 edge 信息；HMM 仍优于 HSMM。这个阶段首先证明“新版 9D 上游数据能够完整跑通”，后来的 63-run full benchmark 才负责全面模型选择。

## 本目录快照

- [9D Hybrid rerun 报告](records/HYBRID_9DIM_RERUN_REPORT.md)
- `configs/social_gnn_hybrid_9dim_v1.yaml`
- `configs/social_gnn_downstream_9dim_v1.yaml`

本地产物：`evaluation_runs/hybrid_9dim_v1/`。
