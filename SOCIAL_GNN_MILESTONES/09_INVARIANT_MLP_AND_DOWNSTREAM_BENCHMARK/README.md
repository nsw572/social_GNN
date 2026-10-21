# 09 — Invariant MLP 与最终 9D downstream benchmark

时间：2026-10-19。

## 开发跨步

- 实现两只动物输入的精确置换不变 MLP：对原顺序和一致交换后的 node/edge 输入使用共享 `F`，再平均两个输出。
- 训练 MLP Full 与 Hybrid P2，各 3 seeds、100 epochs，并与 Graph Full/Hybrid P2 公平比较。
- 统一导出 frozen 64D embedding。
- 比较 KMeans、GMM、HMM、显式 duration HSMM；screening 覆盖 K=4/6/8/10/12，confirmation 覆盖 3 encoder seeds × 3 downstream seeds。
- 共完成 120 个 screening 与 144 个 confirmation downstream runs。

## 当前结论

- 两只动物经验最强候选：`Invariant MLP Full + HMM K=6`。
- 可扩展到更多动物的图候选：`Graph Full + HMM K=6`。
- HMM 明显减少 KMeans/GMM 的碎片化，在每个固定 encoder/PCA 空间内优于 HSMM likelihood。
- HSMM 的 Dmax=25/50/100 敏感性不能解释其劣势；当前 duration family 没有展示净收益。
- K=6 是工作值；K=4 是粗粒度 control，K=8 是 sensitivity analysis。
- GNN 与 MLP 的 K=6 partition ARI 只有约 0.22–0.30，说明 encoder inductive bias 会实质改变下游状态。
- 没有新独立 trial，A4_C4 是复用开发数据；当前 state 只适合定向视频复核，不能声称行为准确率。

## 本目录快照

- [完整 downstream benchmark 报告](records/DOWNSTREAM_9DIM_BENCHMARK_REPORT.md)
- `configs/social_gnn_invariant_mlp_9dim_v1.yaml`
- `configs/social_gnn_downstream_benchmark_9dim_v1.yaml`

本地产物：`evaluation_runs/gnn_9dim_downstream_v1/`。
