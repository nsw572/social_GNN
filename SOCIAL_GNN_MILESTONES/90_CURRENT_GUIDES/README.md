# 90 — 当前使用说明快照

本目录保存最近一次归档时的用户向文档与主报告快照：

- [Noob's Guide](records/NOOBS_GUIDE.md)：项目概念、数据、训练、评估和常见坑的入门说明。
- [Social-GNN README](records/SOCIAL_GNN_README.md)：较正式的模块、命令和配置说明。
- [0.04 s 迁移说明](records/POINTWISE_0P04_MIGRATION.md)：为何旧 0.16 s
  artifacts 不能直接复用。
- [当前完整 benchmark](records/POINTWISE_0P04_BENCHMARK_REPORT.md)：四种
  GNN/MLP encoder 与 HMM/HSMM 的正式数字。
- [Sticky prior sensitivity](records/STICKY_PRIOR_SENSITIVITY_REPORT.md)：
  50000/2 s 参数依据、prior leverage 与复现性结果。

这些文件是 2026-10-23 的静态快照。继续开发时应优先阅读主目录中的
`social_gnn/README.md` 和 `social_gnn/NOOBS_GUIDE.md`；若想知道本次归档时
的结论，再看这里的快照。

目录中保留的 `GNN_9DIM_FULL_BENCHMARK_REPORT.md`、
`NODE_PRESERVATION_REPORT.md` 和 `downstream/DOWNSTREAM_REPORT.md` 是旧的
0.16 s/6D 历史报告；当前结论以 pointwise 与 sticky 两份新报告为准。

## 当前不要混淆的事项

- 当前 node 是 9D；历史报告中 6D 只代表旧上游版本。
- edge 是 8 values + 8 per-trait confidences，模型输入 16D，不是 16 个独立物理 trait。
- 当前 patch grid 是权威 0.04 s pointwise clock；每行对应一个 25 Hz 上游
  latent 时点。旧 0.16 s 结果只作历史对照。
- encoder 是 patch-local；HMM/HSMM 在 frozen `g_t` 之后，不属于神经 encoder。
- 当前默认 `g_t=64`；32D 是失败消融。
- VAE 和 TCN 都不是当前主线。
- 当前 sticky HMM 工作值为 K=6、prior strength=50000、dwell target=2 s；
  这是 development 默认值，不是生物学时长结论。
- 当前两只动物 retention 最强的是 Invariant MLP Full，最强 message-passing
  模型是 Graph Full；两者都尚未通过独立 trial 和人工行为标签验证。
