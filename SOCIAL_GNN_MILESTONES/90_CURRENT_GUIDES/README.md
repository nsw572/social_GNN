# 90 — 当前使用说明快照

本目录保存建立里程碑档案库时的用户向文档快照：

- [Noob's Guide](records/NOOBS_GUIDE.md)：项目概念、数据、训练、评估和常见坑的入门说明。
- [Social-GNN README](records/SOCIAL_GNN_README.md)：较正式的模块、命令和配置说明。

这些文件是 2026-10-20 的静态快照。继续开发时应优先阅读主目录中的 `social_gnn/README.md` 和 `social_gnn/NOOBS_GUIDE.md`；若想知道某次结论当时到底写了什么，再看这里的快照。

## 当前不要混淆的事项

- 当前 node 是 9D；历史报告中 6D 只代表旧上游版本。
- edge 是 8 values + 8 per-trait confidences，模型输入 16D，不是 16 个独立物理 trait。
- patch grid 是权威 0.16 s clock；node latent 可以来自更长且重叠的感受野。
- encoder 是 patch-local；HMM/HSMM 在 frozen `g_t` 之后，不属于神经 encoder。
- 当前默认 `g_t=64`；32D 是失败消融。
- VAE 和 TCN 都不是当前主线。
