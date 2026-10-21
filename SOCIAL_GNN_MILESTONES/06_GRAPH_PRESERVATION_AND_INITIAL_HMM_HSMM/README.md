# 06 — Graph bottleneck、node preservation 与首套 HMM/HSMM

时间：2026-10-14 至 2026-10-16。

## 开发跨步

- 将 decoder 强制放在最终 graph bottleneck 后，减少绕过 `g_t` 的 shortcut。
- 系统比较 graph-level full preservation、masked reconstruction、hybrid、soft-role 和 set reconstruction。
- 建立完整 frozen downstream：embedding export、train-only scaling/PCA、Gaussian HMM、显式 duration Gaussian HSMM、K 搜索、state/bout export 和可视化。
- 严格保持 trial 与 invalid-gap 边界，不允许时序转移跨边界。

## 当时的结论

- SSL 相对随机 GNN 明显增强 edge retention，但 node preservation 仍需单独约束。
- Hybrid preservation + masked reconstruction 成为鲁棒候选。
- HMM 比当前 shifted-Poisson HSMM 更少碎片、held-out likelihood 更好；HSMM 被保留为有效负结果。
- 这些主实验使用历史 6D node，后续已由 9D benchmark 取代，不能与当前 9D 数字混用。

## 本目录快照

- [next-phase validation](records/NEXT_PHASE_REPORT.md)
- [node preservation](records/NODE_PRESERVATION_REPORT.md)
- [首套 downstream 报告](records/DOWNSTREAM_REPORT.md)
- [6D Hybrid HMM/HSMM 比较](records/HYBRID_HMM_HSMM_COMPARISON_6D.md)
- 关键配置位于 `configs/`。

## 关键代码

- `social_gnn/graph_level_ssl.py`
- `social_gnn/node_preservation_ssl.py`
- `social_gnn/downstream/gaussian_hmm.py`
- `social_gnn/downstream/gaussian_hsmm.py`
- `social_gnn/downstream/search.py`
- `social_gnn/downstream/state_export.py`
