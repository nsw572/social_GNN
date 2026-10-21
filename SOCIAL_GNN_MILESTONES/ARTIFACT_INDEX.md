# 本地实验产物索引

`evaluation_runs/` 中的数据量大且多数被 `.gitignore` 排除。本文件只记录位置，不复制 checkpoint、embedding、CSV、NPZ、图片或视频。

| 阶段 | 本地产物目录 | 主要内容 |
| --- | --- | --- |
| edge sensitivity | `evaluation_runs/baseline_edge_sensitivity_v1/` | edge 干预与位移响应 |
| information retention | `evaluation_runs/baseline_information_retention_v1/`, `v2/` | frozen probes |
| strict baseline | `evaluation_runs/final_strict_baseline/` | 6D 选定 checkpoint 与评估 |
| relational candidates | `evaluation_runs/relational_ssl_models/` | coupled/relational SSL 候选 |
| next phase | `evaluation_runs/next_phase/` | graph-bottleneck 与 SSL gain |
| node preservation | `evaluation_runs/node_preservation_v1/` | hybrid、角色、set reconstruction、HMM sanity |
| initial downstream | `evaluation_runs/downstream_v1/` | 6D HMM/HSMM 搜索与 state export |
| aligned 9D hybrid | `evaluation_runs/hybrid_9dim_v1/` | 9D nodes、encoder、probes、HMM/HSMM |
| full 9D GNN | `evaluation_runs/gnn_9dim_full_v1/` | 63 runs、probes、role/geometry audits |
| final downstream benchmark | `evaluation_runs/gnn_9dim_downstream_v1/` | invariant MLP、embedding、120 screening + 144 confirmation runs |
| ARI stability decomposition | `evaluation_runs/gnn_9dim_downstream_v1/comparisons/ari_stability_decomposition/` | 576 pairwise ARIs、alignment audit、quality tables、32 plots 与报告 |

这些目录不是长期可移植档案。若要在另一台机器完整复现，需要同时备份对应 `evaluation_runs` 子目录，或从本目录中的配置快照重新运行。
