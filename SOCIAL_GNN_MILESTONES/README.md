# Social-GNN 项目里程碑档案库

建立日期：2026-10-20；最近更新：2026-11-03。

这个目录把 Social-GNN 从视频特征工程到 9D encoder/downstream benchmark 的主要开发跨步集中在 `deepOF` 根目录下。它是一套可浏览、可版本管理的项目史，不替代 `social_gnn/` 中的正式实现。

## 当前主线

```text
原视频
  -> idtracker.ai 质心/身份 + MouseGPT 姿态
  -> 固定 0.04 s 权威 social clock 上的 8 edge values + 8 confidences

上游 IMU latent
  -> 每个 0.04 s 时点一个 9D latent 的 canonical node package

node + edge
  -> SocialTrialPackage 严格校验
  -> patch-local encoder（GNN 或精确置换不变 MLP）
  -> 64D g_t
  -> frozen downstream sticky HMM，当前工作值 K=12、strength=50000、dwell target=2 s
  -> State Atlas 定量/QC/视频复核后端
  -> 外部生物学验证
```

当前没有 TCN/RNN/Transformer；VAE 路线已经从活动实现中移除。
`g_t=32` 是失败消融，默认仍为 64D。当前两只动物数据上，
`Invariant MLP Full + HMM K=6` 是 retention 候选，
`Invariant MLP Hybrid P2 + HMM K=6` 的 fixed-encoder stability 更高；可扩展
图模型候选是 `Graph Full + HMM K=6`。这些排名都不能当作行为准确率结论，
因为尚无全新独立 trial 和人工行为标签验证。

旧的 0.16 s benchmark 现在只保留作历史对照。当前正式数字来自 0.04 s
pointwise 重建；node/edge clock、encoder、PCA 和 downstream assignments 均已
重新生成，不能把旧 assignments 直接重标为 0.04 s。

## 时间线

| 日期 | 目录 | 向前跨步 | 当前地位 |
| --- | --- | --- | --- |
| 2026-09-05 | [01_EDGE_AND_IDENTITY_V0](01_EDGE_AND_IDENTITY_V0/README.md) | 视频身份、平滑、8+8 edge 提取与固定时钟 | 仍是 edge 主线基础 |
| 2026-09-07 | [02_TRIAL_PACKAGE_AND_TCN_PROTOTYPE](02_TRIAL_PACKAGE_AND_TCN_PROTOTYPE/README.md) | full-trial 数据契约与变长 DataLoader；首次 TCN 原型 | 数据层保留，TCN 已废弃 |
| 2026-09-19 | [03_AUTHORITATIVE_NODE_CONVERSION](03_AUTHORITATIVE_NODE_CONVERSION/README.md) | H ZIP 到 canonical node NPZ，权威 clock 对齐 | 已演化为正式输入转换层 |
| 2026-10-11 | [04_PATCH_LOCAL_SSL_ARCHITECTURE](04_PATCH_LOCAL_SSL_ARCHITECTURE/README.md) | 移除 TCN；固定 patch GNN、masked/hybrid SSL、IMU↔视频身份 | 当前 encoder 架构基础 |
| 2026-10-13–14 | [05_EDGE_AND_REPRESENTATION_AUDIT](05_EDGE_AND_REPRESENTATION_AUDIT/README.md) | edge 使用、信息保留、角色与 32D bottleneck 审计 | 证明 edge 被使用，否决 32D |
| 2026-10-14–16 | [06_GRAPH_PRESERVATION_AND_INITIAL_HMM_HSMM](06_GRAPH_PRESERVATION_AND_INITIAL_HMM_HSMM/README.md) | graph bottleneck、node preservation 与首套 HMM/HSMM | 形成 downstream 框架，6D 结果仅历史参考 |
| 2026-10-16 | [07_9D_ALIGNMENT_AND_HYBRID_RERUN](07_9D_ALIGNMENT_AND_HYBRID_RERUN/README.md) | 新 9D latent 权威对齐、重训与时序模型复跑 | 证明 9D 数据链路可运行 |
| 2026-10-19 | [08_9D_FULL_GNN_BENCHMARK](08_9D_FULL_GNN_BENCHMARK/README.md) | 63 个正式 GNN runs 与完整 retention/geometry 审计 | Graph Full 与 Hybrid P2 候选来源 |
| 2026-10-19 | [09_INVARIANT_MLP_AND_DOWNSTREAM_BENCHMARK](09_INVARIANT_MLP_AND_DOWNSTREAM_BENCHMARK/README.md) | invariant MLP、四类聚类/时序模型、K 与 HSMM 检验 | 0.16 s 历史 benchmark |
| 2026-10-20 | [10_ARI_STABILITY_DECOMPOSITION](10_ARI_STABILITY_DECOMPOSITION/README.md) | 分解 encoder-seed 与 downstream-seed 的 partition instability | 0.16 s 历史稳定性基线 |
| 2026-10-22–23 | [11_POINTWISE_0P04_GNN_MLP_BENCHMARK](11_POINTWISE_0P04_GNN_MLP_BENCHMARK/README.md) | 0.04 s node/edge 重建，GNN/MLP 重训与 72-run HMM/HSMM benchmark | 当前综合 benchmark |
| 2026-10-23 | [12_STICKY_HMM_PRIOR_SENSITIVITY](12_STICKY_HMM_PRIOR_SENSITIVITY/README.md) | 90 配置 prior strength/dwell sweep、soft transition-count audit 与状态复现性 | 当前 HMM 参数依据 |
| 2026-11-01 | [13_STATE_ATLAS_V1](13_STATE_ATLAS_V1/README.md) | 冻结 K=12 状态的 directed edge atlas、quality/boundary QC 与四类视频采样 | 当前人工复核后端 |
| 2026-10-30–11-03 | [14_RSLDS_FINITE_PILOT_AND_RUNTIME_AUDIT](14_RSLDS_FINITE_PILOT_AND_RUNTIME_AUDIT/README.md) | genuine rSLDS 隔离实现、有限 pilot 产物与 K=8 外部中断审计 | 可选探索分支；pilot 未完成 |
| 持续更新 | [90_CURRENT_GUIDES](90_CURRENT_GUIDES/README.md) | 当前 README 与 noob guide 快照 | 入门与操作入口 |

## 档案规则

- `records/` 和 `configs/` 是建立档案时的静态快照，保证历史结论不会随主文件改写。
- 正式代码仍在 `../social_gnn/`，正式配置仍在 `../configs/`，测试仍在 `../tests/`。
- 大型 checkpoint、NPZ、CSV、视频和完整搜索输出不复制。它们仍在 `../evaluation_runs/`，由各里程碑 README 给出路径。
- Git commit 是早期阶段最准确的可复现快照；本目录只作导航和结论整理。
- “实验指标好”不等于“行为分类正确”。当前最重要的未完成验证仍是新独立 trial、人工视频复核和外部标签。

## 快速入口

- 第一次接触项目：先看 [90_CURRENT_GUIDES](90_CURRENT_GUIDES/README.md)。
- 想知道为何不用 TCN/VAE/32D：看 02、04、08。
- 想查 edge 是否真的进入了表示：看 05。
- 想查当前 0.04 s 模型选择和 HMM/HSMM 结论：看 11。
- 想查 sticky prior 到底有多强、参数为何保留 50000/2 s：看 12。
- 想查当前 K=12 states 的定量特征、质量混杂和视频采样：看 13。
- 想查 rSLDS 做到哪里、哪些 cell 有产物、K=8 为何没有结果：看 14。
- 想追溯旧 0.16 s instability 分解：看 09、10。
- 想找本地未入 Git 的运行结果：看 [ARTIFACT_INDEX.md](ARTIFACT_INDEX.md)。
