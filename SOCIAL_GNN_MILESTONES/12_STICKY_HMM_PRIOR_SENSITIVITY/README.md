# 12 — Sticky HMM prior sensitivity 与状态复现性

时间：2026-10-23。

## 实验问题

0.04 s benchmark 的 K=6 Gaussian HMM 使用 sticky Dirichlet transition
prior。本阶段冻结 Graph Full embedding 与每个 encoder seed 已拟合的 PCA16，
只重新拟合 HMM，用于回答：状态边界和六状态 partition 到底有多依赖 sticky
prior，以及更强正则能否解决随机种子不稳定。

正式网格包括：

- strength sweep：10、1000、10000、25000、50000、100000，固定 dwell=2 s；
- dwell grid：strength 10000/50000 × dwell 1/2/4 s；
- 3 encoder seeds × 3 HMM seeds；
- 去重后共 90 个配置，其中严格校验并引用已有的 9 个 50000/2 s runs。

没有重训 encoder 或 PCA，没有使用历史 0.16 s assignments。

## Prior 实现审计

- `p_stay,target = 1 - dt / tau`；dt=0.04 s、tau=2 s 时目标为 0.98。
- strength 是每个 source-state transition row 的总 sticky pseudocount mass，
  按目标 transition matrix 分配，并非只加在 diagonal。
- 每个单元另有 0.01 isotropic pseudocount。
- prior 既用于初始化，也在每次 EM M-step 中重新加入。
- 数据证据是各独立 valid sequence 的 forward-backward soft `xi` expected
  transition counts；不会跨 trial 或 invalid gap 增加转移。

在 50000/2 s 下，每行 prior mass 为 50000.06，数据 soft transitions 平均为
23510.3；prior/data row ratio 平均 2.216，范围 1.544–4.461。因此该先验有
实质性作用，不是一个可忽略的初始化偏好。

## 核心结果

| strength，dwell=2 s | fitted self-transition | bouts/min | median bout | ARI vs 50000 | within-encoder ARI | cross-encoder ARI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.9034 | 130.06 | 0.311 s | 0.821 | 0.701 | 0.332 |
| 10000 | 0.9418 | 98.66 | 0.400 s | 0.856 | 0.722 | 0.353 |
| 25000 | 0.9581 | 88.54 | 0.431 s | 0.914 | 0.741 | 0.364 |
| 50000 | 0.9674 | 81.56 | 0.476 s | 1.000 | 0.741 | 0.361 |
| 100000 | 0.9728 | 78.94 | 0.493 s | 0.972 | 0.743 | 0.363 |

- prior 明显控制时间碎片化和 bout boundary。
- 六状态 partition 大体保留但不是完全不变；25000–100000 构成较宽的
  high-strength stability region。
- 增强 prior 只能小幅提高 fixed-representation reproducibility；
  cross-encoder ARI 始终约 0.33–0.36，representation variation 仍是主要限制。
- 50k 下把 dwell target 改成 1 s 或 4 s，相对 2 s baseline 的 ARI 分别为
  0.930 和 0.969；没有足够证据更换当前 target。
- 所有 90 个 runs 收敛并占用全部六个 states，K effective 约 5.82–5.88，
  没有 prior-induced state collapse。

当前开发建议仍为 **strength=50000、target dwell=2 s**。这是稳定区域内的
工程默认值，不是生物学 dwell-time 的估计。

## 本目录快照

- [完整 sensitivity 报告](records/STICKY_PRIOR_SENSITIVITY_REPORT.md)
- [Prior 数学与实现审计](records/PRIOR_IMPLEMENTATION_AUDIT.md)
- `records/strength_summary.csv`
- `records/dwell_grid_summary.csv`
- `records/experiment_manifest.json`

## 关键实现与本地产物

- HMM 与 soft expected counts：`social_gnn/downstream/gaussian_hmm.py`
- sensitivity runner：`social_gnn/downstream/sticky_sensitivity.py`
- tests：`tests/test_sticky_sensitivity.py`、`tests/test_downstream_temporal.py`
- 完整模型、assignments、逐 run CSV 和 figures：
  `evaluation_runs/gnn_9dim_0p04_downstream_v2/sticky_prior_sensitivity/`

本阶段全套仓库测试结果为 156 passed。稳定性仍不等于行为正确性；A4_C4
不是 untouched test，也没有独立人工行为标签。
