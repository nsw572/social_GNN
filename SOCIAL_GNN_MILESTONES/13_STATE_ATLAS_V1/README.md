# 13 — State Atlas v1

时间：2026-11-01。

## 向前跨步

本阶段在冻结的 0.04 s Social-GNN 状态系统之上建立了只读 State Atlas：

```text
Graph Full 64D g_t
  -> train-only scaler/PCA16
  -> frozen Sticky HMM K=12
  -> directed edge State Atlas + QC + video review catalog
```

Atlas 不重训 encoder、不重拟合 PCA/HMM、不改变 K，也不自动给状态赋行为名称。
它只处理 A1_C1、A3_C3、A4_B4、A4_C4；A5_C5 保持锁定排除。

## 数据契约

- 每场一个 `[T,2,2,10]` directed-edge tensor；A->B 与 B->A 分开。
- 10 个通道为原始 8 edge traits、`dynamic_quality`、`pose_quality`。
- 原始 per-trait confidence/coverage/validity 另存，不与 trait value 相乘。
- 缺失 trait 在分析 tensor 中为 NaN，并由 `edge_valid` 明确标记。
- `dynamic_quality = mean(sqrt(confidence * coverage))`，覆盖 radial
  approach、pair closing、movement alignment；不使用运动幅度。
- `pose_quality` 用同一 support 公式汇总 4 个 pose-related traits，缺失会降低
  分数。
- dynamic sign flips 与 absolute jumps 单独保存，不偷偷折进 quality。

## 正式运行结果

- 冻结 A4_C4 assignment 逐点复现，包括 posterior。
- 四场 tensor 长度：50,792、47,012、52,284、46,128。
- 有效 0.04 s 点合计 184,315。
- 12 个状态 occupancy 为 4.37%–12.57%。
- pose availability 最低的状态为 2、3、8；其 median pose_quality 分别约
  0.388、0.393、0.420。
- dynamic_quality 在所有状态的中位数均饱和为 1.0，因此当前 v1 proxy 主要应
  结合 low-support fraction、sign flips 和 jump 表阅读，不能拿中位数细排。
- 状态 6、7 的短 bout 比例最高，分别约 74.8% 和 74.0%。
- 全局 boundary-adjacent pose_quality 比 interior 中位数低约 0.032，低 pose
  quality 比例约富集 1.20–1.25 倍；这是 QC 描述，不是边界伪影的因果证明。
- 已生成 144 个 deterministic clip 计划：12 states × 4 categories × 3。

## 视频状态

四段原视频在生成时均随 NAS/Y 盘离线而不可访问，因此没有伪造“已渲染”结果。
Atlas 保存了基于物理时间的 clip catalog、参考 frame span 和 PowerShell/bash
提取脚本。视频重新挂载后可按：

```text
video_pts = social_time + video_time_offset_s
```

渲染；不假设一个 25 Hz state step 等于一个约 29.97 fps 视频帧。

## 关键实现与产物

- 核心：`social_gnn/downstream/state_atlas.py`
- clip planner：`social_gnn/downstream/state_atlas_video.py`
- 配置：`configs/social_gnn_state_atlas_v1.yaml`
- tests：`tests/test_state_atlas.py`
- 本地完整产物：`evaluation_runs/state_atlas_v1/`
- 完整报告：`evaluation_runs/state_atlas_v1/state_atlas_v1_report.md`

验证：State Atlas + downstream temporal/video 回归子集共 33 tests passed；
另做了真实 NPZ schema、锁定 trial 排除、144 clip 网格与有效点总数审计。

## v1 可视化扩展

在同一冻结 Atlas 上增加了且只增加了三种只读视图：

1. 四场独立、可离线打开的 Plotly State Timeline Explorer；支持物理时间窗、
   trial 切换、通道选择、缩放/平移、raw/min-max/median-IQR 显示和可选百分位裁剪。
2. 13 通道 State Fingerprint；CSV 使用全部有效样本给出 median、IQR、P10、
   P90、10% trimmed mean 和 support，热图颜色标准化但格内保留 raw median，
   并附 violin distributions。
3. 6 通道 State x Pose Validity Matrix；同时导出绝对有效率、相对通道全局
   baseline 的差值、valid-only confidence，以及四场 trial-stratified 明细。

实现入口为 `social_gnn/downstream/state_atlas_visualizations.py`，配置为
`configs/social_gnn_state_atlas_visualizations_v1.yaml`，本地产物位于
`evaluation_runs/state_atlas_v1/visualizations/`。本扩展不重训模型、不使用
A5_C5、不将 confidence 乘入原始 trait，也不把低 pose validity 直接解释成
行为状态伪影。
