# 01 — 固定时钟 edge 与视频身份管线

时间：2026-09-05；Git 基线：`8ea96f6`，随后由 `db712ba` 固定 16-channel edge 模型输入。

## 这一步解决了什么

- 将原视频并行交给 idtracker.ai 和 MouseGPT：前者提供质心和稳定身份，后者提供鼻尖、头向和关键点置信度。
- 增加 idtracker.ai 参数校准、身份交换峰的额外平滑、逐帧身份匹配与可视化检查。
- 建立固定 social clock 上的 edge extraction：每个 patch 输出 8 个 directed edge traits 和逐 trait confidence。
- state-like traits 在 patch 内做 confidence-aware aggregation；dynamic traits由相邻 patch 位置和 timestamp-derived `dt` 计算。
- 保持 `edge_values [T,N,N,8]` 与 `edge_confidence [T,N,N,8]` 分离，只在模型输入处拼为 16 channels。

## 关键代码

- `social_gnn/video_feature_pipeline.py`
- `social_gnn/edge_extraction.py`
- `social_gnn/idtracker_export.py`
- `social_gnn/idtracker_identity_smoothing.py`
- `social_gnn/identity_matching.py`
- `social_gnn/identity_match_visualization.py`
- `social_gnn/graph_builder.py`

## 保留下来的设计决定

帧号和时间戳必须严格对齐原视频；FPS 不硬编码；丢失姿态时保留 value/confidence 的分离语义；edge 保持 dense directed `[N,N]` 契约。这个阶段至今仍是视频 edge 主线的基础。

早期完整快照可用 `git show 8ea96f6` 查看。
