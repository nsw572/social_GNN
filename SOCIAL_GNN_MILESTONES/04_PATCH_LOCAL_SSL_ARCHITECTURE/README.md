# 04 — Patch-local GNN 与 SSL 实验框架

时间：2026-10-11；Git 基线：`1d4a7d9`。

## 架构跃迁

架构从 `Graph -> GNN + TCN` 改为：

```text
fixed social patch
  -> node features + directed edge values/confidences
  -> patch-level GNN
  -> g_t
  -> SSL decoder/head
  -> loss
```

TCN 被彻底移除，不再让 encoder 学第二层神经时序表示。`g_t` 冻结后才交给 HMM/HSMM。

## 同期完成

- masked reconstruction：随机 node/edge trait masking 与 modality dropout。
- full-input preservation / hybrid objective 的实验基础。
- latent → canonical node package 的动态维度转换。
- IMU↔视频身份匹配、IMU/视频同步查看器和按顺序入场确定身份。
- idtracker.ai calibration 的批量管线与五场数据处理入口。

## 关键代码

- `social_gnn/models.py`
- `social_gnn/masked_social_ssl.py`
- `social_gnn/graph_level_ssl.py`
- `social_gnn/experiment.py`
- `social_gnn/training.py`
- `social_gnn/imu_video_identity.py`
- `social_gnn/sequential_entry_identity.py`
- `social_gnn/imu_video_sync_viewer.py`

历史提交里曾同时存在 VAE 支路；VAE 后来经用户决策从活动代码和 benchmark 中移除。
