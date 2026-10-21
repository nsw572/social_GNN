# Noob's Guide to the Social-GNN Project

这是一份给第一次接触本项目的人看的操作指南。它不要求你提前理解
GNN、SSL、HMM 或 MouseGPT，但默认你会使用 PowerShell，并能看懂基本的
Python/YAML。

如果只记住一句话，请记住：

> 我们把每只鼠看成一个 node，把两只鼠之间的距离、朝向和相对运动看成
> directed edge；GNN 每 0.16 秒生成一个社会行为向量 `g_t`，之后再由
> HMM/HSMM 在时间上发现行为状态。

## 1. 这个项目到底在做什么？

目标不是直接给视频逐帧贴上“打架”“社交”“独处”等人工标签，而是先学习
一个无监督的社会行为表示：

```text
IMU -> 每只鼠的 node latent
视频 -> 两只鼠之间的 edge traits
            |
            v
同一个固定 social patch 内严格对齐
            |
            v
patch-level GNN -> g_t
            |
            v
HMM / HSMM -> 离散状态、bout 和转移规律
```

当前神经网络阶段是严格 patch-local 的：

```text
SocialTrialPackage
-> node features + directed edge features
-> GNN
-> graph embedding g_t
-> SSL decoder/head
-> loss
```

这里没有 TCN、RNN、Transformer，也没有其它神经时序编码器。时间关系留给
后面的 HMM/HSMM。VAE 分支已经从活跃代码中删除。

## 2. 最重要的概念：一只鼠才是一个 node

本项目不是 body-part graph。

两只鼠时：

- `N = 2`，所以图里只有两个 node；
- MouseGPT 的鼻尖、耳朵、脊柱和髋部关键点不是 GNN node；
- 这些关键点用于估计头向、鼻尖位置和 edge confidence；
- idtracker.ai 提供身份稳定的质心轨迹；
- IMU encoder 为每只鼠提供 node latent。

因此当前正式输入是：

```text
node_features     [T, 2, 9]
node_mask         [T, 2]
edge_values       [T, 2, 2, 8]
edge_confidence   [T, 2, 2, 8]
patch_start_s     [T]
patch_end_s       [T]
identity          [2]
trial_id          scalar/string
```

`T` 是这场 trial 的 social patch 数。node 维度目前是 9，但读取和模型构建尽量
动态推断 `D_node`，避免上游再次改变 latent 维度时大面积改代码。

## 3. 0.16 秒是什么意思？

当前 social clock 的 stride 是 0.16 秒。每个 `g_t` 对应权威 node clock 中的
一个 patch。

上游 IMU latent 可能使用大约一秒的感受野。这不代表 Social-GNN patch 也应改成
一秒。正确理解是：

- 输出仍然每 0.16 秒一个 patch；
- 相邻 node latent 因上游感受野重叠而不完全独立；
- node 和 edge 必须描述完全相同的物理时间区间；
- 实际 `dt` 从时间戳计算，代码不把 0.16 硬编码进速度公式。

不要为了“看起来独立”擅自抽掉 3/4 patch，也不要自己重建另一套视频 clock。

## 4. 八个 edge traits 是什么？

对于一条有向边 `source i -> target j`，当前八个 value 是：

1. `centroid_distance`：两鼠质心距离；
2. `nose_to_nose_distance`：鼻尖间距离；
3. `source_nose_to_target_centroid_distance`：source 鼻尖到 target 质心距离；
4. `source_faces_target`：source 面向 target 的程度；
5. `head_direction_alignment`：两鼠头向一致程度；
6. `source_radial_approach_speed`：source 朝 target 接近的径向速度；
7. `pair_closing_speed`：两鼠整体接近/分离速度；
8. `movement_direction_alignment`：两鼠运动方向一致程度。

其中部分 trait 数值上是对称的，部分是有方向的。为了让数据契约统一，它们仍然
全部存进稠密的 `[N,N]` 有向矩阵。对角线 `i -> i` 不参与消息传递和损失。

每个 value 都有自己独立的 confidence：

```text
edge input = concat(edge_values, edge_confidence)
           = 8 values + 8 confidences
           = 16 channels
```

confidence 不会永久乘进 value。低置信数据仍保持“值”和“这个值有多可信”两个
独立信息。coverage 是 QC 数据，目前不作为第 17 个输入通道。

## 5. edge 如何从视频产生？

视频分成两条并行支路：

```text
视频
|- idtracker.ai -> 身份稳定的质心、轨迹、速度
|- MouseGPT     -> bbox、关键点、鼻尖、头向和 pose confidence
```

然后逐帧把 MouseGPT 实例匹配到 idtracker.ai 身份。idtracker.ai 身份是视频侧的
权威身份；MouseGPT 的临时 `track_id` 只用于诊断。

帧级信息进入权威 patch 后：

- 五个 state-like traits 做 confidence-aware weighted mean；
- 三个 dynamic traits由相邻 patch 的稳定质心摘要做有限差分；
- 缺失值使用零 confidence，而不是伪造一个正常观测；
- 最后一格没有 `t -> t+1`，所以动态 trait 的 value/confidence/coverage 为零。

完整视频流程入口是：

```powershell
python -m social_gnn.video_feature_pipeline `
  --config configs\video_feature_pipeline_v0.yaml `
  --dry-run
```

第一次运行先保留 `--dry-run` 检查所有外部命令和路径。确认后再去掉它。详细 GUI、
idtracker 平滑和可视化选项见 [README.md](README.md)。

## 6. IMU 身份如何与视频身份对齐？

“两个文件都写着 animal 1”不代表它们是同一只鼠。

当前可靠思路是利用两只鼠依次进入场地的时间差：

1. 从每只鼠的 IMU 加速度突变确定进入时间；
2. 在只有第一只鼠位于场地的时间窗中，检查哪个 idtracker.ai identity 有有效轨迹；
3. 生成 IMU animal 到视频 identity 的可审计映射；
4. 按这个映射重排真实 latent stream；
5. 最后才写 `--output-identity 1 2`。

只修改 identity 标签而不重排 latent 数组是错误的。相关实现位于：

- `sequential_entry_identity.py`；
- `imu_video_identity.py`；
- `imu_video_sync_viewer.py`。

映射程序不会修改原始 idtracker.ai session，而是输出新的 canonical CSV 和证据
manifest。证据不充分时应报错停止，不应猜测身份。

## 7. `SocialTrialPackage` 为什么重要？

它是 GNN 前的最后一道防线。一份 package 代表一整场连续 trial，并严格验证：

- node 和 edge 的 `T` 完全一致；
- identity 顺序完全一致；
- `patch_start_s`/`patch_end_s` 完全一致；
- tensor shape 正确；
- value 有限；
- confidence 位于 `[0,1]`；
- 固定 clock 合法。

遇到不一致时会带着 trial ID 报错，不会自动裁切、重采样或交换身份。

DataLoader 的规则同样重要：

- 一个 sample 永远是一整场 trial；
- shuffle 只能改变 trial 顺序；
- 不会打乱 trial 内时间；
- 不会把两场 trial 的时间轴拼接；
- batching 只在 batch 内把 `T` pad 到最长，并产生 `time_mask [B,T]`。

## 8. 目录地图

第一次读代码建议按这个顺序：

| 文件/目录 | 用途 |
| --- | --- |
| `social_gnn/data.py` | `SocialTrialPackage`、Dataset、DataLoader 和严格验证 |
| `social_gnn/graph_builder.py` | 把 8 value 与 8 confidence 拼成 16D edge input |
| `social_gnn/models.py` | 核心 patch-level Social-GNN encoder |
| `social_gnn/experiment.py` | 正式训练入口、切分、标准化、checkpoint 和 embedding 导出 |
| `social_gnn/masked_social_ssl.py` | node/edge 动态 masking 与基础 masked reconstruction |
| `social_gnn/graph_level_ssl.py` | 只允许 decoder 从最终 `g_t` 重建的 graph bottleneck |
| `social_gnn/node_preservation_ssl.py` | full/masked hybrid、set target 和 dual branch 实验 |
| `social_gnn/evaluation.py` | 冻结 encoder、载入 checkpoint 和公共评估工具 |
| `social_gnn/edge_extraction.py` | 帧级视频特征到固定 patch edge traits |
| `social_gnn/video_feature_pipeline.py` | MouseGPT/idtracker/身份匹配/edge 提取总调度 |
| `social_gnn/downstream/` | HMM、HSMM、模型搜索、状态导出和可视化 |
| `configs/` | 可复现实验 YAML |
| `tests/` | 数据契约、模型、身份、edge、HMM/HSMM 的单元测试 |
| `evaluation_runs/` | 本地 checkpoint、embedding 和结果；默认不提交 Git |

如果只是想理解核心网络，先读 `data.py -> models.py -> experiment.py`。视频和身份流程
可以随后再读。

## 9. 第一次运行：五分钟检查

打开 PowerShell：

```powershell
Set-Location C:\Users\DELL\Desktop\deepOF
conda activate deepof
```

如果当前 PowerShell 不能 `conda activate`，后续命令中的 `python` 可以替换为：

```text
C:\Users\DELL\anaconda3\envs\deepof\python.exe
```

先检查 PyTorch 和 GPU：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
nvidia-smi
```

再运行正式测试集：

```powershell
python -m unittest discover -s tests -v
```

不要从仓库根目录直接运行不带 `-s tests` 的 `unittest discover`。仓库里有历史目录名
会被 unittest 错认成 Python package。

最后做一轮 9D 冒烟测试：

```powershell
python -m social_gnn.experiment `
  --config configs\social_gnn_full_benchmark_9dim_v1.yaml `
  --device cuda:1 `
  --smoke-test
```

`--smoke-test` 只用第一份完整 trial、每个 condition 跑一个 train/eval step。它用于
检查路径、shape、显存、前向、反向和导出，不能当作科学结果。

## 10. 如何正式训练？

当前完整 9D benchmark：

```powershell
python -m social_gnn.experiment `
  --config configs\social_gnn_full_benchmark_9dim_v1.yaml `
  --device cuda:1 `
  --seed 20261013
```

这个配置包含 21 个 condition，不是最小生产配置。如果只想训练当前候选，复制一份
YAML，保留共同的 trials/model/ssl/training，只把 `runs` 改为以下之一。

平衡信息保留候选：

```yaml
runs:
  - name: graph_full
    objective: graph_full_ssl
    edge_mode: full
    selection_metric: total
```

带 masked robustness 的候选：

```yaml
runs:
  - name: graph_hybrid_p2
    objective: graph_hybrid_ssl
    edge_mode: full
    preserve_loss_weight: 2.0
    selection_metric: total
```

当前不建议把 `graph_dim` 改成 32。9D benchmark 中 32D 的 node、edge 和 role 信息
保留都明显下降。默认继续使用：

```yaml
model:
  hidden_dim: 64
  graph_dim: 64
  layers: 2
```

## 11. YAML 配置怎么看？

训练配置的主要部分：

| 字段 | 含义 |
| --- | --- |
| `output_dir` | 实验输出根目录 |
| `seed` | 随机种子 |
| `device` | `cpu`、`cuda:0` 或 `cuda:1` |
| `trials` | 每场 trial 的 node NPZ 与 edge NPZ |
| `split.validation_trials` | checkpoint 选择使用的整场 trial |
| `split.excluded_trials` | 不参与训练和选择的 trial |
| `model.hidden_dim` | GNN message passing 宽度 |
| `model.graph_dim` | 最终 `g_t` 维度 |
| `ssl` | mask 概率、modality dropout 和 loss 权重 |
| `training` | epochs、batch size、学习率等 |
| `runs` | 在同一数据切分上比较的模型条件 |

Windows 路径最好放在单引号中。包含中文的 YAML 应保存为 UTF-8。旧版 PowerShell
有时会把正确的中文路径显示成乱码；最终以 Python 能否打开文件和 `Test-Path
-LiteralPath` 的结果为准，不要直接把屏幕上的乱码复制回配置。

## 12. 主要 SSL objective 怎么区别？

| objective | 简单解释 | 适合做什么 |
| --- | --- | --- |
| `masked_ssl` | 随机遮 node/edge，再重建被遮内容 | 基础 masked multimodal SSL |
| `graph_full_ssl` | 完整输入经过 GNN 和 `g_t` bottleneck，再重建图摘要 | 当前最均衡的信息保留候选 |
| `graph_hybrid_ssl` | 同时做 full preservation 和 masked reconstruction | 兼顾保留与缺失输入鲁棒性 |
| `graph_set_full_ssl` | 使用 permutation-invariant set target | 身份无序目标实验 |
| `dual_*` | GNN social branch 与 node-set branch 融合 | node 保留消融实验 |
| `mlp_graph_ssl` | 不做图消息传递的 ordered MLP | 诊断 control，不是正式 Social-GNN |

`graph_hybrid_p2` 中的 `p2` 表示 preservation loss weight 为 2，不是两层网络，也不是
两个 patch。

## 13. 训练输出在哪里？

每次正式运行会创建新的 timestamped experiment 目录：

```text
evaluation_runs/<experiment>/training/experiment_<timestamp>/
|- manifest.json
|- normalization.json
|- graph_full/
|  |- best.pt
|  |- last.pt
|  |- metrics.jsonl
|  `- embeddings/
`- graph_hybrid_p2/
   `- ...
```

- `best.pt`：按 validation metric 选择的 checkpoint；
- `last.pt`：最后一个 epoch，不一定最好；
- `metrics.jsonl`：逐 epoch train/validation loss；
- `embeddings/`：保留权威 clock 的每场 `g_t` NPZ；
- `normalization.json`：只由训练 trial 拟合的标准化参数。

不要把 `evaluation_runs`、视频、日志或大型分析结果提交到 GitHub。应该提交的是代码、
YAML、测试和小型 Markdown 报告。

## 14. 如何评估一个 checkpoint？

信息保留 probe 会冻结 encoder，只训练小型 Ridge/MLP probe：

```powershell
python -m social_gnn.evaluate_information_retention `
  --experiment-config configs\social_gnn_full_benchmark_9dim_v1.yaml `
  --experiment-dir <包含各模型子目录的 experiment 目录> `
  --output-dir evaluation_runs\my_probe `
  --validation-trial A4_B4 `
  --test-trial A4_C4 `
  --exclude-trial A5_C5 `
  --native-only `
  --skip-mlp `
  --no-random-control
```

当前常用四项 R2 是：

- node mean；
- node absolute difference；
- all edge；
- directed edge。

它们回答“`g_t` 中还保留多少输入信息”，不能回答“聚类是否对应真实行为”。完整 9D
结果见 [GNN_9DIM_FULL_BENCHMARK_REPORT.md](GNN_9DIM_FULL_BENCHMARK_REPORT.md)。

## 15. 如何进入 HMM/HSMM？

先用冻结的 GNN checkpoint 导出 embedding：

```powershell
python -m social_gnn.downstream extract `
  --experiment-config configs\social_gnn_full_benchmark_9dim_v1.yaml `
  --checkpoint <experiment目录>\graph_full\best.pt `
  --output-dir evaluation_runs\downstream_9dim\graph_full\embeddings `
  --device cuda:1
```

然后搜索 HMM/HSMM：

```powershell
python -m social_gnn.downstream search `
  --embedding-dir evaluation_runs\downstream_9dim\graph_full\embeddings `
  --output-dir evaluation_runs\downstream_9dim\graph_full\search `
  --train-trial A1_C1 --train-trial A3_C3 --train-trial A4_B4 `
  --validation-trial A4_C4 `
  --model hmm --model hsmm `
  --states 4 6 8 10 12 `
  --seed 0 1 2 3 4 `
  --pca-dim 16 `
  --max-duration 100
```

HMM/HSMM state 编号没有天然语义。`state 3` 不自动等于“打架”。必须结合原始 node、
edge、bout 视频和外部标注解释。无效 patch 会切断序列，模型不会跨 trial 或缺口学习
转移。

注意：`configs/social_gnn_downstream_9dim_v1.yaml` 当前记录的是较早的 hybrid checkpoint。
若改用新 benchmark 的 `graph_full`，必须同步更新 experiment config、checkpoint 和 hash，
不能只改显示名称。

## 16. 当前结果应该怎样理解？

截至 2026-10-19：

- `graph_full` 是四项信息保留指标最均衡的模型；
- `graph_hybrid_p2` 是 masked robustness 候选；
- 64D `g_t` 明显优于当前测试的 32D；
- node-only 和 shuffled-edge control 的 edge R2 接近或低于零，说明真实对齐 edge 被
  模型实际使用；
- role-aware readout 有很高的 directed/role retention，但它依赖 identity 顺序，不是
  permutation-invariant canonical representation；
- MLP control 的 edge retention 很高，但没有 GNN message passing，也存在置换误差。

这些结论来自目前五场数据，其中 `A4_C4` 已被用于 encoder checkpoint 选择，所以不是
完全独立的最终测试集。不要把当前排名写成已经证明某个模型能发现最佳生物学行为状态。

## 17. 最常见的错误

### 错误 1：把 MouseGPT 的关键点当成 GNN nodes

不对。当前 node 是鼠，关键点只用于构建 edge。

### 错误 2：看到上游约一秒感受野，就把 patch 改成一秒

不对。当前 social stride 仍为 0.16 秒；相邻特征相关是允许的。

### 错误 3：node 和 edge 长度不同就裁到较短者

不允许。必须找到 clock 或数据缺失的原因。`SocialTrialPackage` 会故意报错。

### 错误 4：只改 identity 标签，不交换 latent stream

这会造成静默的身份错误。先依据 IMU-video 映射重排数据，再写 identity。

### 错误 5：把 confidence 乘进 value 后只保存一个数

不对。value 和 confidence 必须保持独立输入。

### 错误 6：把所有 trial 拼成一条时间序列

不允许。训练 shuffle 以 trial 为单位；HMM/HSMM 也不能跨 trial 学转移。

### 错误 7：只比较 SSL loss

不同 objective 的 loss 定义和尺度不同，不能直接横比。至少同时看冻结 probe、角色
保持、置换误差、embedding geometry 和下游状态质量。

### 错误 8：把 smoke test 当正式结果

smoke test 只验证代码路径，不提供有效的泛化结论。

## 18. 新人接手 checklist

开始改代码前确认：

- [ ] 当前环境是 `deepof`；
- [ ] 已运行 `python -m unittest discover -s tests -v`；
- [ ] 明白一只鼠是一个 node；
- [ ] 明白 edge 是 `[value(8), confidence(8)]`；
- [ ] node/edge 使用相同 identity 顺序和权威 clock；
- [ ] train/validation/test 以完整 trial 切分；
- [ ] 没有把时间模型塞回 GNN encoder；
- [ ] 没有把大型输出加入 Git；
- [ ] 新实验先 smoke test，再正式训练；
- [ ] 对行为状态的命名有视频或标注证据。

## 19. 进一步阅读

- [README.md](README.md)：完整实现与命令参考；
- [GNN_9DIM_FULL_BENCHMARK_REPORT.md](GNN_9DIM_FULL_BENCHMARK_REPORT.md)：最新
  9D GNN benchmark；
- [NODE_PRESERVATION_REPORT.md](NODE_PRESERVATION_REPORT.md)：node 信息保留实验；
- [downstream/README.md](downstream/README.md)：HMM/HSMM 命令与时间契约；
- [downstream/DOWNSTREAM_REPORT.md](downstream/DOWNSTREAM_REPORT.md)：下游状态发现报告；
- `tests/`：最可靠、最短的行为契约示例。

如果代码与文字冲突，先看测试和当前配置，再查看 Git 历史确认设计是否已经更新。
