# 14 — genuine rSLDS finite pilot 与运行中断审计

时间：2026-10-30 至 2026-11-03。

## 向前跨步

本阶段给 0.04 s Social-GNN 增加了一条**隔离、可选、非主线**的 genuine rSLDS 分支：

```text
frozen Graph Full 64D g_t
  -> train-only scaler / PCA16
  -> genuine recurrent SLDS (lindermanlab/ssm)
  -> discrete states + continuous latent trajectories
```

实现只接受 `ssm.transitions.RecurrentTransitions`，不会把 stationary SLDS、AR-HMM
或普通 HMM 冒充为 rSLDS。由于上游 `ssm` 依赖 Cython/C++ 扩展，运行环境被隔离在
Linux/WSL 的 `social-rslds` conda 环境中；它是 CPU 优化程序，没有 GPU 并行训练路径。

这条分支是动力学模型探索，不替换当前冻结的 sticky HMM / State Atlas 主线，也没有
使用锁定的 `A5_C5`。

## 数据与实验契约

- encoder：`graph_full`，seed `20261013`；选择规则在 rSLDS 结果产生前已固定。
- observation：PCA16，时间步长 `0.04 s`。
- train：`A1_C1`、`A3_C3`、`A4_B4`。
- development：`A4_C4`。
- excluded：`A5_C5`。
- inference：structured-mean-field Laplace EM；正式拟合 50 iterations。
- 预定有限网格：`(K,M)={(4,4),(6,4),(8,4),(6,8),(8,8)}`，每格 3 seeds。
- 权威训练 clock 有 141,518 个有效点；rSLDS adapter 实际建模 141,478 个点，差 40 个，
  已在逐 trial support audit 中显式记录，没有静默补点或跨 trial 拼接。

## 已完成与未完成

当前本地正式产物只支持以下结论：

| cell | seeds | 状态 | 可用产物 |
| --- | --- | --- | --- |
| `K=4, M=4` | 0, 1, 2 | 已形成评估产物 | parameters、A4_C4 states/latents、partial metrics；seed 1/2 另有 model pickle |
| `K=6, M=4` | 0, 1, 2 | 已形成评估产物 | model、parameters、A4_C4 states/latents、partial metrics |
| `K=8, M=4` | 0 | 中断，无正式结果 | feasibility initialization 完成；正式 50-step EM 停在 32/50 |
| `K=6, M=8` | — | 未形成产物 | 无 |
| `K=8, M=8` | — | 未形成产物 | 无 |

因此有限 pilot **尚未完成**，也没有足够证据比较 `K=8,M=4` 与 `K=8,M=8`，更不能据此
推进更大的 rSLDS 网格或宣称优于 sticky HMM。

## K=8 中断审计

单 worker 的 `K=8,M=4,seed=0` 重试于 2026-11-02 18:09:12 启动。日志显示：

- AR-HMM initialization 的 25/25 steps 已完成，约用时 1:06:04；
- 正式 rSLDS EM 运行到 32/50，ELBO 从约 `-6,788,950` 改善到 `-3,800,928`；
- stderr 尾部没有 Python traceback、`MemoryError` 或 `ssm` exception；
- 最后日志时间约 21:08:26，之后 WSL 实例被外部关闭，未执行最终保存。

这证明本次不是模型代码主动抛错，但不能仅凭时间重合断言唯一根因。工作站同时长期运行
Inscopix，并出现过极高内存占用，因此宿主机资源压力是高风险候选；它仍需通过系统级
监控和一次受控单任务重跑确认，不能写成已经证实的因果关系。

后续重跑应保持：单 worker、仅一个 cell、输入全部本地化、独立 stdout/stderr、宿主机与
WSL 内存监控，并优先增加中间 checkpoint / resume 能力，避免三小时运行因外部中断全部
作废。在完成这套保护前，不再并行启动 `K=8` cells。

## 实现与验证

- genuine backend 与数据 adapter：`social_gnn/downstream/rslds.py`
- 原始 pilot：`social_gnn/downstream/rslds_pilot.py`
- finite-pilot completion / selected-cell runner：`social_gnn/downstream/rslds_completion.py`
- 隔离环境：`environment/rslds_ssm_linux.yml`
- 正式配置：`configs/social_gnn_rslds_completed_pilot_0p04_v1.yaml`
- tests：`tests/test_rslds.py`、`tests/test_rslds_completion.py`

为保持 Windows 主测试集可导入，`rslds_completion.py` 的 Unix-only `resource` 模块已改为
可选导入；Windows 下只把 peak RSS 记录为不可用，不改变模型结果。当前仓库测试结果为
`209 passed`。

## 本地产物

- backend audit 与早期 pilot：
  `evaluation_runs/gnn_9dim_0p04_downstream_v2/rslds_pilot_v1/`
- 已完成有限单元：
  `evaluation_runs/gnn_9dim_0p04_downstream_v2/rslds_pilot_v1/completed_finite_pilot/`
- 中断的 K=8 重试日志：
  `evaluation_runs/rslds_k8_retry/k8_m4_seed0_20261102_180912.stderr.log`

大型模型、NPZ、JSON 与日志不复制到 milestone，也不提交 Git；本目录只保存代码状态、
实验契约、证据边界和恢复路线。
