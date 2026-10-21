# 10 — ARI 稳定性分解

时间：2026-10-20。

## 实验问题

既有 9D downstream benchmark 的状态划分并不完全稳定。这一步不重训任何模型，而是利用已经保存的 144 个 K=6 confirmation assignments，区分三类一致性：

1. 固定 encoder，只改变 downstream seed；
2. 改变 encoder seed，但保持 downstream seed 数字相同；
3. encoder 与 downstream seed 都改变。

每个 encoder-condition/algorithm 的 3×3 grid 产生严格的 9/9/18 对比较，共 36 对；16 个 blocks 合计 576 个 ARI。

## 核心结论

- 总体 fixed-encoder ARI 为 0.762，cross-encoder ARI 为 0.332：不稳定主要与 learned representation 改变相关。
- matched downstream seed 与 different downstream seed 的跨 encoder ARI 分别为 0.331 和 0.333；相同 seed 数字无法挽救不同 embedding space 的一致性。
- KMeans 在固定表示上接近确定性（0.987），跨 encoder 降至约 0.435，是 representation sensitivity 最清楚的证据。
- GMM/HMM/HSMM 固定表示时也只有 0.733/0.664/0.662，因此概率 downstream 自身的 local-optimum instability 仍然重要。
- invariant MLP 的 fixed-representation stability（0.808）高于 GNN（0.715），但并非 seed-invariant。
- instability 明显依赖具体 encoder seed：例如 Graph Full seed `20261029` 的 HMM/HSMM 块内 ARI 接近 1，而另外两个 seeds 明显更低。
- 144 个 runs 全部占用 6 个 states，K_eff 为 4.121–5.962，没有重现旧 6D 的 K_eff≈1.6 collapse。
- 两个 Graph Full/GMM seed `20261107` runs 未收敛但被原样保留，没有静默删除。

这不是严格的方差分解，也不能把 ARI 差值解释成“encoder 导致了多少百分比不稳定”。A4_C4 仍是复用开发数据，稳定性不等于行为正确性。

## 本目录快照

- [完整 ARI stability 报告](records/ARI_STABILITY_DECOMPOSITION_REPORT.md)

## 关键实现和产物

- 实现：`social_gnn/downstream/ari_stability.py`
- 测试：`tests/test_ari_stability.py`
- 本地产物：`evaluation_runs/gnn_9dim_downstream_v1/comparisons/ari_stability_decomposition/`
- 产物包括 576-row pairwise dataset、alignment/grid audit、三类 summary、run quality、occupancy/fragmentation 关联、16 张 9×9 heatmaps 和 16 张 pair-distribution plots。

没有执行 encoder/downstream retraining，也没有进行 label matching；ARI 本身已经对 state label permutation 不变。
