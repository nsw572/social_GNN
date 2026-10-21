# 9D Social-GNN full benchmark

Date: 2026-10-19

## Outcome

The VAE branch has been removed from the active Social-GNN implementation.
The remaining canonical path is patch-local:

`SocialTrialPackage -> GNN -> g_t -> graph/SSL decoder -> loss`

The complete non-redundant GNN experiment matrix was repeated on the aligned
9D upstream latent with three independent seeds. All 63 formal training runs
completed for 100 epochs, followed by frozen information-retention probes,
role-binding probes, and representation-geometry audits.

The best balanced information-retention model is `graph_full`. It has the
highest unweighted mean of the four predefined R2 diagnostics while remaining
an actual message-passing GNN with zero measured animal-permutation error.
This is a retention result, not proof that `graph_full` will yield the best
behavioral states. `graph_hybrid_p2` remains the stronger robustness-oriented
candidate because it combines full-input preservation with masked
reconstruction.

## Data and protocol

- Node input: `[T, 2, 9]`.
- Edge value/confidence input: `[T, 2, 2, 8]` each, concatenated to 16 edge
  channels only at model input.
- Fixed authoritative patch grid: 0.16 s patches, with node and edge clocks
  validated by `SocialTrialPackage`.
- Encoder training trials: `A1_C1`, `A3_C3`, `A4_B4`.
- Encoder checkpoint-selection trial: `A4_C4`.
- Excluded trial: `A5_C5`.
- Seeds: `20261013`, `20261029`, `20261107`.
- Epochs: 100 per condition; 21 conditions per seed.
- Default message width and graph width: 64; the explicit bottleneck ablation
  uses `g_t=32`.

For frozen probes, Ridge models were fit on `A1_C1` and `A3_C3`, tuned on
`A4_B4`, and evaluated on `A4_C4`. Consequently, `A4_C4` is a
model-selection diagnostic, not an untouched biological test set: the frozen
encoder itself was selected on this trial. Patch rows are never treated as
independent experimental replicates; means and sample standard deviations are
computed across the three encoder seeds.

The four primary diagnostics are:

1. node mean R2;
2. absolute node-difference R2;
3. all directed/symmetric edge outputs R2;
4. directed-edge-only R2.

The `score` column below is their unweighted arithmetic mean and is only a
compact retention ranking. It is not a learned objective or biological
metric.

## Three-seed results

| condition | score | node mean R2 | node difference R2 | edge R2 | directed edge R2 | primary role R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `graph_full` | 0.7840 | 0.9860 +/- 0.0016 | 0.8791 +/- 0.0147 | 0.7123 +/- 0.0015 | 0.5589 +/- 0.0032 | 0.4982 +/- 0.0354 |
| `graph_hybrid_p2` | 0.7627 | 0.9836 +/- 0.0012 | 0.8068 +/- 0.0158 | 0.7068 +/- 0.0016 | 0.5535 +/- 0.0022 | 0.5304 +/- 0.0084 |
| `mlp_graph_bottleneck` | 0.7615 | 0.9569 +/- 0.0051 | 0.3714 +/- 0.0147 | 0.8646 +/- 0.0029 | 0.8533 +/- 0.0071 | 0.6369 +/- 0.0221 |
| `graph_hybrid_p1` | 0.7522 | 0.9817 +/- 0.0013 | 0.7673 +/- 0.0086 | 0.7059 +/- 0.0020 | 0.5539 +/- 0.0027 | 0.5267 +/- 0.0279 |
| `graph_set_full` | 0.7513 | 0.9897 +/- 0.0004 | 0.7498 +/- 0.0078 | 0.7108 +/- 0.0029 | 0.5548 +/- 0.0043 | 0.5224 +/- 0.0086 |
| `graph_set_hybrid` | 0.7465 | 0.9839 +/- 0.0007 | 0.7446 +/- 0.0042 | 0.7058 +/- 0.0013 | 0.5517 +/- 0.0021 | 0.5202 +/- 0.0093 |
| `graph_hybrid_p05` | 0.7385 | 0.9792 +/- 0.0016 | 0.7234 +/- 0.0307 | 0.7013 +/- 0.0013 | 0.5503 +/- 0.0014 | 0.5451 +/- 0.0459 |
| `graph_hybrid_p025` | 0.7289 | 0.9774 +/- 0.0015 | 0.6911 +/- 0.0271 | 0.6981 +/- 0.0017 | 0.5489 +/- 0.0013 | 0.5451 +/- 0.0571 |
| `dual_branch_hybrid` | 0.7286 | 0.9735 +/- 0.0012 | 0.7952 +/- 0.0105 | 0.6572 +/- 0.0023 | 0.4885 +/- 0.0050 | 0.4658 +/- 0.0157 |
| `dual_set_hybrid` | 0.7116 | 0.9784 +/- 0.0007 | 0.6973 +/- 0.0174 | 0.6688 +/- 0.0035 | 0.5019 +/- 0.0063 | 0.4770 +/- 0.0081 |
| `dual_joint_hybrid` | 0.7100 | 0.9774 +/- 0.0012 | 0.7080 +/- 0.0072 | 0.6521 +/- 0.0364 | 0.5025 +/- 0.0271 | 0.2459 +/- 0.0146 |
| `graph_bottleneck_mean` | 0.6627 | 0.9751 +/- 0.0027 | 0.4734 +/- 0.0106 | 0.6566 +/- 0.0069 | 0.5456 +/- 0.0041 | 0.5716 +/- 0.0944 |
| `masked_coupled_roleaware` | 0.6522 | 0.7994 +/- 0.0063 | 0.2509 +/- 0.0133 | 0.7371 +/- 0.0164 | 0.8212 +/- 0.0208 | 0.6945 +/- 0.0127 |
| `masked_strict_mean` | 0.6444 | 0.9604 +/- 0.0039 | 0.3511 +/- 0.0058 | 0.7118 +/- 0.0038 | 0.5542 +/- 0.0063 | 0.4795 +/- 0.0712 |
| `masked_coupled_relational` | 0.6302 | 0.9630 +/- 0.0033 | 0.3858 +/- 0.0250 | 0.6365 +/- 0.0171 | 0.5354 +/- 0.0110 | 0.6078 +/- 0.0628 |
| `masked_coupled_mean` | 0.6255 | 0.9606 +/- 0.0073 | 0.3659 +/- 0.0203 | 0.6350 +/- 0.0067 | 0.5407 +/- 0.0105 | 0.5671 +/- 0.0598 |
| `graph_bottleneck_soft_role` | 0.6003 | 0.9247 +/- 0.0088 | 0.4139 +/- 0.0936 | 0.5772 +/- 0.0078 | 0.4854 +/- 0.0048 | 0.5561 +/- 0.0554 |
| `masked_coupled_g32` | 0.5679 | 0.9255 +/- 0.0067 | 0.2761 +/- 0.0046 | 0.5712 +/- 0.0280 | 0.4988 +/- 0.0296 | 0.2738 +/- 0.0421 |
| `node_branch_hybrid` | 0.4454 | 0.9617 +/- 0.0015 | 0.8575 +/- 0.0039 | -0.0174 +/- 0.0003 | -0.0200 +/- 0.0002 | -0.0209 +/- 0.0085 |
| `masked_coupled_node_only` | 0.3496 | 0.9658 +/- 0.0022 | 0.4546 +/- 0.0169 | -0.0108 +/- 0.0009 | -0.0112 +/- 0.0012 | -0.0072 +/- 0.0049 |
| `masked_coupled_shuffled` | 0.3206 | 0.9663 +/- 0.0044 | 0.3438 +/- 0.0098 | -0.0132 +/- 0.0041 | -0.0147 +/- 0.0048 | 0.5263 +/- 0.0668 |

## Interpretation

1. `graph_full` is the balanced retention winner. Its A4_C4 entropy effective
   rank is `16.37 +/- 0.05`, total embedding variance is `24.45`, and measured
   simultaneous-animal-permutation error is zero. There is no evidence that
   its high probe scores are caused by embedding collapse.

2. Reducing `g_t` from 64 to 32 is harmful in the current architecture.
   `masked_coupled_g32` falls to a 0.5679 four-metric score and an effective
   rank of only `3.81 +/- 0.23`. The 32D run is therefore an ablation, not the
   recommended default.

3. Real edges are being used. The node-only and time-shuffled-edge controls
   have near-zero or negative held-out edge R2, while full aligned-edge models
   are strongly positive. This argues against the edge decoder merely
   recovering trial averages from node latents.

4. The ordered MLP control is strong on edge retention but is not a valid
   replacement for the Social-GNN. It has no graph message passing and has a
   mean animal-permutation error of about 0.236 on A4_C4.

5. `masked_coupled_roleaware` preserves directed role information unusually
   well, but its readout is deliberately identity-order-sensitive. Its mean
   permutation error is about 0.708, and node information retention is much
   worse. It is useful as a diagnostic upper bound for explicit identity order,
   not as the canonical permutation-invariant social representation.

6. Increasing the preservation term in the graph hybrid objective improves
   node-difference retention monotonically over the tested weights. Among the
   hybrid variants, `p2` is the current choice.

7. Frozen reconstruction probes measure information availability, not
   behavioral separability. Choosing between `graph_full` and
   `graph_hybrid_p2` should ultimately use downstream HMM/HSMM stability and
   biological state interpretation on additional held-out trials.

## Reproducible artifacts

- Experiment matrix: `configs/social_gnn_full_benchmark_9dim_v1.yaml`.
- Formal training runs: `evaluation_runs/gnn_9dim_full_v1/training/`.
- Frozen probe outputs: `evaluation_runs/gnn_9dim_full_v1/probes/`.
- Role-binding outputs: `evaluation_runs/gnn_9dim_full_v1/role_binding/`.
- Geometry outputs: `evaluation_runs/gnn_9dim_full_v1/geometry/`.
- Three-seed unified table:
  `evaluation_runs/gnn_9dim_full_v1/summary/unified_comparison.csv`.

The `evaluation_runs` artifacts are intentionally not version-controlled. The
configuration, implementation, tests, and this report are the reproducible
repository deliverables.
