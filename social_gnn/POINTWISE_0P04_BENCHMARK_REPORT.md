# 0.04 s pointwise Social-GNN benchmark

Date: 2026-10-23

## Outcome

The 9D Social-GNN and its HMM/HSMM downstream models were rebuilt and rerun on
the new pointwise clock. One social timestep is now exactly one 0.04 s aligned
upstream latent sample, with edge traits reconstructed on the same physical
interval. No 0.16 s four-sample node average is used in this benchmark.

The formal encoder comparison now contains `graph_full`, `graph_hybrid_p2`,
`invariant_mlp_full`, and `invariant_mlp_hybrid_p2`, with three independent
encoder seeds and 100 epochs per condition. The invariant MLP is an explicit
permutation-invariant, non-message-passing control, not a replacement for the
Social-GNN. Frozen probes use A1_C1 and A3_C3 for fitting, A4_B4 for Ridge
selection, and A4_C4 for evaluation. A5_C5 remains excluded.

The formal downstream comparison fixes K=6 and PCA=16. Each encoder condition
uses three encoder seeds crossed with three downstream seeds, giving nine runs
per encoder/algorithm block. HMM uses a 2.0 s dwell target and a calibrated
sticky prior strength of 50000. HSMM uses an explicit maximum duration of 200
timesteps, equal to 8 s at the new clock.

## Rebuilt data

| trial | timesteps | valid two-animal timesteps | node shape | dense edge shape |
| --- | ---: | ---: | --- | --- |
| A1_C1 | 50,792 | 48,716 | `[50792,2,9]` | `[50792,2,2,8]` |
| A3_C3 | 47,012 | 42,969 | `[47012,2,9]` | `[47012,2,2,8]` |
| A4_B4 | 52,284 | 49,833 | `[52284,2,9]` | `[52284,2,2,8]` |
| A4_C4 | 46,128 | 42,797 | `[46128,2,9]` | `[46128,2,2,8]` |
| A5_C5 | 43,804 | 42,389 | `[43804,2,9]` | `[43804,2,2,8]` |

Rows outside latent support and internal invalid samples remain on the physical
trial clock with `node_mask=false`; they are not fabricated or silently
cropped. Downstream temporal models split at every invalid run and never learn
transitions across a gap or trial boundary.

## Frozen information-retention R2

Values are mean +/- sample SD across three encoder seeds. `score` is the
unweighted mean of the four displayed R2 diagnostics, not a training objective.

| encoder | score | node mean R2 | node absolute-difference R2 | all edge R2 | directed edge R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| invariant_mlp_full | 0.7857 | 0.9877 +/- 0.0012 | 0.8731 +/- 0.0065 | 0.7160 +/- 0.0012 | 0.5660 +/- 0.0020 |
| graph_full | 0.7648 | 0.9836 +/- 0.0009 | 0.8613 +/- 0.0053 | 0.6857 +/- 0.0017 | 0.5285 +/- 0.0017 |
| invariant_mlp_hybrid_p2 | 0.7630 | 0.9839 +/- 0.0011 | 0.7886 +/- 0.0115 | 0.7143 +/- 0.0026 | 0.5651 +/- 0.0045 |
| graph_hybrid_p2 | 0.7420 | 0.9812 +/- 0.0014 | 0.7789 +/- 0.0092 | 0.6816 +/- 0.0022 | 0.5263 +/- 0.0016 |

`invariant_mlp_full` is the pointwise benchmark's strongest frozen
information-retention baseline, while `graph_full` remains the strongest actual
message-passing encoder. Relative to the historical 0.16 s benchmark, the
pointwise targets preserve 25 Hz variation instead of averaging four samples.
The comparison is therefore informative about the cost of pointwise
resolution, but it is not an identical-target replication.

## Formal HMM/HSMM metrics

Run-level values below are mean +/- sample SD across nine runs (three encoder
seeds x three downstream seeds). `overall ARI` is the mean of 36 dependent
pairwise comparisons across all nine partitions in one block. `fixed-encoder
ARI` first computes the three downstream-seed comparisons within each encoder
seed, then reports mean +/- sample SD across the three encoder seeds.

| encoder / model | validation fit per point | silhouette | K effective | largest state | bouts/min | median bout s | one-point bouts | learned self-transition | overall ARI | fixed-encoder ARI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| graph_full / sticky HMM | -29.5388 +/- 0.0521 | 0.0487 +/- 0.0041 | 5.8825 +/- 0.0232 | 0.2067 +/- 0.0062 | 81.56 +/- 7.27 | 0.476 +/- 0.065 | 0.0172 +/- 0.0038 | 0.9674 +/- 0.0018 | 0.4562 | 0.7406 +/- 0.2158 |
| graph_hybrid_p2 / sticky HMM | -29.1704 +/- 0.1469 | 0.0621 +/- 0.0078 | 5.8013 +/- 0.0423 | 0.2296 +/- 0.0172 | 81.21 +/- 4.46 | 0.476 +/- 0.024 | 0.0113 +/- 0.0046 | 0.9678 +/- 0.0011 | 0.4127 | 0.5931 +/- 0.0867 |
| invariant_mlp_full / sticky HMM | -27.6293 +/- 0.4547 | 0.0674 +/- 0.0100 | 5.1258 +/- 0.0392 | 0.2579 +/- 0.0203 | 96.23 +/- 5.78 | 0.396 +/- 0.013 | 0.0524 +/- 0.0032 | 0.9638 +/- 0.0011 | 0.5416 | 0.7468 +/- 0.2173 |
| invariant_mlp_hybrid_p2 / sticky HMM | -26.4548 +/- 0.8995 | 0.0739 +/- 0.0035 | 5.0806 +/- 0.0491 | 0.2575 +/- 0.0058 | 91.81 +/- 6.71 | 0.400 +/- 0.035 | 0.0615 +/- 0.0069 | 0.9649 +/- 0.0014 | 0.5458 | 0.8349 +/- 0.1191 |
| graph_full / HSMM Dmax=200 | -29.7166 +/- 0.0503 | 0.0462 +/- 0.0056 | 5.8794 +/- 0.0262 | 0.2144 +/- 0.0118 | 139.13 +/- 49.17 | 0.360 +/- 0.162 | 0.0692 +/- 0.0701 | n/a | 0.4129 | 0.6924 +/- 0.2567 |
| graph_hybrid_p2 / HSMM Dmax=200 | -29.3134 +/- 0.0975 | 0.0621 +/- 0.0075 | 5.7792 +/- 0.0600 | 0.2377 +/- 0.0305 | 126.50 +/- 40.77 | 0.396 +/- 0.136 | 0.0529 +/- 0.0536 | n/a | 0.4038 | 0.5838 +/- 0.0833 |
| invariant_mlp_full / HSMM Dmax=200 | -27.7495 +/- 0.4591 | 0.0608 +/- 0.0104 | 5.2092 +/- 0.0576 | 0.2535 +/- 0.0184 | 120.07 +/- 7.07 | 0.427 +/- 0.028 | 0.0860 +/- 0.0106 | n/a | 0.5226 | 0.7354 +/- 0.2265 |
| invariant_mlp_hybrid_p2 / HSMM Dmax=200 | -26.5945 +/- 0.8870 | 0.0677 +/- 0.0036 | 5.1606 +/- 0.0522 | 0.2557 +/- 0.0074 | 113.94 +/- 8.89 | 0.440 +/- 0.040 | 0.1000 +/- 0.0065 | n/a | 0.5117 | 0.8267 +/- 0.1473 |

All 72 formal downstream runs converged. Every block occupied all six requested
states; K effective stayed between about 5.08 and 5.88 and the largest-state
occupancy stayed below 0.26, so the previous state-collapse failure was not
reproduced. The MLP partitions are less occupancy-balanced than the GNN
partitions, but they are not collapsed. No HSMM run reached its 8 s duration
cap.

## Sticky-prior sensitivity

A controlled Graph Full sensitivity analysis refit 90 unique HMM
configurations while reusing every encoder embedding and its train-fitted
PCA16. At the 2 s dwell target, increasing strength from 10 to 100000 raised
the mean fitted self-transition probability from 0.9034 to 0.9728, reduced
bouts/min from 130.06 to 78.94, and increased median Viterbi bout duration from
0.311 s to 0.493 s. The prior therefore materially controls segmentation.

The underlying K=6 partition is substantially but not perfectly conserved.
Matched-seed ARI against 50000/2 s was 0.821 at strength 10, 0.914 at 25000,
and 0.972 at 100000. The 25000--100000 range is a broad high-strength region.
Within-encoder ARI rose only modestly from 0.701 at strength 10 to 0.741 at
50000, while cross-encoder ARI remained low (0.332--0.364). A stronger temporal
prior therefore does not solve encoder-seed disagreement.

Strength is total Dirichlet pseudocount mass per source-state row, distributed
according to the target transition matrix; it is not diagonal-only mass. At
50000/2 s, exact posterior soft transition counts average 23510 per row and
the prior/data row-mass ratio averages 2.216 (range 1.544--4.461). The chosen
prior has substantial leverage. Nevertheless, 50000/2 s remains the supported
development default: it lies in the stable region, preserves six occupied
states, and no tested alternative offers a compelling joint likelihood and
reproducibility improvement. This is not evidence that 2 s is a biological
dwell-time optimum.

## Interpretation

1. `invariant_mlp_full` is the retention winner, and `graph_full` is the
   strongest message-passing model. Likelihood values must not be compared
   directly across different encoder/PCA observation spaces.
2. The calibrated sticky HMM is the better current temporal default. It has
   fewer bouts per minute, longer median bouts, fewer one-point bouts, higher
   overall ARI, and better within-representation validation fit than HSMM for
   each encoder.
3. Explicit duration alone does not guarantee smoother states. The fitted HSMM
   duration distributions favored relatively short bouts even though Dmax was
   8 s.
4. Fixed-encoder ARI is much higher than overall ARI, especially for
   graph_full. Encoder-seed variation therefore remains a major source of state
   disagreement. Downstream initialization also matters, but less uniformly.
5. The invariant MLP controls have higher overall ARI than the GNNs in this
   benchmark. `invariant_mlp_hybrid_p2` is also the most conditionally stable
   block (fixed-encoder ARI 0.835 for HMM), whereas `invariant_mlp_full` has the
   strongest retention. This is evidence that message passing is not required
   to obtain the present retention/stability scores; it is not evidence that
   the MLP states are more biologically valid.
6. A4_C4 is reused development data and was involved in encoder selection. The
   metrics measure retention, optimization stability, and segmentation
   geometry; they do not establish biological state validity or generalization
   to an untouched animal pair.

## Reproducible artifacts

- Active encoder config: `configs/social_gnn_pointwise_9dim_v2.yaml`
- Active downstream config:
  `configs/social_gnn_downstream_benchmark_9dim_0p04_v2.yaml`
- Rebuilt nodes/edges, training runs, embeddings, and probes:
  `evaluation_runs/gnn_9dim_0p04_v2/`
- Four-R2 table:
  `evaluation_runs/gnn_9dim_0p04_v2/summary_with_mlp/unified_comparison.csv`
- Invariant-MLP encoder config:
  `configs/social_gnn_invariant_mlp_9dim_0p04_v2.yaml`
- Formal sticky-HMM runs:
  `evaluation_runs/gnn_9dim_0p04_downstream_v2/sticky_hmm_strength50000/`
- Formal sticky-HMM aggregate:
  `evaluation_runs/gnn_9dim_0p04_downstream_v2/aggregate_sticky_hmm_strength50000/`
- HSMM Dmax=200 runs:
  `evaluation_runs/gnn_9dim_0p04_downstream_v2/{graph_full,graph_hybrid_p2}/`
- HSMM aggregate:
  `evaluation_runs/gnn_9dim_0p04_downstream_v2/aggregate_hsmm_dmax200/`
- Invariant-MLP HMM/HSMM runs and aggregate:
  `evaluation_runs/gnn_9dim_0p04_downstream_v2/{invariant_mlp_full,invariant_mlp_hybrid_p2,aggregate_invariant_mlp}/`
- Weak-prior calibration controls:
  `evaluation_runs/gnn_9dim_0p04_downstream_v2/sticky_calibration/`
- Full 90-configuration sticky-prior analysis and report:
  `evaluation_runs/gnn_9dim_0p04_downstream_v2/sticky_prior_sensitivity/`

Large `evaluation_runs` artifacts remain intentionally untracked. The config,
implementation, tests, and this report are the repository deliverables.
