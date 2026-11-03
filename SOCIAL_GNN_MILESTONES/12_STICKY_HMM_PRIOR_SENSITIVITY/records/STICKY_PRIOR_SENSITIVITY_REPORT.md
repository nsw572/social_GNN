# 0.04 s Social-GNN: Sticky HMM Prior Sensitivity, State Stability, and Temporal Segmentation

## 1. Executive summary

All 90 unique configurations completed; 36 runs were validated existing artifacts or resumptions. The analysis reused frozen Graph Full embeddings and each encoder seed's saved PCA16; no encoder or PCA was retrained.

The prior materially controls temporal segmentation: increasing strength from 10 to 100000 changed mean learned self-transition from 0.9034 to 0.9728, bouts/min from 130.06 to 78.94, and median bout duration from 0.311 s to 0.493 s.

The six-state partition is substantially, but not perfectly, conserved. Matched-seed ARI against the current baseline is 0.821 at strength 10, 0.914 at 25000, and 0.972 at 100000. Thus 25000-100000 is a broad high-strength region rather than evidence for one sharply tuned optimum.

At the current 50000/2 s setting, mean baseline-relative ARI is exactly 1.000 by definition, within-encoder ARI is 0.741, and cross-encoder ARI is 0.361. The highest within-encoder stability occurred at strength 100000 (ARI 0.743); the best mean development likelihood delta occurred at strength 50000 (+0.0000 per point versus 50000/2 s).

Changing the dwell target at strength 50000 has a smaller effect than weakening the prior: 1 s and 4 s targets retain baseline ARI 0.930 and 0.969, respectively. The 1 s target improves mean development log-likelihood by +0.0073 per point but lowers within-encoder ARI to 0.715; the 4 s target changes likelihood by +0.0015 and leaves within-encoder ARI at 0.740.

Stronger regularization improves conditional HMM reproducibility modestly but does not resolve representation variation: within-encoder ARI rises from 0.701 at strength 10 to 0.741 at 50000, whereas cross-encoder ARI remains only 0.361.

The six-state solutions are not evaluated as biological ground truth. ARI, likelihood, occupancy and fragmentation are complementary diagnostics, and A4_C4 remains reused development data.

## 2. Transition-prior implementation audit

The HMM uses a row-wise Dirichlet pseudocount prior in every EM M-step. Strength is total sticky mass per row, distributed according to the target transition matrix; it is not diagonal-only mass. Forward-backward soft expected transition counts are added independently for every valid run, then rows are normalized. See `prior_implementation_audit.md` and `transition_prior_vs_counts.csv` for the exact audit.

At 50000/2 s, each row receives 50000.06 total prior pseudocounts (50000 sticky plus six 0.01 isotropic counts). The fitted models contain a mean 23510.3 posterior-expected data transitions per source-state row. Prior/data row ratios average 2.216 and range from 1.544 to 4.461; the prior therefore has substantial, state-dependent leverage rather than acting as a negligible initializer.

## 3. Experiment A: strength sensitivity

| strength | learned pself | bouts/min | median bout s | dev LL delta | ARI vs 50000 | within ARI | cross ARI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10.0000 | 0.9034 | 130.0597 | 0.3111 | -0.0476 | 0.8207 | 0.7007 | 0.3318 |
| 1000.0000 | 0.9085 | 127.8750 | 0.3111 | -0.0480 | 0.8223 | 0.6993 | 0.3317 |
| 10000.0000 | 0.9418 | 98.6596 | 0.4000 | -0.0267 | 0.8557 | 0.7220 | 0.3528 |
| 25000.0000 | 0.9581 | 88.5381 | 0.4311 | -0.0392 | 0.9138 | 0.7410 | 0.3641 |
| 50000.0000 | 0.9674 | 81.5556 | 0.4756 | 0.0000 | 1.0000 | 0.7406 | 0.3613 |
| 100000.0000 | 0.9728 | 78.9386 | 0.4933 | -0.0090 | 0.9720 | 0.7432 | 0.3633 |

## 4. State partition robustness

`baseline_partition_correspondence.csv` reports matched-seed comparisons against 50000/2 s, while `pairwise_strength_ari.csv` and the seed-specific heatmaps expose the full strength-by-strength geometry. Hungarian matches are used only to summarize overlap quality; they are not treated as proof that two states are identical. Split/merge flags use a documented 20% substantial-component threshold.

## 5. Probability fit and occupancy

Across the strength sweep, entropy-effective state count ranged from 5.822 to 5.884, and mean largest-state occupancy ranged from 0.206 to 0.211. Development likelihood changes are always calculated within the same encoder seed, HMM seed, and frozen PCA space before aggregation.

## 6. Random-seed decomposition

Across the six 2 s strength settings, matched-downstream-seed cross-encoder ARI ranged from 0.342 to 0.376. Within-encoder and cross-encoder comparisons are kept separate because overlapping ARI pairs are descriptive and not independent biological replicates.

## 7. Experiment B: dwell-target sensitivity

| strength | target dwell s | learned pself | dev LL delta | bouts/min | ARI vs baseline | within ARI | cross ARI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10000.0000 | 1.0000 | 0.9340 | -0.0261 | 102.2891 | 0.8578 | 0.7301 | 0.3527 |
| 10000.0000 | 2.0000 | 0.9418 | -0.0267 | 98.6596 | 0.8557 | 0.7220 | 0.3528 |
| 10000.0000 | 4.0000 | 0.9464 | -0.0327 | 95.9725 | 0.8906 | 0.7307 | 0.3617 |
| 50000.0000 | 1.0000 | 0.9528 | 0.0073 | 84.6282 | 0.9297 | 0.7152 | 0.3511 |
| 50000.0000 | 2.0000 | 0.9674 | 0.0000 | 81.5556 | 1.0000 | 0.7406 | 0.3613 |
| 50000.0000 | 4.0000 | 0.9747 | 0.0015 | 79.8459 | 0.9687 | 0.7402 | 0.3607 |

## 8. Limitations

- A4_C4 has been reused for encoder development and prior calibration; it is not an untouched test.
- Temporal patches and pairwise ARI comparisons are correlated, so no naive patch-level or pair-level inferential statistics are reported.
- Absolute Gaussian log densities are not compared across encoder/PCA spaces; only within-seed deltas are aggregated.
- Stability, fewer bouts, and uniform occupancy do not establish behavioral correctness.
- No independent behavioral labels or new animal pair are available for biological validation.

## 9. Recommendation

Retain the current **50000 / 2 s** setting for the next development stage. It lies inside the 25000-100000 partition-stability plateau, preserves all six occupied states, gives substantially less fragmentation than weak priors, and has within-encoder ARI 0.741. Strength 100000 provides no meaningful stability gain and reduces development likelihood by -0.0090 per point. A 1 s target has slightly better development likelihood but lower within-encoder reproducibility, while 4 s is nearly partition-equivalent without a compelling advantage. This is a development-setting recommendation, not a biological optimum; encoder-seed instability remains the larger unresolved limitation.

Reproducible run-level evidence is in `strength_sweep_runs.csv`, `dwell_grid_runs.csv`, `transition_prior_vs_counts.csv`, `baseline_partition_correspondence.csv`, `within_encoder_stability.csv`, and `cross_encoder_stability.csv`.
