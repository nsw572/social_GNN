# 9D Social-GNN: Decomposing Encoder and Downstream Clustering Instability

Date: 2026-10-20

## 1. Executive summary

This is a post-hoc analysis of the existing 144 K=6 confirmation runs. No encoder or downstream model was retrained.

**Direct answer:** instability is associated primarily with changing the learned representation, but the probabilistic downstream models also contribute substantial instability. Across all 16 encoder-condition x algorithm blocks, mean ARI was **0.762** when the encoder was fixed and only the downstream seed changed, versus **0.332** across independently trained encoder seeds. Cross-encoder matched-seed ARI (0.331) and different-seed ARI (0.333) were nearly identical, so preserving the numerical downstream seed did not rescue agreement after the representation changed. This gap is descriptive, not an additive or causal variance decomposition.

- **KMeans:** fixed-encoder mean ARI 0.987; cross-encoder matched-seed 0.435; both seed identifiers different 0.435.
- **GMM:** fixed-encoder mean ARI 0.733; cross-encoder matched-seed 0.269; both seed identifiers different 0.274.
- **HMM:** fixed-encoder mean ARI 0.664; cross-encoder matched-seed 0.319; both seed identifiers different 0.319.
- **HSMM:** fixed-encoder mean ARI 0.662; cross-encoder matched-seed 0.302; both seed identifiers different 0.303.

KMeans makes the distinction clearest: it was almost deterministic for a fixed embedding (mean ARI 0.987), yet fell to about 0.435 across encoder seeds. GMM/HMM/HSMM were less stable even conditionally (0.733/0.664/0.662), so their observed instability reflects **both** encoder-seed variation and downstream local optima. Conditional fixed-representation agreement averaged **0.715** for the two GNN conditions and **0.808** for the two invariant-MLP conditions. The invariant MLP is therefore more conditionally stable in this benchmark, not universally seed-invariant.

The largest within-minus-cross gap occurred for Graph Hybrid P2 / KMeans (0.655); the smallest was Graph Hybrid P2 / HMM (0.260).

Graph Full/GMM encoder 20261107, downstream 0; Graph Full/GMM encoder 20261107, downstream 1 were retained and marked nonconverged. No run was removed because of convergence status. Stability remains a property of partitions on reused development trial A4_C4, not behavioral accuracy or untouched biological generalization.

## 2. Experimental setup and alignment

For each of four encoder conditions and four algorithms, the confirmation grid contains three encoder seeds (`20261013`, `20261029`, `20261107`) crossed with downstream seeds (`0`, `1`, `2`). Each 3 x 3 block contributes exactly 36 unique pairs: 9 fixed-encoder/different-downstream, 9 cross-encoder/matched-downstream-seed, and 18 pairs where both identifiers differ.

Assignments were aligned by `(trial_id, patch_start_s, patch_end_s)`, never by row order alone. Alignment integrity status: **passed**. Every pair retained 10700 to 10700 mutually valid physical patches; the minimum retained fraction of the valid union was 1.000000. Invalid patches and gaps were excluded rather than interpolated, and no padded timestep was introduced.

Using the same numerical downstream seed in two different PCA/embedding spaces does not guarantee equivalent initialization. The matched-seed group is therefore a controlled cross-encoder reproducibility diagnostic, not a pure causal encoder effect.

## 3. Overall ARI results

Values are mean +/- sample SD across dependent pairwise comparisons. They are descriptive; no naive pair-level p-values or confidence intervals are reported.

| Encoder | Algorithm | Fixed-encoder ARI | Cross-encoder matched-seed ARI | Cross-encoder different-seed ARI |
| --- | --- | ---: | ---: | ---: |
| Graph Full | KMeans | 0.985 +/- 0.009 | 0.389 +/- 0.058 | 0.389 +/- 0.056 |
| Graph Full | GMM | 0.821 +/- 0.253 | 0.281 +/- 0.045 | 0.281 +/- 0.043 |
| Graph Full | HMM | 0.613 +/- 0.298 | 0.303 +/- 0.100 | 0.296 +/- 0.076 |
| Graph Full | HSMM | 0.599 +/- 0.299 | 0.286 +/- 0.084 | 0.283 +/- 0.069 |
| Graph Hybrid P2 | KMeans | 0.986 +/- 0.009 | 0.331 +/- 0.044 | 0.331 +/- 0.043 |
| Graph Hybrid P2 | GMM | 0.580 +/- 0.252 | 0.224 +/- 0.046 | 0.249 +/- 0.075 |
| Graph Hybrid P2 | HMM | 0.557 +/- 0.196 | 0.289 +/- 0.032 | 0.300 +/- 0.085 |
| Graph Hybrid P2 | HSMM | 0.580 +/- 0.179 | 0.293 +/- 0.048 | 0.301 +/- 0.086 |
| Invariant MLP Full | KMeans | 0.987 +/- 0.008 | 0.419 +/- 0.055 | 0.419 +/- 0.053 |
| Invariant MLP Full | GMM | 0.691 +/- 0.234 | 0.299 +/- 0.091 | 0.299 +/- 0.083 |
| Invariant MLP Full | HMM | 0.727 +/- 0.249 | 0.369 +/- 0.112 | 0.367 +/- 0.095 |
| Invariant MLP Full | HSMM | 0.714 +/- 0.250 | 0.352 +/- 0.100 | 0.349 +/- 0.083 |
| Invariant MLP Hybrid P2 | KMeans | 0.990 +/- 0.008 | 0.602 +/- 0.061 | 0.602 +/- 0.059 |
| Invariant MLP Hybrid P2 | GMM | 0.840 +/- 0.215 | 0.272 +/- 0.079 | 0.266 +/- 0.069 |
| Invariant MLP Hybrid P2 | HMM | 0.758 +/- 0.289 | 0.314 +/- 0.062 | 0.314 +/- 0.061 |
| Invariant MLP Hybrid P2 | HSMM | 0.754 +/- 0.294 | 0.276 +/- 0.023 | 0.278 +/- 0.026 |

Detailed medians, quartiles, extrema and counts are in `ari_group_summary.csv`; all 576 traceable comparisons are in `all_pairwise_ari.csv`.

## 4. Within-encoder downstream stability

The fixed-encoder group asks whether a downstream method repeatedly recovers the same partition from exactly the same saved representation. The lowest seed-specific conditional results were:

- Graph Full / HMM / encoder seed 20261013: 0.388 mean ARI.
- Graph Full / HSMM / encoder seed 20261013: 0.391 mean ARI.
- Graph Full / HSMM / encoder seed 20261107: 0.423 mean ARI.
- Graph Hybrid P2 / GMM / encoder seed 20261013: 0.434 mean ARI.
- Graph Hybrid P2 / HMM / encoder seed 20261013: 0.455 mean ARI.
- Graph Full / HMM / encoder seed 20261107: 0.458 mean ARI.
- Graph Hybrid P2 / HSMM / encoder seed 20261013: 0.463 mean ARI.
- Graph Full / GMM / encoder seed 20261107: 0.485 mean ARI.

The complete 48-row encoder-seed breakdown is in `within_encoder_seed_summary.csv`. Low values here indicate that downstream initialization/local optima remain important even before changing the encoder. High values here combined with lower cross-encoder values point instead to representation reproducibility as the larger concern.

The instability is strongly encoder-seed specific rather than uniform. For Graph Full temporal models: HMM seed 20261013=0.388; HMM seed 20261029=0.995; HMM seed 20261107=0.458; HSMM seed 20261013=0.391; HSMM seed 20261029=0.983; HSMM seed 20261107=0.423. Thus one Graph Full encoder seed (`20261029`) yielded almost identical HMM/HSMM partitions across downstream starts, whereas seeds `20261013` and `20261107` did not. Conversely, Invariant MLP Hybrid P2 seed `20261107` was essentially deterministic for GMM/HMM/HSMM. These patterns indicate an encoder-by-downstream interaction: some learned geometries create much easier optimization landscapes for the same downstream implementation.

## 5. Across-encoder stability

Both cross-encoder groups compare independently trained representations. The matched downstream seed identifier controls the recorded integer only; it does not establish equivalent initialization in changed embedding spaces. `cross_encoder_seed_summary.csv` preserves results separately for each matched downstream seed and each unordered different-seed stratum.

The descriptive deltas in `stability_differences.csv` report fixed-encoder mean ARI minus (a) matched-seed cross-encoder ARI and (b) all cross-encoder ARIs. They must not be interpreted as a percentage of instability caused by the encoder.

## 6. Algorithm comparison

- **KMeans:** fixed-encoder mean ARI 0.987; cross-encoder matched-seed 0.435; both seed identifiers different 0.435.
- **GMM:** fixed-encoder mean ARI 0.733; cross-encoder matched-seed 0.269; both seed identifiers different 0.274.
- **HMM:** fixed-encoder mean ARI 0.664; cross-encoder matched-seed 0.319; both seed identifiers different 0.319.
- **HSMM:** fixed-encoder mean ARI 0.662; cross-encoder matched-seed 0.302; both seed identifiers different 0.303.

KMeans, GMM, HMM and HSMM therefore need to be judged separately: temporal coherence and held-out likelihood from the main benchmark do not guarantee seed-stable state identities. Heatmaps expose whether disagreement is blockwise between encoder seeds or dispersed within each fixed-encoder block.

## 7. Stability versus degeneracy

All requested occupancy, effective-state, fragmentation, silhouette and convergence fields are preserved in `run_quality_metrics.csv` and joined pairwise in `ari_vs_occupancy.csv`. Every run occupied 6 to 6 states; entropy-effective state count ranged from 4.121 to 5.962, and largest-state occupancy from 0.192 to 0.436. The old K_eff around 1.6 collapse was therefore not reproduced in this 9D confirmation grid. Graph Full/GMM encoder 20261107, downstream 0; Graph Full/GMM encoder 20261107, downstream 1 were retained and marked nonconverged.

Low-ARI pairs did coincide with larger differences in state geometry and fragmentation. For the 206 pairs with ARI < 0.3, mean absolute K_eff difference was 0.255 and mean bouts/min difference was 13.68; for the 106 pairs with ARI >= 0.6, those values were 0.022 and 1.78. No pair differed in the number of occupied states, so disagreement concerned occupancy balance, geometry and temporal boundaries rather than outright missing states.

The largest absolute exploratory Spearman association was -0.689 for `abs_bouts_per_minute_difference` in scope `encoder:invariant_mlp_full`. These overlapping pairs are not independent, so this is a diagnostic association only.

High ARI can coexist with an uninformative or degenerate partition; low ARI can reflect label-independent boundary/state-geometry changes without proving biological failure. The recent K_eff values must therefore be read alongside largest-state occupancy, smallest occupied fraction, bouts/min, median duration and convergence.

## 8. Interpretation and limitations

- **Encoder-seed sensitivity** is evidenced when fixed-encoder agreement is high but cross-encoder agreement is lower.
- **Downstream-seed sensitivity** is evidenced directly by low fixed-encoder agreement.
- **Joint variation** is summarized by the 18-pair group where both identifiers differ, but is not an interaction-effect estimate.
- Every pair reuses runs and the same A4_C4 patches, so the ARIs are dependent diagnostics, not biological replicates.
- A4_C4 was already used for encoder checkpoint selection. Nothing here estimates generalization to an untouched animal pair or experiment.
- State labels were not matched before ARI; ARI is already invariant to label permutations.
- The optional dense co-assignment analysis was not run because it was unnecessary for the complete ARI decomposition and would add a second dependent-pair diagnostic without changing the primary question.

## 9. Actionable recommendations

| Observation | Recommended next step |
| --- | --- |
| Low within-encoder HMM ARI | Inspect HMM initialization/local optima and K before changing the encoder. |
| High within-encoder, low cross-encoder ARI | Prioritize encoder representation reproducibility and compare state signatures across encoder seeds. |
| Strong KMeans but weak HMM stability | Inspect probabilistic temporal-model initialization and emission/transition assumptions. |
| Stability associated with degenerate occupancy | Investigate state collapse or background-state dominance before biological naming. |
| Nonconverged GMM fits | Keep them visible, increase diagnostic scrutiny, and do not use their ARI as evidence of a clean optimum. |
| High stability across both factors | Proceed to blinded video review and a genuinely independent trial; do not equate stability with validity. |

The next action should be chosen from the observed block-specific pattern rather than automatically retraining models. The heatmaps in `heatmaps/` and pair dots in `distributions/` are the fastest way to identify seed-specific failures.
