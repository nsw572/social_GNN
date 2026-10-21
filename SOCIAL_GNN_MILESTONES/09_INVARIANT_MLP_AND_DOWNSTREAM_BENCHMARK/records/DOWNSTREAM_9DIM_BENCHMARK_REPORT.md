# 9D Social-GNN Downstream Benchmark: Invariant MLP, Clustering, HMM and HSMM

Date: 2026-10-19

## 1. Executive summary

This development benchmark is complete. It includes a genuinely
permutation-invariant MLP control, six new 100-epoch encoder runs, frozen
64-dimensional embedding exports, a full K=4/6/8/10/12 screening, HSMM
duration-cap sensitivity, and a 3 encoder-seed x 3 downstream-seed
confirmation at K=6. In total, 120 screening and 144 confirmation downstream
runs completed. No fit crashed. Two of nine Graph-Full GMM runs reached their
100-iteration limit without convergence; they remain visible in the results.

Main conclusions:

1. The invariant MLP is genuinely competitive. MLP-Full slightly exceeds
   Graph-Full on the four frozen retention probes (mean score 0.792 versus
   0.784), has exact zero animal-swap error, and has a nearly identical
   parameter count. This does not invalidate the GNN: the GNN retains a
   stronger relational inductive bias, has much higher effective embedding
   rank, and is the architecture that naturally extends beyond two animals.
2. The learned representation materially changes the recovered partitions.
   With the same K=6 downstream algorithm and matched seeds, GNN-versus-MLP
   assignment ARI is only about 0.22-0.30. Representation choice is therefore
   not a cosmetic preprocessing decision.
3. HMM is the most defensible primary downstream model. It roughly halves
   fragmentation relative to KMeans/GMM while preserving approximately
   5.4-5.8 effective states out of six. It also beats HSMM validation
   likelihood within every fixed encoder/PCA space.
4. HSMM does not provide a demonstrated advantage here. HMM and HSMM often
   find similar partitions, but HSMM is usually slightly more fragmented and
   has slightly worse held-out likelihood. Increasing Dmax from 25 to 50 or
   100 barely changes the result, and Dmax=50/100 never forces a bout to end at
   the cap.
5. K=6 is the recommended working value. K=4 remains a useful coarse control
   and K=8 a useful sensitivity analysis. K>=10 increases fragmentation and
   lowers geometric separation without current biological evidence for the
   extra states.
6. Current assignments are suitable for targeted biological/video validation,
   but not yet for claims of behavioral accuracy. A4_C4 is reused development
   data, no untouched trial is available, and multiseed temporal-model ARI is
   moderate rather than near-perfect.

The strongest two-animal empirical candidate is **Invariant MLP Full + HMM,
K=6**. The preferred scalable graph candidate is **Graph Full + HMM, K=6**.
Both should remain alive until an independent trial and manual video review
decide whether the MLP's stronger two-animal diagnostics or the GNN's graph
inductive bias is more useful biologically.

## 2. Methods

### Inputs and clock

Every social timestep uses the authoritative fixed 0.16 s social clock:

```text
node_features      [T, 2, 9]
edge_values        [T, 2, 2, 8]
edge_confidence    [T, 2, 2, 8]
encoder output g_t [T, 64]
```

Edge values and per-trait confidence are concatenated to 16 channels only at
model input. The encoder remains patch-local: no TCN, RNN, Transformer, or
other neural temporal stage was introduced. Invalid patches split sequences,
and no transition is learned across an invalid gap or trial boundary.

The data split is unchanged:

- encoder/downstream fitting: A1_C1, A3_C3, A4_B4;
- reused development/validation: A4_C4;
- excluded from fitting and selection: A5_C5.

No new independent trial was found. A4_C4 was previously used for encoder
checkpoint selection, so all downstream A4_C4 results are development
diagnostics rather than unbiased generalization estimates.

### Encoders and SSL objectives

Four principal frozen representations were evaluated:

| Representation | Encoder | SSL objective |
| --- | --- | --- |
| Graph Full | two-layer GINE Social-GNN | full-input reconstruction |
| Graph Hybrid P2 | two-layer GINE Social-GNN | 2 x full preservation + masked reconstruction |
| Invariant MLP Full | shared symmetrized MLP | full-input reconstruction |
| Invariant MLP Hybrid P2 | shared symmetrized MLP | 2 x full preservation + masked reconstruction |

For the two-animal MLP, the exact construction is

\[
g(H,E)=\frac{F(H,E)+F(PH,PEP^\top)}{2}.
\]

The same MLP parameters are used on the original and consistently swapped
node, directed-edge, confidence, and validity inputs. Self-social edges are
excluded, invalid pairs are zeroed, and the two outputs are averaged. There is
no stochastic dropout in the two symmetry branches. Decoders receive only the
final 64D graph bottleneck.

Each MLP objective was trained for 100 epochs with encoder seeds 20261013,
20261029, and 20261107 using the same normalization, masking, train/validation
split, checkpoint-selection rule, and reconstruction targets as its GNN
counterpart.

### Downstream preprocessing and algorithms

For every encoder and encoder seed independently:

1. scaling was fitted on the three training trials only;
2. PCA-16 was fitted on those scaled training patches only;
3. the fitted transform was applied unchanged to A4_C4;
4. every trial and every invalid-gap-delimited run remained a separate
   sequence.

The same PCA-16 observations were passed to:

- KMeans: Euclidean independent-patch clustering;
- diagonal GMM: probabilistic independent-patch clustering;
- diagonal Gaussian HMM: Gaussian states plus first-order transitions;
- diagonal Gaussian HSMM: Gaussian states plus transitions and a truncated
  shifted-Poisson explicit-duration distribution.

KMeans and GMM labels were mapped back to the original clock without hidden
temporal smoothing. Bout durations use Voronoi boundaries between adjacent
patch centers, so overlapping source windows do not double-count time.

The screening used K={4,6,8,10,12}, one representative encoder seed, and one
downstream seed. HSMM additionally used Dmax={25,50,100} patches. The K=6,
Dmax=50 confirmation used all three encoder seeds and downstream seeds 0, 1,
and 2: nine partitions per encoder/algorithm combination.

Gaussian likelihood, AIC, and BIC are compared only among algorithms operating
in the same encoder-specific PCA space. Their absolute values must not be used
to rank GNN versus MLP embeddings.

### Metrics

- ARI/NMI: label-name-invariant partition agreement across seeds; neither is
  behavioral accuracy.
- K_eff: entropy effective state count. A value near one indicates a dominant
  state; a value near requested K indicates balanced use, not necessarily
  biological correctness.
- largest occupancy: fraction of valid patches in the largest state.
- bouts/min and bout duration: temporal fragmentation/coherence diagnostics.
- silhouette: geometric separation in the fixed PCA space.
- original-measurement signatures: state-conditioned summaries of the 9D node
  latent and eight edge traits with their confidence/coverage.

## 3. Representation comparison

The values below are means across three encoder seeds on A4_C4. The score is
the unweighted mean of the four primary R2 diagnostics.

| Encoder | Score | Node mean R2 | Node difference R2 | Edge R2 | Directed edge R2 | Role R2 | Effective rank | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Graph Full | 0.784 | 0.986 | 0.879 | 0.712 | 0.559 | 0.498 | 16.37 | 54,758 |
| Graph Hybrid P2 | 0.763 | 0.984 | 0.807 | 0.707 | 0.554 | 0.530 | 15.25 | 54,758 |
| Invariant MLP Full | **0.792** | 0.987 | 0.884 | 0.725 | 0.573 | 0.671 | 6.95 | 55,078 |
| Invariant MLP Hybrid P2 | 0.771 | 0.983 | 0.815 | 0.720 | 0.567 | 0.663 | 8.15 | 55,078 |

All four have measured simultaneous-animal permutation error 0. The MLP has
only 0.58% more trainable parameters than the GNN, so its retention result is
not explained by a large capacity advantage. On one RTX 4090 throughput audit
using all 11,532 A4_C4 patches, MLP-Full processed about 625k patches/s versus
528k for Graph-Full and used about 22.7 MB versus 36.3 MB peak allocated CUDA
memory. These very short GPU timings are engineering diagnostics, not robust
hardware benchmarks.

The important counterpoint is geometry: MLP-Full's effective rank is only 6.95
versus Graph-Full's 16.37. The MLP reconstructs the tested inputs well but uses
a more concentrated 64D representation. Neither frozen reconstruction nor
effective rank alone establishes better behavioral states.

## 4. Clustering comparison

Across representations, screening means were:

| Algorithm | K | Silhouette | Mean K_eff | Bouts/min | Median bout (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| KMeans | 4 | 0.128 | 3.51 | 103.7 | 0.36 |
| KMeans | 6 | 0.109 | 5.40 | 135.9 | 0.28 |
| KMeans | 8 | 0.102 | 7.11 | 155.1 | 0.24 |
| KMeans | 12 | 0.094 | 10.45 | 191.7 | 0.16 |
| GMM | 4 | 0.079 | 3.60 | 118.0 | 0.32 |
| GMM | 6 | 0.042 | 5.25 | 150.7 | 0.24 |
| GMM | 8 | 0.020 | 6.83 | 164.2 | 0.24 |
| GMM | 12 | 0.024 | 10.96 | 182.3 | 0.16 |

Independent clustering uses most requested states and does not show the old
6D K_eff≈1.6 collapse. Its weakness is temporal fragmentation: at K=6,
KMeans/GMM produce roughly 136/151 bouts per minute, with many one-patch bouts.
KMeans is the most seed-stable method in the K=6 confirmation, but that high
stability describes a repeatable geometric partition, not a temporally
coherent behavioral segmentation.

GMM gives no clear advantage over KMeans. It is less seed-stable for every
encoder, remains highly fragmented, and two Graph-Full runs did not converge
within 100 iterations. GMM remains useful as the controlled probabilistic
non-temporal baseline.

## 5. HMM/HSMM comparison

K=6 confirmation means aggregate all nine encoder/downstream seed
combinations. Validation fit is shown only to compare GMM/HMM/HSMM within a
row's fixed encoder space.

| Encoder | Model | ARI | K_eff | Largest occupancy | Bouts/min | Median bout (s) | Validation fit/patch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Graph Full | KMeans | 0.538 | 5.90 | 0.212 | 162.2 | 0.27 | n/a |
| Graph Full | GMM | 0.416 | 5.36 | 0.306 | 154.2 | 0.27 | -30.925 |
| Graph Full | HMM | 0.377 | 5.75 | 0.223 | **61.6** | 0.66 | **-30.216** |
| Graph Full | HSMM | 0.363 | 5.78 | 0.220 | 66.9 | 0.68 | -30.287 |
| Graph Hybrid P2 | KMeans | 0.495 | 5.78 | 0.254 | 127.7 | 0.32 | n/a |
| Graph Hybrid P2 | GMM | 0.325 | 5.22 | 0.294 | 154.9 | 0.27 | -30.366 |
| Graph Hybrid P2 | HMM | 0.362 | 5.77 | 0.225 | **59.7** | 0.71 | **-29.674** |
| Graph Hybrid P2 | HSMM | 0.369 | 5.79 | 0.221 | 66.8 | 0.76 | -29.746 |
| Invariant MLP Full | KMeans | 0.561 | 5.07 | 0.308 | 133.6 | 0.32 | n/a |
| Invariant MLP Full | GMM | 0.397 | 4.79 | 0.343 | 137.1 | 0.32 | -28.854 |
| Invariant MLP Full | HMM | **0.457** | 5.48 | 0.261 | **63.8** | 0.55 | **-28.285** |
| Invariant MLP Full | HSMM | 0.441 | 5.53 | 0.255 | 67.6 | 0.68 | -28.364 |
| Invariant MLP Hybrid P2 | KMeans | 0.699 | 4.74 | 0.374 | 112.0 | 0.32 | n/a |
| Invariant MLP Hybrid P2 | GMM | 0.411 | 4.86 | 0.348 | 127.2 | 0.32 | -27.379 |
| Invariant MLP Hybrid P2 | HMM | **0.425** | 5.40 | 0.274 | **55.9** | 0.64 | **-26.682** |
| Invariant MLP Hybrid P2 | HSMM | 0.397 | 5.61 | 0.258 | 59.2 | 0.80 | -26.708 |

Within every representation, HMM has better A4_C4 likelihood and lower
training AIC/BIC than HSMM. HMM also has fewer bouts. HMM and HSMM nevertheless
agree substantially when representation and random seeds are matched: mean
ARI is 0.777 (Graph Full), 0.743 (Graph Hybrid P2), 0.777 (MLP Full), and
0.699 (MLP Hybrid P2). Thus HSMM is usually modifying boundaries around a
similar partition rather than discovering a radically different state system.

At K=6, mean HSMM Dmax sensitivity across encoders was:

| Dmax (patches) | Approx. cap (s) | K_eff | Bouts/min | Median bout (s) | Fraction ending at cap |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 4 | 5.792 | 66.84 | 0.68 | 0.0017 |
| 50 | 8 | 5.788 | 66.30 | 0.68 | 0.0000 |
| 100 | 16 | 5.788 | 66.26 | 0.68 | 0.0000 |

This rules out forced transitions at Dmax as the main explanation for HSMM
fragmentation. Confirmation diagnostics also show that fitted duration means
are typically 1.11-1.23 times observed means for HSMM, whereas HMM geometric
duration expectations range from approximately 1.01 to 1.18 times observed
means. The current shifted-Poisson duration family does not improve the fit
enough to justify its additional complexity.

## 6. Encoder x downstream interaction

The representation has a large downstream effect:

- matched Graph-Full versus MLP-Full ARI is 0.245 for KMeans, 0.275 for GMM,
  0.296 for HMM, and 0.293 for HSMM;
- matched Graph-Hybrid versus MLP-Hybrid ARI is 0.295, 0.220, 0.272, and 0.252
  for the same algorithms.

Therefore, a downstream algorithm cannot erase the encoder's inductive bias.
The MLP's slightly higher frozen-probe R2 is accompanied by better K=6 HMM
seed stability, but not by agreement with the GNN state system. Conversely,
the GNN's higher effective rank does not automatically yield more stable
unlabeled states.

Full objectives are the stronger default in this dataset. MLP-Full has better
HMM stability than MLP-Hybrid P2 (0.457 versus 0.425), and Graph-Full is
slightly better than Graph-Hybrid P2 (0.377 versus 0.362). Hybrid models tend
to be somewhat smoother, but the differences are not large enough to establish
a downstream advantage.

## 7. State signatures

State signatures are exported for every run using permutation-invariant node
summaries and confidence-aware edge summaries. As an illustrative numerical
example only, one Invariant-MLP-Full/HMM/K=6 run separates:

| Run-local state | Patches | Centroid distance | Facing | Radial approach | Pair closing | Motion alignment |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1,307 | 744 | -0.154 | 5.3 | 4.7 | -0.020 |
| 1 | 1,528 | 898 | -0.271 | 11.1 | 16.0 | -0.001 |
| 2 | 2,618 | 590 | 0.161 | 14.7 | 12.4 | 0.020 |
| 3 | 2,191 | 1,072 | -0.222 | -0.2 | -7.3 | -0.001 |
| 4 | 2,116 | 660 | 0.207 | 11.1 | 1.7 | 0.002 |
| 5 | 940 | 721 | 0.008 | 55.8 | -44.4 | -0.087 |

These are clearly different measurement regimes: close/facing configurations,
distant configurations, pair closing/separation, and asymmetric source motion
are all represented. They are not behavior labels. In this example,
centroid/dynamic confidence is generally 0.84-1.00, while nose/facing
confidence is only about 0.28-0.61. Any future biological interpretation must
inspect confidence and source video rather than naming states from means alone.

## 8. Limitations

1. Only four trials participate in fitting/development evaluation.
2. A4_C4 was used during encoder checkpoint selection and is not an untouched
   biological test.
3. A5_C5 was previously inspected and was excluded from fitting, selection,
   and independent-test claims.
4. Patch observations are autocorrelated and may have overlapping upstream
   receptive fields. Patch rows are not independent biological replicates.
5. There are no manually validated social-behavior labels, so ARI, likelihood,
   silhouette, and temporal smoothness are not behavioral accuracy.
6. A dominant background state could be real; conversely, balanced occupancy
   could be artificial. K_eff alone cannot decide this.
7. Gaussian density likelihoods are not calibrated across different learned
   embedding coordinate systems.
8. The MLP construction is exact only for the current two-animal graph. The
   GNN's proposed scalability has not yet been tested with more than two
   animals.
9. Nose/head-derived state characteristics have materially lower confidence
   than centroid-derived traits in some states.
10. HSMM conclusions apply to the current diagonal emissions and truncated
    shifted-Poisson duration family, not every possible semi-Markov model.

## 9. Recommendation

1. Use **HMM with K=6** as the primary downstream model for the next biological
   review round.
2. Carry **Invariant MLP Full** as the strongest empirical two-animal encoder
   and **Graph Full** as the primary graph/scalability candidate. Do not discard
   either before independent/video validation.
3. Keep K=4 as a coarse control and K=8 as a sensitivity check. Do not promote
   K>=10 without external evidence for the extra, shorter-lived states.
4. Keep KMeans and GMM as non-temporal controls. KMeans demonstrates that high
   seed stability can coexist with excessive fragmentation; GMM isolates the
   contribution of Markov transitions.
5. Keep HSMM as a diagnostic control, not the default. Dmax=50 is sufficient,
   but the current duration model has no defensible advantage over HMM.
6. Proceed to biological validation, but treat current state IDs as provisional.
   Review high-confidence representative bouts in video, quantify agreement
   with independent annotations, and add genuinely untouched trials before
   selecting a final encoder/state taxonomy.

## Reproducible artifacts

Version-controlled inputs and code:

- `configs/social_gnn_invariant_mlp_9dim_v1.yaml`
- `configs/social_gnn_downstream_benchmark_9dim_v1.yaml`
- `social_gnn/downstream/independent.py`
- `social_gnn/downstream/benchmark.py`
- `social_gnn/downstream/aggregate.py`
- `social_gnn/evaluate_encoder_cost.py`
- `tests/test_invariant_mlp.py`
- `tests/test_downstream_independent.py`

Generated, intentionally unversioned artifacts:

- `evaluation_runs/gnn_9dim_downstream_v1/invariant_mlp/`
- `evaluation_runs/gnn_9dim_downstream_v1/embeddings/`
- `evaluation_runs/gnn_9dim_downstream_v1/screening/`
- `evaluation_runs/gnn_9dim_downstream_v1/confirmation/`
- `evaluation_runs/gnn_9dim_downstream_v1/comparisons/screening/`
- `evaluation_runs/gnn_9dim_downstream_v1/comparisons/confirmation/`

The confirmation comparison directory contains tidy run, occupancy, bout,
state-measurement, stability, ARI/NMI, contingency, and plot artifacts. The
plot set includes occupancy, K_eff, stability, bout durations, bouts/min,
train/validation fit, chronological sequences, agreement matrices, state
measurement signatures, and an encoder comparison dashboard.

All 122 repository unit tests pass in the `deepof` conda environment.
