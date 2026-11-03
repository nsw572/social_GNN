# Social-GNN downstream state discovery

This package consumes frozen, patch-level `g_t` embeddings. It never trains
the Social-GNN and never adds a neural temporal encoder.

The canonical 9D comparison uses the same train-fitted scaling and PCA-16 for
four downstream algorithms: KMeans, diagonal GMM, diagonal Gaussian HMM, and
explicit-duration diagonal Gaussian HSMM. KMeans/GMM treat patches as
independent during fitting, then map labels back to the untouched social clock.

## Commands

Extract a full-clock embedding package:

```powershell
python -m social_gnn.downstream extract `
  --experiment-config configs\social_gnn_selected_graph_mean_6dim_v1.yaml `
  --checkpoint evaluation_runs\next_phase\selected_graph_mean\best.pt `
  --output-dir evaluation_runs\downstream_v1\graph_mean\embeddings `
  --device cuda:1
```

Run trial-level model selection:

```powershell
python -m social_gnn.downstream search `
  --embedding-dir evaluation_runs\gnn_9dim_0p04_v2\embeddings\graph_full `
  --output-dir evaluation_runs\gnn_9dim_0p04_v2\downstream\graph_full `
  --train-trial A1_C1 --train-trial A3_C3 --train-trial A4_B4 `
  --validation-trial A4_C4 `
  --model hmm --model hsmm `
  --states 4 6 8 10 12 --seed 0 1 2 3 4 `
  --pca-dim 16 --max-duration 200 `
  --hmm-timestep-s 0.04 `
  --hmm-sticky-expected-duration-s 2.0 `
  --hmm-sticky-prior-strength 50000.0
```

Run the unified independent/temporal benchmark:

```powershell
python -m social_gnn.downstream benchmark `
  --embedding-dir evaluation_runs\gnn_9dim_0p04_v2\embeddings\graph_full\seed_20261013 `
  --output-dir evaluation_runs\gnn_9dim_0p04_v2\screening\graph_full\seed_20261013 `
  --experiment-config configs\social_gnn_pointwise_9dim_v2.yaml `
  --encoder-name graph_full --encoder-seed 20261013 `
  --train-trial A1_C1 --train-trial A3_C3 --train-trial A4_B4 `
  --validation-trial A4_C4 `
  --algorithm kmeans --algorithm gmm --algorithm hmm --algorithm hsmm `
  --states 4 6 8 10 12 --seed 0 1 2 --max-duration 100 200 400 `
  --hmm-timestep-s 0.04 `
  --hmm-sticky-expected-duration-s 2.0 `
  --hmm-sticky-prior-strength 50000.0
```

Use `configs/social_gnn_downstream_benchmark_9dim_0p04_v2.yaml` as the active
matrix and limitation record. The earlier `...9dim_v1.yaml` and its outputs are
historical 0.16 s artifacts. A4_C4 is reused development/validation data; A5_C5
is excluded from fitting and selection. Density likelihoods are comparable
across algorithms only while the encoder/PCA observation space is held fixed.

Run the controlled Graph Full sticky-prior sensitivity grid without retraining
the encoder or PCA:

```powershell
python -m social_gnn.downstream.sticky_sensitivity `
  --embedding-root evaluation_runs\gnn_9dim_0p04_v2\embeddings\graph_full `
  --baseline-root evaluation_runs\gnn_9dim_0p04_downstream_v2\sticky_hmm_strength50000\graph_full `
  --output-dir evaluation_runs\gnn_9dim_0p04_downstream_v2\sticky_prior_sensitivity `
  --workers 3
```

The formal grid contains 90 unique HMM configurations: six prior strengths at
a 2 s dwell target plus a 2 x 3 strength/dwell grid, crossed with three encoder
and three HMM seeds. It validates and references the nine matching formal
50000/2 s checkpoints instead of overwriting them. The exported diagnostics
include exact forward-backward expected transition counts, prior/data leverage,
likelihood deltas matched within encoder/PCA space, temporal fragmentation,
baseline-relative state correspondence, and within-/cross-encoder stability.

Decompose the saved K=6 confirmation ARIs without fitting any model:

```powershell
python -m social_gnn.downstream ari-stability `
  --runs-csv evaluation_runs\gnn_9dim_downstream_v1\comparisons\confirmation\all_runs.csv `
  --output-dir evaluation_runs\gnn_9dim_downstream_v1\comparisons\ari_stability_decomposition `
  --trial-id A4_C4 --k 6
```

This audit aligns assignments by `trial_id`, `patch_start_s`, and
`patch_end_s`, then separates fixed-encoder/downstream-seed variation from
cross-encoder comparisons. The resulting pairwise ARIs are dependent
diagnostics, not additive variance components or biological replicates.

Audit source-video timing and state-level HMM correspondence without fitting
any model:

```powershell
python -m social_gnn.downstream consistency-audit --extract-qc
```

The command writes
`evaluation_runs/gnn_9dim_downstream_v1/comparisons/temporal_and_state_consistency/`.
It validates node/edge/embedding/assignment clocks, maps patches and bouts to a
packet-PTS-validated source-video timing grid, produces limited synchronization
contact sheets, and compares all 36 saved HMM K=6 runs at the state level.
Trial-specific offsets are explicit in
`configs/social_gnn_temporal_alignment_v1.yaml`; a missing offset is reported,
never guessed. Hungarian mappings remain diagnostics and do not replace
run-local state IDs.

Export the five longest validated bouts for every occupied state in one saved
HMM K=6 run, with exact 15-frame neutral-gray separators:

```powershell
python -m social_gnn.downstream montage `
  --encoder-condition graph_full --encoder-seed 20261029 `
  --downstream-seed 0 --trial-id A4_C4
```

The exporter resolves the source video and run artifacts from the completed
temporal audit. It uses validated PTS frame intervals rather than idtracker
`frame_id / fps`, does not retrain or relabel anything, and writes traceable
clip/frame manifests plus decode, PTS, frame-count, duration, and gray-frame
QC under `evaluation_runs/gnn_9dim_downstream_v1/video_review/`.

Export patch states, posteriors, bouts, raw-feature state summaries, and plots:

```powershell
python -m social_gnn.downstream export `
  --embedding-dir evaluation_runs\downstream_v1\graph_mean\embeddings `
  --checkpoint evaluation_runs\downstream_v1\selected_hmm\model.npz `
  --preprocessor evaluation_runs\downstream_v1\selected_hmm\preprocessing.npz `
  --output-dir evaluation_runs\downstream_v1\selected_hmm\state_exports `
  --experiment-config configs\social_gnn_selected_graph_mean_6dim_v1.yaml
```

Synthetic recovery checks:

```powershell
python -m social_gnn.downstream synthetic `
  --output evaluation_runs\downstream_v1\synthetic_checks.json
```

## Time and sequence contract

Each trial remains an independent sequence. Invalid patches terminate one run
and begin another; no transition is learned across the gap. Embedding files
retain every authoritative clock row plus `valid_time_mask`.

Bout boundaries are midpoints between adjacent patch centers. The first and
last boundaries extend by half the local center interval. Thus each social
stride contributes time once even when source windows overlap. Both timestep
count and physical seconds are exported. Start/end bouts are marked truncated.

The HMM uses diagonal Gaussian emissions and Baum-Welch EM. On the canonical
0.04 s clock, its default 2.0 s sticky dwell target gives `p_self=0.98`. The
current benchmark uses a Dirichlet transition-prior strength of 50000 per
source-state row. The complete sensitivity sweep found a learned mean diagonal
probability of 0.9674 at that setting. Exact fitted-model diagnostics give about
23510 posterior-expected data transitions per source-state row on average and
a mean prior/data row-mass ratio of 2.216, so this is a consequential MAP prior,
not a negligible initializer. It is applied during initialization and every EM
M-step; it is not post-hoc bout deletion. Always report the learned diagonal
probability and recalibrate the strength if the amount of training data changes
materially. The HSMM uses the same emissions, excludes self-transitions, and
uses a state-specific shifted Poisson distribution over explicit segment
durations. Its forward/backward, posterior, and Viterbi algorithms are duration
aware; it is not a filtered HMM.

State numbers are arbitrary. Interpret `state_summary.csv` using original node
and social-edge measurements, and do not assign behavioral names without
external annotation or video review.
