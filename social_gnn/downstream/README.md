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

Run formal state-number selection on the frozen Graph Full and invariant-MLP
Full embeddings/PCA16 spaces:

```powershell
python -m social_gnn.downstream.k_selection `
  --root C:\Users\DELL\Desktop\deepOF `
  --output-dir evaluation_runs\gnn_9dim_0p04_downstream_v2\k_selection_v1 `
  --workers 3
```

The primary matrix is `K={4,5,6,7,8,10,12}` crossed with three encoder seeds,
three HMM seeds, and two architectures (126 configurations). Exact K=6 formal
artifacts are validated and referenced rather than refit. The analysis exports
conventional train AIC/BIC, matched development likelihood deltas, occupancy,
temporal fragmentation, within-/cross-encoder stability, rectangular cross-K
split/merge diagnostics, and per-row prior/data leverage. A limited shortlist
also compares fixed strength 50000 with `50000*6/K`.

Refresh deterministic tables and the report from completed artifacts without
loading or fitting HMM workers:

```powershell
python -m social_gnn.downstream.k_selection `
  --output-dir evaluation_runs\gnn_9dim_0p04_downstream_v2\k_selection_v1 `
  --aggregate-existing
```

The current AIC/BIC and development likelihood curves improve through the
tested upper boundary K=12. K=12 is therefore not an identified interior
optimum. Graph Full favors K=12 on the declared statistical and reproducibility
criteria; invariant MLP Full favors K=12 statistically but K=6 for
cross-encoder reproducibility. See `k_selection_v1/hmm_k_selection_report.md`.

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

## Graph Full HMM K-selection through K=30

The formal extension reuses the completed K<=12 Graph Full artifacts, fits
independent fixed-prior models at K=14,16,18,20,24,27,30, and runs the declared
`50000*6/K` sensitivity control at K=16,20,24,30. It never retrains the encoder
or PCA and never warm-starts a higher-K model from a lower-K fit.

```powershell
python -m social_gnn.downstream.k_selection_k30 --root . --workers 1
```

The resumable output is
`evaluation_runs/gnn_9dim_0p04_downstream_v2/k_selection_k30/`.

## Optional genuine rSLDS pilot

`social_gnn.downstream.rslds` separates the frozen-PCA16 sequence adapter from
its optional backend. The only model accepted under the rSLDS name is a
`lindermanlab/ssm` `SLDS` whose runtime transition class is verified to be
`RecurrentTransitions`; stationary SLDS and AR-HMM models are not fallbacks.

The upstream package compiles extensions, so use the isolated Linux/WSL
environment in `environment/rslds_ssm_linux.yml` rather than changing the
working HMM environment. Audit the active interpreter and run the synthetic
and data-contract smoke tests with:

```powershell
python -m social_gnn.downstream.rslds_pilot --root .
```

After backend validation and the 100 s feasibility fit pass, add `--run-full`
to run the bounded K={4,6,8}, M={4,8} pilot. Whole-sequence smoothed
reconstruction and prefix-conditioned forecasting are separate contracts; an
rSLDS ELBO is never compared numerically with the HMM's exact marginal
likelihood.

## State Atlas v1 (frozen Graph Full / PCA16 / Sticky HMM K=12)

State Atlas is a read-only state-characterization, QC, and video-review layer.
It reproduces the selected frozen assignment, aligns raw edge values,
confidence, coverage, bouts, posterior confidence, and physical timestamps, and
writes one directed `[T,N,N,10]` atlas tensor per trial. It never refits the
encoder, scaler, PCA, or HMM and never assigns biological behavior names.

```powershell
conda run -n deepof python -m social_gnn.downstream state-atlas `
  --config configs\social_gnn_state_atlas_v1.yaml
```

The default output is `evaluation_runs/state_atlas_v1/`. Video clips are
planned using physical seconds (`video_pts = social_time + offset`), not by
assuming one 25 Hz state point is one ~29.97 fps video frame. Rendering is off
by default; the output includes deterministic clip tables and PowerShell/bash
extraction scripts for use when the source videos are mounted.

The frozen atlas can be inspected without refitting any model using exactly
three visualization components: a standalone interactive timeline, a 13-channel
state fingerprint, and a state-by-pose-validity matrix.

```powershell
conda run -n deepof python -m social_gnn.downstream state-atlas-viz `
  --config configs\social_gnn_state_atlas_visualizations_v1.yaml
```

Outputs are written under
`evaluation_runs/state_atlas_v1/visualizations/`. Trait values and confidence
remain separate, missing observations remain explicit, symmetric channels are
counted once, and directed channels retain their `0->1` / `1->0` labels.
