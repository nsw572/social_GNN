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
  --embedding-dir evaluation_runs\downstream_v1\graph_mean\embeddings `
  --output-dir evaluation_runs\downstream_v1\graph_mean\search `
  --train-trial A1_C1 --train-trial A3_C3 --train-trial A4_B4 `
  --validation-trial A4_C4 `
  --model hmm --model hsmm `
  --states 4 6 8 10 12 --seed 0 1 2 3 4 `
  --pca-dim 16 --max-duration 100
```

Run the unified independent/temporal benchmark:

```powershell
python -m social_gnn.downstream benchmark `
  --embedding-dir evaluation_runs\gnn_9dim_downstream_v1\embeddings\graph_full\seed_20261013 `
  --output-dir evaluation_runs\gnn_9dim_downstream_v1\screening\graph_full\seed_20261013 `
  --experiment-config configs\social_gnn_invariant_mlp_9dim_v1.yaml `
  --encoder-name graph_full --encoder-seed 20261013 `
  --train-trial A1_C1 --train-trial A3_C3 --train-trial A4_B4 `
  --validation-trial A4_C4 `
  --algorithm kmeans --algorithm gmm --algorithm hmm --algorithm hsmm `
  --states 4 6 8 10 12 --seed 0 1 2 --max-duration 25 50 100
```

Use `configs/social_gnn_downstream_benchmark_9dim_v1.yaml` as the authoritative matrix
and limitation record. A4_C4 is reused development/validation data; A5_C5 is
excluded from fitting and selection. Density likelihoods are comparable across
algorithms only while the encoder/PCA observation space is held fixed.

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

The HMM uses diagonal Gaussian emissions and Baum-Welch EM. The HSMM uses the
same emissions, excludes self-transitions, and uses a state-specific shifted
Poisson distribution over explicit segment durations. Its forward/backward,
posterior, and Viterbi algorithms are duration aware; it is not a filtered HMM.

State numbers are arbitrary. Interpret `state_summary.csv` using original node
and social-edge measurements, and do not assign behavioral names without
external annotation or video review.
