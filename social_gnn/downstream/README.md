# Social-GNN downstream HMM/HSMM

This package consumes frozen, patch-level `g_t` embeddings. It never trains
the Social-GNN and never adds a neural temporal encoder.

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
