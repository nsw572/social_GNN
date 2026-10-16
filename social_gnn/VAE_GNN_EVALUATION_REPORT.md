# VAE-GNN reconstruction and frozen-probe evaluation (2026-10-16)

## Protocol

- Input: historical 6D canonical node packages, so results are directly
  comparable with the existing Hybrid SSL evaluation.
- Encoder training: A1_C1, A3_C3, A4_B4.
- Encoder checkpoint selection: A4_C4 validation total loss.
- A5_C5 excluded because it was previously inspected.
- Three independent seeds: 20261013, 20261029, 20261107.
- Architecture: two-layer 64D SocialV0 GNN, 32D variational posterior mean,
  node and edge self-reconstruction, KL beta 0.001, 100 epochs.
- Frozen Ridge probes: fit on A1_C1 and A3_C3, alpha selected on A4_B4, final
  reported comparison on A4_C4.

The primary VAE representation is deterministic posterior mean `mu` (32D).
The pre-variational graph activation `g_t` (64D) is also probed to localize
information loss.

## Reconstruction

Mean +/- sample SD across seeds at each validation-selected checkpoint:

| Metric | Train | A4_C4 validation |
|---|---:|---:|
| Reconstruction | 0.684 +/- 0.006 | 1.555 +/- 0.033 |
| Node reconstruction | 0.497 +/- 0.002 | 1.374 +/- 0.031 |
| Edge reconstruction | 0.187 +/- 0.005 | 0.181 +/- 0.004 |

Best epochs were 72, 99, and 80. The train-validation gap is almost entirely
in node reconstruction; edge reconstruction transfers without a comparable
gap. Mean validation KL was 3.261, contributing only about 0.0033 to total
loss at beta 0.001.

## Frozen linear probes

Values are mean +/- sample SD across three seeds on A4_C4.

| Representation | Node mean R2 | Node difference R2 | Edge R2 | Directed R2 |
|---|---:|---:|---:|---:|
| VAE `mu` (32D) | 0.960 +/- 0.005 | -0.174 +/- 0.218 | 0.732 +/- 0.018 | 0.543 +/- 0.032 |
| Pre-VAE `g_t` (64D) | 0.966 +/- 0.004 | 0.056 +/- 0.044 | 0.740 +/- 0.020 | 0.552 +/- 0.034 |
| Hybrid SSL reference | 0.925 +/- 0.025 | 0.814 +/- 0.053 | 0.753 +/- 0.005 | 0.585 +/- 0.004 |

## Interpretation

The VAE-GNN is effective at retaining the average node state and much of the
edge signal, but it does not preserve the two-animal node difference. Most of
that loss is already visible in pre-VAE `g_t`; compression to `mu` worsens it.
The likely structural cause is that a permutation-invariant mean-pooled graph
embedding is trained to reconstruct identity-ordered per-animal node vectors.
That target conflicts with the readout symmetry and encourages an average-like
solution. The small KL contribution is not the primary cause.

Under the current objective, VAE-GNN is therefore not a replacement for the
selected Hybrid SSL representation. A defensible next VAE variant would use an
invariant node target (mean plus absolute difference), or an explicit
permutation-matched node-set decoder, before retesting beta or latent width.

These are 6D comparison results. The new 9D upstream latent needs canonical
clock-aligned node packages and fresh Hybrid/VAE training before a 9D model
selection claim can be made.
