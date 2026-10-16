# Hybrid SSL downstream HMM versus HSMM comparison (2026-10-16)

## Protocol

- Encoder: selected Social-GNN Hybrid SSL checkpoint, seed 20261107, epoch 100,
  SHA-256 `7701B3B382D76502EF8F26CDC5F68531E7AA194557482B6DBF3227AE2223F52E`.
- Encoder objective: 2:1 full-input preservation plus masked reconstruction.
- Frozen representation: 64D patch-level `g_t`; no neural temporal encoder.
- Temporal preprocessing: train-only standardization followed by PCA16.
- Temporal train trials: A1_C1, A3_C3, A4_B4.
- Temporal validation: A4_C4. A5_C5 remained excluded from selection.
- Models: diagonal Gaussian HMM and shifted-Poisson explicit-duration Gaussian
  HSMM with `Dmax=100` timesteps (nominally 16 s).
- Grid: K in {4, 6, 8}; temporal seeds 0, 1, 2. HMM used up to 40 EM
  iterations and HSMM up to 20. All 18 fits converged.
- Trials and invalid runs remained independent; no transition crossed a trial
  boundary or invalid clock gap.

## Validation results on A4_C4

Mean and population SD of likelihood are across three temporal seeds. ARI is
the mean pairwise agreement between seed solutions after accounting for label
permutation.

| Model | K | LL/step | Pairwise ARI | Median bout (s) | Bouts/min | Effective states | States >=1% |
|---|---:|---:|---:|---:|---:|---:|---:|
| HMM | 4 | -35.236 +/- 0.006 | 0.886 | 2.187 | 19.67 | 2.20 | 3.0 |
| HSMM | 4 | -35.804 +/- 0.049 | 0.911 | 1.280 | 40.40 | 2.52 | 4.0 |
| HMM | 6 | -34.986 +/- 0.721 | 0.144 | 1.280 | 28.43 | 3.06 | 5.0 |
| HSMM | 6 | -35.180 +/- 0.599 | 0.170 | 1.120 | 47.69 | 3.31 | 5.0 |
| HMM | 8 | -33.768 +/- 1.359 | 0.237 | 1.227 | 19.06 | 2.56 | 4.67 |
| HSMM | 8 | -34.099 +/- 1.200 | 0.216 | 1.280 | 40.77 | 2.98 | 4.67 |

At every matched K, HMM has better mean held-out likelihood and substantially
less switching. HSMM uses somewhat more states, but explicit duration does not
produce longer or cleaner bouts under the present shifted-Poisson fit. Instead,
it approximately doubles the K=4 bout rate and remains highly fragmented at
K=6 and K=8.

Likelihood and train BIC favor larger K, but K=6 and K=8 are not reproducible
across temporal initializations. Their pairwise ARI is only about 0.14-0.24.
K=4 is the only stable region for both model families. Under a criterion that
requires both held-out fit and reproducibility, HMM K=4 is the provisional
choice. HMM K=8 seed 2 has the best single-run likelihood (`-32.188`) but must
not be treated as selected biological states because the K=8 seed solutions do
not agree.

## Exported K=4 comparison

The best-likelihood K=4 temporal seed is seed 1 for both models. Across all
five exploratory trials, its HMM export has 28.21 bouts/min, while HSMM has
39.38 bouts/min. Both occupy all four states. On A4_C4, the two state sequences
have ARI 0.419 and NMI 0.351, so HSMM is not merely a duration-smoothed relabeling
of the HMM solution.

The selected HMM and matched HSMM directories include patch-level state CSV/NPZ
files, bout tables, state summaries in original node/edge units, transition
matrices, representative patches, timelines, duration histograms, and PCA
plots.

## Replication on another Hybrid encoder seed

The same 18-fit grid was also run on Hybrid encoder seed 20261013. HMM again
had better likelihood and fewer bouts at every K. At K=6, HMM reached LL/step
`-33.951`, ARI `0.773`, and 9.40 bouts/min, versus HSMM `-35.619`, ARI `0.551`,
and 60.96 bouts/min. The preference for HMM therefore reproduces across these
two independently trained Hybrid encoders, although the preferred stable K is
encoder-dependent.

## Decision and limitations

Use HMM K=4 as the current reproducible downstream candidate for the selected
Hybrid encoder. Preserve HSMM as a valid negative comparison, not the primary
state sequence. State numbers remain provisional and require video review;
neither likelihood nor temporal stability establishes biological meaning.

This comparison uses the historical 6D canonical node packages so it matches
the trained Hybrid checkpoint. It is not a result for the newly delivered 9D
latents. The 9D comparison requires authoritative clock-aligned canonical node
packages and a newly trained Hybrid encoder.
