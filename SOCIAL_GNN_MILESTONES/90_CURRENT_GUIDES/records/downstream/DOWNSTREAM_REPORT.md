# Gaussian HMM / HSMM downstream development report

Date: 2026-10-15

## Scope and data contract

The downstream pipeline was implemented and evaluated with the Social-GNN
fully frozen. No neural temporal encoder was introduced. Five trials were
exported from both the selected graph-level SSL encoder and the preserved
strict SSL encoder. Model selection used only:

- Train: `A1_C1`, `A3_C3`, `A4_B4`.
- Validation: `A4_C4`.
- Exploratory only: `A5_C5`, which had already been inspected previously.

The five trials contain 59,969 valid patch rows in total; the
training trials contain 37,486 valid rows and validation contains 11,532. The
authoritative grid in these files has a median patch duration and stride of
0.16 s. A1_C1 has four invalid patches and A3_C3 has 32. Those rows remain in
the saved embedding packages but split the temporal input into independent
runs. No transition is created across an invalid gap or a trial boundary.

Each embedding NPZ contains the full chronological `[T,D]` representation,
authoritative start/end/center timestamps, `valid_time_mask`, encoder path,
configuration, and checkpoint hash. Nonfinite embeddings are rejected.

Bout boundaries use Voronoi boundaries between patch centers. Consequently,
each stride contributes physical time exactly once even if future source
patches overlap. For a constant 0.16 s stride, an interior bout of `n` patches
has duration `n * 0.16 s`; source-window duration is not multiplied by `n`.

## Implemented models

The Gaussian HMM uses state-specific diagonal Gaussian emissions, covariance
flooring, Baum-Welch EM over multiple independent sequences, posterior
inference, and Viterbi decoding.

The Gaussian HSMM is a genuine explicit-duration model. It excludes
self-transitions and uses a separately fitted shifted-Poisson distribution for
each state's duration, truncated at configurable `Dmax`. Its forward,
backward, posterior, EM, and segmental Viterbi recursions all include duration
probabilities. It is not an HMM followed by median filtering.

Both models were initialized reproducibly from KMeans/HMM fits. Standardization
and PCA were fitted on training trials only.

## Synthetic verification

The HMM forward and Viterbi recursions and the HSMM forward, posterior, and
duration-aware Viterbi recursions match exact enumeration on tiny sequences.
Additional variable-length synthetic trials gave:

| Scenario | HMM ARI | HSMM ARI | HMM LL/step | HSMM LL/step |
|---|---:|---:|---:|---:|
| Well-separated Gaussian emissions | 1.000 | 1.000 | -1.972 | -1.892 |
| Overlapping emissions with non-geometric durations | 0.466 | 0.700 | -2.997 | -2.963 |
| Extreme-overlap stress test | 0.063 | 0.016 | -2.903 | -2.905 |

Thus explicit duration helps in an identifiable overlapping-emission setting,
but neither model can recover deliberately near-indistinguishable emissions.
The failed stress test is retained; synthetic success is not evidence of
biological validity.

## Embedding preprocessing

All 64 dimensions were tested first. The full-dimensional HMM was numerically
finite and converged, but produced a poor state geometry: at K=12 it used only
about 1.82 entropy-effective states and generated approximately 23 bouts/min.
PCA16 produced lower seed variability and less pathological segmentation.

Raw likelihoods across 64D and PCA16 are not directly comparable because they
are densities in different transformed spaces. PCA16 was selected from its
stability, occupancy, and segmentation behavior, not because its numerical
log-likelihood is less negative.

## K selection for the graph-level SSL encoder

### HMM, PCA16, five seeds

| K | Validation LL/step | ARI | Median bout (s) | Bouts/min | Effective states | States >=1% |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | -31.696 +/- 0.222 | 0.550 | 2.464 | 12.25 | 1.97 | 3.0 |
| 6 | -29.856 +/- 0.072 | 0.930 | 1.808 | 10.44 | 1.80 | 4.2 |
| 8 | -29.370 +/- 0.326 | 0.825 | 1.264 | 14.05 | 1.97 | 5.2 |
| 10 | -29.472 +/- 0.135 | 0.844 | 1.280 | 15.80 | 2.06 | 5.0 |
| 12 | -29.288 +/- 0.055 | 0.939 | 1.216 | 16.75 | 2.16 | 5.0 |

AIC and BIC continued decreasing through K=12 because of the large number of
correlated patch observations. However, K=12 did not produce 12 substantively
occupied states; it increased switching while providing only about 2.16
entropy-effective states. K=6 was selected as the parsimonious preliminary
choice. K=12 remains a plausible likelihood-oriented alternative and should
be reconsidered when independent validation trials or behavioral labels exist.

### HSMM, PCA16, Dmax=50, five seeds

| K | Validation LL/step | ARI | Median bout (s) | Bouts/min | Effective states |
|---:|---:|---:|---:|---:|---:|
| 4 | -32.157 +/- 0.173 | 0.526 | 1.808 | 30.53 | 2.43 |
| 6 | -30.454 +/- 0.148 | 0.850 | 1.664 | 31.20 | 2.33 |
| 8 | -29.938 +/- 0.329 | 0.786 | 1.248 | 37.43 | 2.50 |
| 10 | -29.967 +/- 0.144 | 0.790 | 1.152 | 39.45 | 2.68 |
| 12 | -29.785 +/- 0.046 | 0.889 | 1.120 | 41.31 | 2.78 |

Increasing K again improved likelihood modestly while increasing
fragmentation. The same K=6 was retained for controlled HMM/HSMM comparison.

## HMM versus HSMM

The selected comparison uses graph-level SSL embeddings, PCA16, and K=6.
HSMM uses the less restrictive Dmax=100, corresponding nominally to 16 s at
the observed stride.

| Model | K | Validation LL/timestep | State stability | Median bout duration | Effective states |
|---|---:|---:|---:|---:|---:|
| HMM | 6 | -29.856 +/- 0.072 | ARI 0.930 | 1.808 s | 1.80 |
| HSMM | 6 | -30.521 +/- 0.012 | ARI 0.991 | 1.760 s | 2.35 |

The best HMM seed has 397 validation bouts, 12.91 bouts/min, and LL/step
`-29.7148`. The selected HSMM seed has 906 bouts, 29.46 bouts/min, and LL/step
`-30.5037`. HSMM is more seed-stable and uses somewhat more states, but has
worse held-out likelihood and substantially more switching. Explicit duration
therefore did not improve the real-data segmentation in this implementation.

Duration sensitivity at K=6 confirms that this is not only a restrictive cap:

| Dmax | Nominal maximum | Validation LL/step | ARI | Bouts/min |
|---:|---:|---:|---:|---:|
| 25 | 4 s | -30.563 | 0.985 | 34.49 |
| 50 | 8 s | -30.454 | 0.850 | 31.20 |
| 100 | 16 s | -30.521 | 0.991 | 29.50 |

The wider cap reduces fragmentation but does not approach the HMM bout rate.
The HSMM is preserved as a valid negative result and future alternative, not
selected as the primary state sequence.

## Upstream encoder comparison

Raw Gaussian likelihood values are not used to rank encoders, even though all
comparisons use PCA16, because each encoder defines a different representation
space.

| Frozen encoder, HMM K=6 | ARI | Median bout (s) | Bouts/min | Effective states | States >=1% |
|---|---:|---:|---:|---:|---:|
| Graph-level SSL | 0.930 | 1.808 | 10.44 | 1.80 | 4.2 |
| Original strict SSL | 0.572 | 0.800 | 51.57 | 5.46 | 6.0 |
| Random untrained GNN | 0.914 | 0.608 | 45.85 | 3.77 | 5.4 |

The strict encoder occupies more states but fragments into rapid, unstable
switching; at K>=8 its median bout is 0.32 s and it reaches 99-115 bouts/min.
An exploratory strict-encoder HSMM run was stopped after two seeds because it
still produced about 81 bouts/min while requiring 100-166 s per seed. The
graph-level SSL representation is markedly smoother and slightly more stable
than the untrained architecture, although its low effective-state count also
suggests partial representation collapse or a dominant background state.

This comparison supports graph-level SSL as the more defensible preliminary
downstream encoder, but it does not establish that its smoother segmentation
is biologically more correct.

## Provisional state signatures

Across all five exploratory exports, the selected HMM uses all six states. The
largest state occupies 36.8% of valid patches and the smallest 5.5%. Raw
feature summaries expose distinct measurement signatures:

- State 1 has comparatively short centroid distance (~489 px), strong mutual
  facing (~0.66/0.68), and positive pair closing speed (~51 px/s).
- State 5 is the most distant (~1116 px), with both animals generally facing
  away and weak/negative approach and closing measurements.
- State 3 is brief (median 0.32 s) and has extreme dynamic measurements,
  including strongly negative pair closing speed; it may represent a real
  transition regime, tracking noise, or both.
- States 0, 2, and 4 occupy intermediate distance/direction regimes.

These are descriptions of measured features, not behavior names. The exports
include the ten highest-confidence patch references per state for later video
review.

## Selected artifacts

Primary HMM:

- Encoder: graph-level SSL checkpoint SHA-256
  `E57A6DBD855FAE7792BE03ABFCAF554196DBF44BB0E5A5D05E8A68C6D4A1544A`.
- PCA dimension: 16.
- K: 6; temporal seed: 4.
- Model: `evaluation_runs/downstream_v1/selected_hmm/model.npz`.
- Model SHA-256:
  `BA013AD212BF1B3595514066FDA13B43D0604FC20CCAC5CEC3197C790D7D73D5`.
- Preprocessor: `evaluation_runs/downstream_v1/selected_hmm/preprocessing.npz`.

Preserved HSMM comparison:

- PCA dimension: 16.
- K: 6; temporal seed: 2; Dmax: 100 timesteps.
- Model: `evaluation_runs/downstream_v1/selected_hsmm/model.npz`.
- Model SHA-256:
  `636685F7B653471CBAA28CC75278CF0956F36635C79F580689DC605FDAC8612F`.

Both selected directories contain preprocessing statistics and complete state
exports. Each trial NPZ contains embeddings, posterior probabilities,
confidence, predicted state, bout IDs, timestamps, and duration fields. CSVs,
bout tables, transition matrices, state summaries, representative patches,
timelines, duration histograms, a transition heatmap, and a visualization PCA
are also present.

## Scientific limitations and next test

The present results establish a functioning and probabilistically verified
temporal pipeline, preliminary K uncertainty, and meaningful differences
between upstream representations. They do not establish biological state
validity. There is only one model-selection trial, A5_C5 is not independent,
overlapping/receptive-field-correlated observations weaken nominal AIC/BIC
sample-size assumptions, and no external labels were available.

The next decisive step is blinded video interpretation on representative
patches and complete bouts, followed by evaluation on newly collected,
untouched trials. State numbers must remain provisional until that work is
done.
