# 9D Hybrid SSL and HMM/HSMM rerun (2026-10-16)

## Scope

This rerun replaces the historical 6D nodes with the upstream batch-3 9D
latents while preserving the existing video-derived edge tensors and each
trial's authoritative 0.16 s social clock. The model remains patch-local:

`9D nodes + 8 edge values + 8 edge confidences -> two-layer GINE -> 64D g_t`

No TCN or other neural temporal encoder is used. HMM/HSMM fitting happens only
after the Hybrid SSL encoder is frozen.

## Clock and identity alignment

The converter now accepts `--authoritative-clock-npz`. It checks the trial ID
and source-animal order, copies `patch_start_s`, `patch_end_s`, `social_step`,
and output identity from the canonical node package, and aggregates four 25 Hz
latent samples per 0.16 s patch. The upstream delivery is cropped relative to
the full trial. Missing boundary patches are retained but marked invalid rather
than shortening the clock or supplying trainable zeros.

| Trial | Full patches | Patches valid for both animals | Coverage |
|---|---:|---:|---:|
| A1_C1 | 12,698 | 12,189 | 96.0% |
| A3_C3 | 11,753 | 10,878 | 92.6% |
| A4_B4 | 13,071 | 12,458 | 95.3% |
| A4_C4 | 11,532 | 10,700 | 92.8% |
| A5_C5 | 10,951 | 10,597 | 96.8% |

All five node packages pass strict `SocialTrialPackage` clock, identity, shape,
finite-value, and edge-alignment validation.

## Hybrid SSL training

Training trials were A1_C1, A3_C3, and A4_B4. A4_C4 selected the checkpoint;
A5_C5 remained excluded from training and selection. Each run used 100 epochs,
the full edge input, and the same Hybrid objective and architecture.

| Encoder seed | Best epoch | A4_C4 total loss | Preserve loss | Masked loss |
|---:|---:|---:|---:|---:|
| 20261013 | 100 | **1.1457** | 0.1561 | 0.8335 |
| 20261107 | 100 | 1.1660 | 0.1569 | 0.8522 |
| 20261029 | 100 | 1.1873 | 0.1603 | 0.8667 |

The validation-selected encoder is seed 20261013. Its checkpoint SHA-256 is
`71308AFECE26700B6AD8E78876F198BA10170FC4FA2A8ACCD8F5C7DD9D0A00A4`.

Frozen Ridge probes on A4_C4, summarized across all three encoder seeds:

| Probe target | Mean R2 | Seed SD |
|---|---:|---:|
| Node mean | 0.984 | 0.001 |
| Node absolute difference | 0.807 | 0.016 |
| All edge targets | 0.707 | 0.002 |
| Directed edge targets | 0.554 | 0.002 |

The representation preserves both average animal state and between-animal
difference. Edge retention remains substantial but is weaker than node-mean
retention. A4_C4 was also used for encoder selection, so these values are
validation diagnostics, not an untouched generalization estimate.

## HMM/HSMM comparison

The selected encoder was frozen. PCA16 was fitted on the three training trials
only. The search covered HMM and shifted-Poisson explicit-duration HSMM,
K = 4, 6, 8, and seeds 0, 1, 2. A4_C4 supplied held-out likelihood and
cross-seed state agreement. Invalid patches split sequences; no transition
crosses an invalid gap or trial boundary.

| Model | K | Validation LL / step | SD | Pairwise ARI | Median bout (s) | Bouts/min |
|---|---:|---:|---:|---:|---:|---:|
| HMM | 4 | -29.856 | 0.020 | 0.399 | 0.907 | 45.0 |
| HMM | 6 | **-29.500** | 0.073 | 0.455 | 0.747 | 63.6 |
| HMM | 8 | -29.253 | 0.122 | 0.417 | 0.587 | 72.7 |
| HSMM | 4 | -29.949 | 0.029 | **0.485** | 0.960 | 52.1 |
| HSMM | 6 | -29.583 | 0.059 | 0.463 | 0.800 | 67.1 |
| HSMM | 8 | -29.309 | 0.123 | 0.413 | 0.640 | 75.9 |

Likelihood improves with more states, but seed agreement is only moderate and
bout fragmentation rises sharply. HSMM does not improve held-out likelihood or
reduce fragmentation at matched K. The provisional operating choice is
therefore **HMM K=6, seed=1** as a compromise, not as a final behavioral-state
claim. Its exported full-dataset decoding has six occupied states, median bout
0.80 s, mean posterior confidence 0.929, and 56.7 bouts/min.

The principal limitation is now state stability rather than missing code. The
next defensible step is video-based interpretation and external validation of
the exported states; state numbers must not yet be assigned behavioral names.

## Artifacts

- Canonical nodes: `evaluation_runs/hybrid_9dim_v1/nodes/`
- Selected encoder: `evaluation_runs/hybrid_9dim_v1/selected_encoder/best.pt`
- Frozen probes: `evaluation_runs/hybrid_9dim_v1/probes/summary/`
- HMM/HSMM grid: `evaluation_runs/hybrid_9dim_v1/downstream/seed_20261013/search/`
- Selected temporal model: `evaluation_runs/hybrid_9dim_v1/selected_temporal/hmm_k6_seed1/`
- HMM and HSMM state exports: `evaluation_runs/hybrid_9dim_v1/downstream/seed_20261013/exports/`

## Verification

Forty-three focused unit tests passed, covering authoritative-clock conversion,
strict trial validation, dynamic 9D model dimensions, Hybrid masking/losses,
experiment output, HMM/HSMM numerics, checkpoint round trips, and prevention of
cross-trial or cross-gap transitions.
