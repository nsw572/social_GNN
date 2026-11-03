# Social-GNN 0.04 s pointwise migration

Date: 2026-10-22

## Canonical architecture

The active Social-GNN clock is now a non-overlapping 0.04 s grid:

```text
one aligned upstream latent per animal
            +
video edge traits on the same [start, end) interval
            -> SocialTrialPackage
            -> patch-local GNN
            -> g_t every 0.04 s
            -> sticky Gaussian HMM / comparison HSMM
```

Node and edge tensors retain their existing shapes. Only `T` becomes roughly
four times longer than in the retired 0.16 s pipeline. The model still has no
neural temporal encoder.

## Pointwise reconstruction contract

- Node conversion defaults to `duration=stride=0.04 s`.
- At the upstream 25 Hz rate, one timestep normally contains one latent sample.
- Edge state traits use video frames in the identical half-open interval.
- Edge dynamic traits use adjacent point summaries and timestamp-derived `dt`.
- Values, per-trait confidence, and coverage remain separate.
- Old 0.16 s authoritative clocks are rejected by the default node and edge
  conversion paths.
- An experiment config may declare its expected duration and stride; loading a
  mismatched `SocialTrialPackage` then fails before training.

## Sticky HMM

The HMM uses a physical-time-derived self-transition target. For timestep
`dt` and expected geometric dwell time `tau`:

```text
p_self = 1 - dt / tau
```

The canonical defaults are `dt=0.04 s`, `tau=2.0 s`, hence `p_self=0.98`.
This target initializes the transition matrix and contributes a configurable
Dirichlet prior during every Baum-Welch M-step. It does not hard-code decoded
bout lengths and does not delete short bouts after decoding. Checkpoints store
the timestep, expected duration, and prior strength; run tables report the
learned mean diagonal transition probability.

HSMM remains a separate explicit-duration comparison. To preserve the old
physical search range of 4/8/16 seconds, its 0.04 s duration caps are
100/200/400 timesteps.

## Assessment of the three supplied reference scripts

- `hsmm_segment.py`: useful for its physical-duration viewpoint, independent
  segment fitting, and explicit warning that state persistence belongs in the
  temporal model. Its HSMM forces diagonal transitions to zero, so it is not a
  sticky HMM and was not copied into the canonical HMM implementation.
- `hsmm_scan.py`: its segmented K scan and no-cross-recording principle agree
  with the existing downstream benchmark. The canonical code already keeps
  trials and invalid-gap runs independent and adds held-out metrics and
  checkpoint traceability.
- `create_video_pairs.py`: useful later for blinded same-state/different-state
  human review. It is an interpretation/QC helper, not an HMM training input,
  and therefore is not coupled to the encoder or transition estimation.

## Artifact compatibility

All existing 0.16 s node NPZs, edge NPZs, GNN checkpoints, embeddings, HMM and
HSMM checkpoints, ARI tables, and video montages remain valid records of the
old experiments, but they are not compatible with the new 0.04 s model. Do not
upsample or relabel them.

The required rerun order is:

1. regenerate 0.04 s node NPZs from upstream aligned latents;
2. extract edge traits against those exact node clocks;
3. validate new `SocialTrialPackage` files;
4. retrain the selected GNN objectives and export new `g_t` sequences;
5. rerun downstream selection with the sticky HMM and duration-rescaled HSMM;
6. repeat stability and video-based biological review.

Code and unit-test migration is complete. Data regeneration and model reruns
are intentionally still pending.
