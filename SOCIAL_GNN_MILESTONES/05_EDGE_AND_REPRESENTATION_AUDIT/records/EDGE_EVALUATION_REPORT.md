# Social-GNN relational representation evaluation (2026-10-13)

## Outcome

The trained GNN clearly responds to and retains video-derived edge values, but
the current evidence does **not** establish that it learned a useful
cross-modal node-edge relationship. None of the controlled SSL candidates
consistently beat the strict original baseline, so no improvement is claimed.

The controlled split was three full trials for training (`A1_C1`, `A3_C3`,
`A4_B4`), one for validation/selection (`A4_C4`), and one final held-out trial
(`A5_C5`). No patches crossed trial boundaries. The old 4/1 experiment used
`A5_C5` for encoder validation and is therefore diagnostic only, not a clean
final test.

## Audit findings

- The model is patch-local: 6D nodes, concatenated 16D value/confidence edges,
  two GINE-style layers, mean node pooling, and 64D `g_t`. There is no TCN.
- Traits 0, 1, 4, 6, and 7 are exact reverse-edge duplicates in all five
  trials. Independent masking can expose a masked answer unchanged on the
  reverse edge.
- Mean pooling is permutation invariant and cannot reliably retain the
  identity-ordered 6D vectors of both mice. Probes therefore report both an
  ordered node target and a graph-level invariant target (node mean plus
  absolute two-mouse difference).

## Added evaluation protocols

`evaluate_edge_sensitivity` tests real, zero, same-trial shuffled,
node-matched/edge-different, value-only, confidence-only, node-shuffled, and
coupled per-trait perturbations. Values and confidence move together except in
explicit stress tests. Displacement is normalized by the real latent RMS
radius, so latent scaling cannot improve the metric.

`evaluate_information_retention` freezes the encoder and fits trial-split
linear Ridge and constrained 32-unit MLP probes. Symmetric traits are counted
once and directed traits in both directions (11 outputs). Metrics include
confidence-weighted R2/MAE/RMSE and high/low-confidence subsets.

## Validation-only candidate selection

| Candidate | Edge R2 | Invariant-node R2 | Ordered-node R2 | Outcome |
|---|---:|---:|---:|---|
| strict original mask + mean readout | 0.768 | 0.507 | -1.877 | selected baseline |
| coupled symmetric mask + mean readout | 0.704 | 0.451 | -1.945 | safer masking, worse retention |
| coupled mask + role-aware readout | 0.599 | -0.402 | -0.358 | rejected |

The role-aware readout did not force `g_t` to preserve roles because the
existing decoders can reconstruct from node embeddings while bypassing the
pooled graph vector. A same-trial real-versus-shuffled matching objective also
failed: train/validation BCE stayed at 0.692/0.693, essentially random-guess
`log(2)`. Increasing its weight would amplify an unsupported objective.

## Final held-out `A5_C5`

### Sensitivity

| Edge intervention | normalized displacement | / node-shuffle displacement |
|---|---:|---:|
| zero/missing | 0.948 | 1.794 |
| same-trial shuffle | 1.186 | 2.244 |
| node-matched, edge-different | 1.709 | 3.234 |
| values shuffled only | 1.173 | 2.220 |
| confidence shuffled only | 0.071 | 0.135 |

Node-matched edge-different real patch pairs were 2.563 times farther apart in
latent space than edge-similar controls selected from the same node-nearest
candidate sets. The network is primarily sensitive to measurements, not just
confidence/missingness.

### Retention

| Frozen representation | probe | edge R2 | invariant-node R2 | ordered-node R2 |
|---|---|---:|---:|---:|
| selected GNN | linear | 0.770 | 0.512 | -0.166 |
| selected GNN | MLP | 0.767 | 0.510 | -0.258 |
| raw node | linear | -0.043 | 0.343 | 1.000 |
| direct node+edge diagnostic ceiling | linear | 1.000 | 0.342 | 1.000 |

Linear per-trait edge R2 was 0.988 centroid distance, 0.980 nose distance,
0.886 source-nose/target-centroid distance, 0.485 source-faces-target, 0.987
head alignment, 0.412 radial approach, 0.965 closing speed, and 0.990 movement
alignment. The two weakest traits are directional, pose-dependent, and not
reverse-edge duplicates.

The similar linear and MLP scores show that retained edge information is
easily accessible. However, the old shuffled-edge-trained encoder retained
real edges almost as well as the full-edge model when real edges were supplied
at inference. This is consistent with a strong edge pass-through pathway and
does not prove learned node-edge compatibility.

## Scientific conclusion

1. The original GNN responds strongly to social edges.
2. Its `g_t` retains recoverable edge information (final linear mean R2 0.770).
3. No tested improvement consistently outperformed the strict baseline.
4. Edge use and retention are demonstrated; genuine multimodal relational
   learning and behavioral usefulness are not yet demonstrated.
5. The next evidence should be cross-animal replication plus blinded behavior
   annotations and downstream HMM/HSMM stability. Five trials and one final
   test trial do not support claims of statistical significance.

The selected local artifact is under
`evaluation_runs/final_strict_baseline/selected_model/` with 65,876 parameters,
epoch 88, and checkpoint SHA-256
`82920AEE1C5FD3AAF29680ADA198FEC53E5FA4A47FB9A993752DE6D76FA13E89`.

## Follow-up: controlled 32D `g_t` bottleneck (2026-10-14)

To isolate graph-embedding dimensionality from message-passing capacity, the
GNN hidden width remained 64 and only the final graph readout was projected
from 64 to 32 dimensions. Decoder inputs were updated accordingly. The same
seed, masking, 100 epochs, and 3/1/1 trial split were retained.

Validation selection results deteriorated:

| `g_t` dimension | validation total loss | edge linear R2 | invariant-node linear R2 |
|---:|---:|---:|---:|
| 64 | 2.763 | 0.768 | 0.507 |
| 32 | 2.773 | 0.713 | 0.196 |

Because `A5_C5` had already been examined in the earlier experiment, the
following is a post-hoc repeated benchmark rather than a new untouched test:

| `g_t` dimension | edge linear R2 | edge MLP R2 | invariant-node linear R2 | ordered-node linear R2 |
|---:|---:|---:|---:|---:|
| 64 | 0.770 | 0.767 | 0.512 | -0.166 |
| 32 | 0.724 | 0.755 | 0.240 | -0.399 |

The 32D model remained highly edge-sensitive: same-trial edge shuffling caused
1.182 latent-radius units of displacement, essentially unchanged from 1.186
for 64D. Its edge/node sensitivity ratio increased from 2.24 to 2.60 mainly
because node-shuffle displacement fell from 0.528 to 0.454. This is precisely
why sensitivity alone is not a sufficient selection metric: the compressed
model appears more edge-dominant while retaining substantially less node and
linearly accessible edge information.

The bottleneck reduced parameters only from 65,876 to 63,860 (3.1%), because
the 64D message-passing stack was intentionally preserved. On current
evidence, 64D remains the preferred `g_t`; 32D is not selected. The 32D
checkpoint SHA-256 is
`8927FE22087F82F7A12652EBD044767A7999560B2D0E1CB945D62E07ECE1F31B`.
