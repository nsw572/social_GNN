# Social-GNN node preservation and role-binding report

## Decision

The most defensible model from this phase is the **unchanged Social GNN encoder
trained with 2:1 full-input preservation plus masked reconstruction for 100
epochs** (`Hybrid_SSL` below). It improves node retention substantially while
also improving edge retention over the prior selected checkpoint. The explicit
dual branch is useful as a mechanistic ablation, but it is not selected because
its small node gain is accompanied by worse edge, directed-edge, role-binding,
and effective-rank results.

This is validation evidence, not a final generalization estimate. A4_C4 is the
only model-selection trial. A5_C5 was previously inspected and was excluded
from all fitting and selection. No new independent trial was available.

## Data and protocol

- Encoder train trials: A1_C1, A3_C3, A4_B4.
- Encoder validation: A4_C4.
- A5_C5: excluded.
- Probe fit: A1_C1 and A3_C3; probe hyperparameters: A4_B4; final comparison:
  A4_C4.
- Upstream node input remains 6D and unchanged.
- Every encoder is patch-local; no neural temporal encoder was introduced.
- All decoders see only final `g_t`.
- Primary common metrics use frozen Ridge probes. Role binding uses the frozen
  small MLP because its target is a product of directional differences.

## Unified comparison

Values are on A4_C4. Mean +/- sample SD is across three encoder seeds where
shown. Role binding is the primary
`(facing_AB-facing_BA)*(approach_AB-approach_BA)` target.

| Experiment | Node mean R2 | Node difference R2 | Edge R2 | Directed R2 | Role-binding R2 |
|---|---:|---:|---:|---:|---:|
| Original graph SSL | 0.861 | 0.072 | 0.684 | 0.557 | 0.355 |
| Hybrid SSL | **0.925 +/- 0.025** | **0.814 +/- 0.053** | **0.753 +/- 0.005** | **0.585 +/- 0.004** | **0.410 +/- 0.064** |
| Node branch only | 0.945 +/- 0.020 | 0.851 +/- 0.040 | -0.046 +/- 0.005 | -0.041 +/- 0.008 | -0.020 |
| Dual branch | 0.924 +/- 0.030 | 0.842 +/- 0.047 | 0.700 +/- 0.007 | 0.510 +/- 0.011 | 0.293 |
| Full-input unordered node set | 0.962 | 0.280 | 0.753 | 0.578 | 0.354 |
| Hybrid unordered node set | 0.950 | 0.267 | 0.736 | 0.559 | 0.362 |
| Joint node-edge set | 0.931 | 0.158 | 0.691 | 0.493 | 0.146 |

The machine-readable source is
`evaluation_runs/node_preservation_v1/unified_comparison/unified_comparison.csv`.
Set-based candidates were pilot-tested with one seed because they failed the
common frozen-probe criterion; their apparent mean retention did not extend to
the much harder animal-difference component.

## Experiment A: probe audit

The earlier low node score was partly a target/probe artifact and partly real
encoder loss.

- Exact raw calculation and a linear probe on the already constructed exact
  invariant target both achieve R2 approximately 1.
- A linear probe from ordered raw nodes cannot express absolute differences;
  this explains why that diagnostic can look poor even though the information
  is present.
- A nonlinear raw-input MLP recovers much more of the difference target.

The trained baseline's frozen linear stage audit was:

| Stage | Node mean R2 | Absolute-difference R2 |
|---|---:|---:|
| Input projection | 0.986 | 0.476 |
| GINE layer 1 | 0.914 | 0.388 |
| GINE layer 2 | 0.870 | 0.265 |
| Final mean pool | 0.856 | 0.064 |

Thus information erodes throughout message passing, but the largest final loss
of individual configuration occurs at graph pooling. The trained baseline also
retained less node difference than a deterministic untrained encoder, showing
that the old SSL objective encouraged the wrong bottleneck, not merely that the
probe lacked capacity.

## Experiment B: hybrid SSL

The implemented objective is

`L = preserve_weight * (node_full + edge_full) + masked_weight * (node_masked + edge_masked)`.

Each component is normalized by its valid/selected weighted target count, so
edge dimensionality cannot dominate merely by supplying more scalars. The
pilot searched preservation weights 0.25, 0.5, 1, and 2 with masked weight 1.
Among hybrids, increasing preservation weight consistently improved invariant
node retention without materially damaging edge retention. Weight 2 was used
for the serious comparison. Full-only reconstruction had better node retention
than 60-epoch hybrids, but discards the desired masked inference task. Extending
the selected hybrid to 100 epochs produced the strong three-seed result above.

## Experiment C: node-preservation branch

The node branch uses a shared nonlinear `phi` per animal followed by encoded
mean and encoded absolute pair difference, then `rho`. This preserves
within-animal feature interactions better than a feature-wise raw mean alone.
The dual model fuses its output with the ordinary GINE graph summary and keeps a
64D final bottleneck.

The ablation behaved as expected:

- Node-only strongly preserves nodes and is insensitive to every edge
  perturbation; its edge and role probes are approximately zero.
- Dual branch preserves nodes, remains edge-sensitive, and is exactly invariant.
- Social-only hybrid is the best trade-off. The dual branch has 76,189
  parameters versus 54,173 and lowers edge R2 from 0.753 to 0.700.

One three-model, 100-epoch C sweep took about 85 seconds on one RTX 4090
(roughly 28 seconds/model under this small five-trial workload). Parameter
count, rather than wall time from a single machine, is the more portable cost
comparison.

On A4_C4, matched edge replacement changes normalized embeddings by 0.954 for
social-only and 1.057 for dual; confidence-only replacement changes them by
only 0.062 and 0.054. Node shuffling changes them by 0.880 and 0.950. These
checks show genuine sensitivity to both modalities without confidence channels
dominating.

## Experiment D: set and joint reconstruction

Exact two-permutation matching is numerically stable, differentiable, and uses
one common node permutation for both axes of the dense directed edge matrix in
the joint case. Unit tests verify the matching rule and nonzero gradients.

Held-out full-input unordered node-set reconstruction RMSE in original 6D units
was 0.268 for full-only set reconstruction, 0.392 for hybrid set
reconstruction, 0.530 for dual set reconstruction, and 0.546 for joint dual
reconstruction. Despite these finite reconstruction results, set objectives
retained node absolute differences poorly in frozen `g_t`. The joint objective
also reduced role binding and effective rank (6.02). This is a concrete example
of lower decoder loss not implying a better representation.

## Experiment E: cross-trait role binding

Three deliberately limited invariant targets were evaluated:

1. facing asymmetry x radial-approach asymmetry;
2. facing asymmetry x nose-to-partner-centroid-distance asymmetry;
3. radial-approach asymmetry x nose-to-partner-centroid-distance asymmetry.

The deterministic raw ceiling is exactly 1.0. An ordered raw-input MLP reaches
0.920 on the primary target, demonstrating that the measurements contain the
signal. The original model reaches 0.355 and selected hybrid reaches
0.410 +/- 0.064 across three encoder seeds;
therefore role correspondence survives pooling only partially. The explicit
dual branch does not improve it. Matched-counterfactual pairs use real observed
patches with close node/marginal-edge summaries and different role targets;
they are diagnostics, not synthetic biological examples.

No node-latent role target was invented because the six upstream latent axes
do not have stable biological names. A successful probe indicates recoverable
measurement correspondence, not recognized social behavior.

## Representation geometry and downstream sanity check

The selected hybrid has zero measured permutation error, total embedding
variance 18.33, and entropy effective rank 16.73, compared with 16.65 and 12.31
for the original model. The dual branch has much larger variance but lower
effective rank (10.60), so scale inflation is not treated as improvement.
For the selected-seed run, final train/validation total reconstruction losses
were 1.075/2.002, so a substantial cross-trial generalization gap remains even
though frozen validation probes improved.

A limited frozen HMM check used PCA-16, K in {4,6,8}, and three seeds. All runs
converged. K=6 had pairwise ARI 0.773, median bout duration 1.65 s, and 9.40
bouts/min; K=8 improved held-out likelihood but had lower ARI 0.544. Effective
occupied states remained low (about 1.55-1.69), so state collapse is still a
downstream concern. This check was not used to tune the upstream SSL model.

## Selected artifacts

- Selected architecture/seed candidate:
  `evaluation_runs/node_preservation_v1/experiment_c/experiment_20261015_210713/social_gnn_only/best.pt`
  (SHA256 `7701B3B382D76502EF8F26CDC5F68531E7AA194557482B6DBF3227AE2223F52E`).
- Three-seed C runs:
  `evaluation_runs/node_preservation_v1/experiment_c/`
- Unified table:
  `evaluation_runs/node_preservation_v1/unified_comparison/`
- Role probes:
  `evaluation_runs/node_preservation_v1/experiment_e_role_binding/`
- Set reconstruction:
  `evaluation_runs/node_preservation_v1/experiment_d_set_reconstruction/`
- HMM sanity check:
  `evaluation_runs/node_preservation_v1/hmm_sanity_hybrid/`

The selected checkpoint is not declared a final production model until at
least one new independent trial is available. The original selected checkpoint
was preserved and never overwritten.
