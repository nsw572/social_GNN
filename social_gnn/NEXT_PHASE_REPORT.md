# Social-GNN next-phase validation report

Date: 2026-10-14

## Scope and evaluation protocol

This phase tested whether SSL improves the final patch-level graph embedding,
removed the decoder shortcut around that embedding, and evaluated a
permutation-consistent role-aware readout. The temporal architecture remains:

```text
fixed social patch -> node/edge inputs -> patch-level encoder -> g_t -> SSL head
```

There is no TCN, RNN, or Transformer in this stage. The training trials were
`A1_C1`, `A3_C3`, and `A4_B4`; all architecture decisions below use only
`A4_C4` as validation. `A5_C5` had already been inspected before this phase,
so any files generated for it by the existing experiment runner are
exploratory and were not used for the selection below. One validation trial is
not enough to establish cross-trial generalization; a new independent trial is
still required.

All reported probes freeze the encoder. Linear-probe hyperparameters are
selected on whole validation-trial data without crossing trial boundaries.
Every checkpoint is evaluated with the same probe sampling seed `20261013`,
so encoder-seed variation is not confounded with probe-sampling variation.
The primary edge metric uses permutation-invariant targets: five symmetric
traits once, and the direction mean plus absolute directional asymmetry for
each of the three directed traits. The ordered edge score is retained as a
secondary diagnostic. "Directed R2" is the mean of the role-aligned summaries
for `source_faces_target` and `source_radial_approach_speed`.

## Main comparison

Values are validation R2 on `A4_C4`; `+/-` is population standard deviation
over seeds. Original SSL, graph-level mean, and MLP use three training seeds.
The random GNN uses five independent initializations. Soft-role was stopped
after one seed because it was already inferior on the selection trial.

| Model | Edge R2 invariant (ordered) | Directed R2 | Invariant node R2 | SSL benefit | Permutation error |
|---|---:|---:|---:|---:|---:|
| Untrained mean-pool GNN | 0.146 +/- 0.064 (0.167 +/- 0.070) | 0.057 +/- 0.053 | 0.665 +/- 0.004 | N/A | 0.000 |
| Original SSL GNN | 0.798 +/- 0.023 (0.771 +/- 0.007) | 0.657 +/- 0.054 | 0.492 +/- 0.115 | +0.653 edge R2 | 0.000 |
| Raw ordered node-edge fusion | 0.788 (1.000) | 0.610 | 0.502 | N/A | N/A: ordered reference |
| Graph-bottleneck MLP fusion | 0.882 +/- 0.018 (0.926 +/- 0.014) | 0.851 +/- 0.029 | 0.307 +/- 0.165 | random MLP not measured | 0.211 mean, 0.959 max |
| Graph-level SSL + mean pooling | 0.754 +/- 0.032 (0.688 +/- 0.011) | 0.734 +/- 0.029 | 0.553 +/- 0.086 | +0.608 edge R2 | 0.000 |
| Graph-level SSL + soft-role readout | 0.732 (0.634) | 0.718 | 0.566 | matching random control not measured | 0.000 |

The fair trained-versus-untrained comparison therefore supports a large SSL
gain for edge information, but not a gain for invariant node information in
the original objective: original SSL is 0.172 below the random-GNN mean on the
node metric. The graph-level mean objective raises invariant-node retention
relative to original SSL, but its three-seed mean remains 0.112 below the
random mean. This criterion is not yet satisfied consistently.

Raw fusion is an uncompressed, target-containing diagnostic rather than a
learned representation. Its ordered edge score is essentially perfect because
the target is present in the input. The lower invariant score reflects the
nonlinear absolute-asymmetry target under a linear probe, not missing raw data.

## Per-trait invariant linear retention

Each directed trait is averaged over its direction-mean and absolute-
asymmetry targets. Values for multi-seed models are mean +/- population SD.

| Trait | Random GNN | Original SSL | Raw fusion | MLP | Graph mean | Soft role |
|---|---:|---:|---:|---:|---:|---:|
| centroid distance | 0.209 +/- 0.095 | 0.982 +/- 0.006 | 1.000 | 0.980 +/- 0.002 | 0.918 +/- 0.035 | 0.853 |
| nose-to-nose distance | 0.361 +/- 0.164 | 0.979 +/- 0.010 | 1.000 | 0.976 +/- 0.011 | 0.961 +/- 0.015 | 0.970 |
| source nose to target centroid | 0.125 +/- 0.068 | 0.624 +/- 0.044 | 0.614 | 0.773 +/- 0.039 | 0.614 +/- 0.027 | 0.642 |
| source faces target | 0.099 +/- 0.068 | 0.656 +/- 0.033 | 0.675 | 0.838 +/- 0.042 | 0.615 +/- 0.039 | 0.515 |
| head-direction alignment | 0.138 +/- 0.087 | 0.983 +/- 0.005 | 1.000 | 0.915 +/- 0.007 | 0.883 +/- 0.065 | 0.771 |
| source radial approach speed | 0.015 +/- 0.040 | 0.658 +/- 0.111 | 0.545 | 0.865 +/- 0.040 | 0.852 +/- 0.022 | 0.922 |
| pair closing speed | 0.190 +/- 0.073 | 0.975 +/- 0.006 | 1.000 | 0.977 +/- 0.002 | 0.904 +/- 0.016 | 0.926 |
| movement-direction alignment | 0.226 +/- 0.079 | 0.987 +/- 0.004 | 1.000 | 0.904 +/- 0.039 | 0.462 +/- 0.069 | 0.376 |

The graph bottleneck materially improves radial-approach retention over the
original objective (+0.194), but reduces facing retention (-0.041)
and loses substantial movement-alignment information. The soft-role readout
pushes radial approach further to 0.922, but harms facing, movement alignment,
aggregate edge retention, and invariant-node retention. This is not a net
improvement, so additional seeds were not justified.

## Nonlinear probes

A constrained 10-epoch MLP probe was also run for one seed. These results are
less stable than the linear probes, but do not reverse the selection:

| Model | Invariant edge R2 | Invariant node R2 |
|---|---:|---:|
| Original SSL | 0.812 | 0.554 |
| Graph-level mean | 0.757 | 0.635 |
| MLP fusion | 0.818 | 0.539 |
| Soft role | 0.687 | 0.468 |

## Decoder bypass and graph-level target design

The old bypass was confirmed. The original node decoder consumes
`[node_embedding, g_t]`, while its edge decoder consumes
`[source_embedding, target_embedding, g_t]`. Reconstruction can therefore
optimize pre-pooling node states without forcing all required information
through the exported `g_t`.

`GraphLevelSocialSSL` removes that path: both decoders receive only `g_t`.
Its invariant targets are explicit:

- Node target: six feature-wise animal means plus six absolute differences.
- Symmetric edge target: traits 0, 1, 4, 6, and 7 once per patch.
- Directed edge target: direction mean and absolute directional asymmetry for
  traits 2, 3, and 5.
- Confidence and missingness remain separate target weights.
- Symmetric edge masks are coupled across both directions.

The decoder input dimension is tested to equal `graph_dim`, and a unit test
checks that decoder loss gives a nonzero gradient through `g_t`. No decoder in
this path receives node embeddings.

## Role-aware readout and permutation consistency

The experimental `soft_role` readout uses one shared scorer per animal over
its node embedding plus incoming and outgoing edge summaries. Softmax scores
form high-role and low-role weighted summaries; the graph representation uses
their global mean, high-role summary, low-role summary, difference, and role
entropy. Swapping both animal axes only swaps the equivariant intermediate
scores, so the final graph embedding is invariant.

Formal swap tests permute node features, node masks, and both axes of edge
values/confidence. Mean and soft-role GNN readouts have zero measured error at
the recorded precision. The ordered MLP has mean error 0.211 and maximum error
0.959, demonstrating that its stronger retention is partly purchased by
depending on arbitrary animal indices.

An invariant `g_t` cannot recover the labels "animal A" and "animal B" after
those labels are exchanged. It can preserve role-aligned statements such as
unilateral versus mutual approach and the magnitude of directional
asymmetry; identity-ordered outputs require a separate equivariant node-level
representation and must not be used as a shortcut around the graph bottleneck.

## Sensitivity and embedding geometry

One-seed sensitivity on `A4_C4` is normalized by each representation's real
latent RMS radius. The ratio column compares shuffled-edge displacement with
node-shuffle displacement.

| Model | Edge shuffle | Matched edge change | Confidence only | Edge/node ratio |
|---|---:|---:|---:|---:|
| Random GNN | 0.053 | 0.084 | 0.013 | 0.045 |
| Original SSL | 1.150 | 1.657 | 0.099 | 2.320 |
| Graph-level mean | 0.667 | 1.167 | 0.077 | 0.831 |
| MLP fusion | 0.771 | 1.590 | 0.085 | 0.944 |
| Soft role | 1.034 | 1.867 | 0.110 | 1.317 |

The graph-level mean candidate remains edge-sensitive, but is less dominated
by edges than the original SSL representation. Confidence-only changes are
small for every trained model, consistent with confidence acting mainly as a
quality channel rather than replacing physical values.

| Model | Parameters | g dimension | Participation ratio | Entropy rank | Total variance |
|---|---:|---:|---:|---:|---:|
| Original SSL | 65,876 | 64 | 4.863 | 8.314 | 18.97 |
| Random GNN | 65,876 | 64 | 6.25-7.57 | 9.92-11.37 | 23.17-24.75 |
| Graph-level mean | 54,173 | 64 | 6.322 | 10.403 | 27.57 |
| MLP fusion | 67,869 | 64 | 5.106 | 7.624 | 66.19 |
| Soft role | 81,118 | 64 | 5.969 | 8.335 | 35.62 |

No trained representation is numerically collapsed, although all use far
fewer than 64 effective dimensions. A previous controlled 32D projection
reduced retention and is documented separately in `EDGE_EVALUATION_REPORT.md`;
dimension reduction is not selected solely for compression.

## Exact shuffled-edge result from the previous evaluation

The previous shuffled-edge-trained encoder had native shuffled-input linear
edge R2 `-0.0963418339`. When that same trained encoder was instead fed real
edges, it reached `0.7698638674`, while the full-edge-trained baseline reached
`0.7651745405`. The exact real-input difference was `+0.0046893269` in favor of
the shuffled-trained checkpoint. This is evidence that the architecture can
pass through real edges even when its training condition did not teach their
correct temporal pairing; it is not evidence that shuffled edges learned the
same relational objective.

## Selected downstream candidate

The preliminary HMM/HSMM candidate is **graph-level SSL with mean pooling**,
seed `20261013`. It was selected by frozen validation representation quality,
not by reconstruction loss: among the three graph-mean seeds it had the best
invariant edge R2 (0.797), directed R2 (0.774), and invariant-node R2 (0.674).
The seed with the lowest reconstruction loss had materially worse retention.

- Best epoch: 90/100.
- Best validation total loss: 1.6686606407.
- Validation node loss: 1.2991596460.
- Validation edge loss: 0.3695010543.
- Checkpoint:
  `evaluation_runs/next_phase/selected_graph_mean/best.pt`.
- SHA-256:
  `E57A6DBD855FAE7792BE03ABFCAF554196DBF44BB0E5A5D05E8A68C6D4A1544A`.
- Reproducible config:
  `configs/social_gnn_selected_graph_mean_6dim_v1.yaml`.

This choice is not the maximum-retention model. It is selected because it
directly optimizes the exported `g_t`, is exactly permutation invariant,
improves the weak radial directional trait, preserves invariant node
information in the selected seed, and does not rely on ordered-input
shortcuts. Its substantial seed variance is a remaining weakness, not a
resolved result. The original strict checkpoint is
preserved unchanged and remains the stronger pure edge-retention baseline.

## What is and is not established

Supported by current validation evidence:

1. SSL strongly improves edge retention over the untrained GNN.
2. The old decoder bypass exists; the new graph-level objective removes it.
3. Useful directional asymmetry survives an invariant graph readout.
4. The selected model is independent of arbitrary mouse indexing.
5. The selected seed balances node and edge sensitivity better than the old
   edge-dominated representation.

Not yet established:

1. Generalization to a genuinely unexamined trial.
2. Behavioral-state usefulness for clustering or HMM/HSMM segmentation.
3. Superiority of GNN over all simpler models: the ordered MLP retains more
   measured information, although it fails the required invariance.
4. A net benefit from the current soft-role mechanism.
5. Whether lost movement-alignment information matters downstream.
6. Consistent preservation of node information across training seeds.

The next scientific test should freeze this selected encoder, export `g_t` on
new independent trials, and compare downstream HMM/HSMM stability and
behavioral coherence against the preserved original SSL, untrained GNN, and
MLP baselines. No biological-usefulness claim should be made before that test.
