# Social-GNN V0

Social-V0 operates on animal-level graphs, not body-part graphs. It accepts
continuous IMU latents from the upstream encoder and video-derived directed
relational features.

```python
import torch
from social_gnn import SocialV0, compose_edge_inputs

model = SocialV0(node_dim=128, edge_dim=16, hidden_dim=64)
h = torch.randn(8, 100, 2, 128)                 # [batch, patches, mice, latent]
value = torch.randn(8, 100, 2, 2, 8)            # eight physical traits
confidence = torch.rand(8, 100, 2, 2, 8)        # per-trait confidence
coverage = torch.rand(8, 100, 2, 2, 8)          # separate QC tensor, not model input
edges = compose_edge_inputs(value, confidence)  # 8 values + 8 confidences
g_t = model(h, edges)                          # [8, 100, 64]
```

The V0 implementation uses only PyTorch for message passing. This is
deliberate: PyTorch Geometric is not installed in the current interpreter.
Once the input contract is validated on real data, the layer can be swapped
for `GINEConv` without changing the dataset or downstream representation API.

## Patch-level representation

The trainable encoder deliberately stops at one independent graph embedding
per fixed physical patch:

```text
fixed social patch -> node/edge features -> GNN -> g_t -> SSL head -> loss
```

There is no TCN, RNN, Transformer, or other neural temporal encoder in this
stage. Changing another patch cannot change `g_t`. After SSL training, export
the valid `g_t` rows selected by `time_mask` for downstream clustering or a
separate Markov-style model such as an HMM/HSMM.

The current implemented SSL baseline is patch-local masked multimodal
reconstruction. `masked_social_ssl.py` clones a DataLoader batch, masks node
features or individual directed edge traits, and can drop a complete node or
edge modality for a patch. The original batch remains the target. The GNN
reconstructs hidden node latent values and eight hidden edge values; edge loss
uses the stored per-trait confidence as a weight without multiplying it into
the target value.

The parallel `vae.py` path is a patch-level VAE self-reconstruction baseline.
It reconstructs all node and edge values and adds a configurable KL term. It
also has no temporal encoder. The two objectives are separate experiments, not
two losses that must be enabled together.

Example:

```python
from social_gnn import MaskedSocialSSL, SocialMaskConfig
from social_gnn.masked_social_ssl import masked_social_reconstruction_loss

model = MaskedSocialSSL(
    node_dim=batch["node_features"].shape[-1],
    hidden_dim=64,
)
output = model(batch, mask_config=SocialMaskConfig(node_mask_probability=0.15))
loss = masked_social_reconstruction_loss(output, batch)["total"]
```

For an actual epoch, use `train_masked_social_ssl_epoch` or the independent
`train_social_patch_vae_epoch` helper with the existing full-trial DataLoader.
The DataLoader still shuffles only whole trials; no helper concatenates time
across trial boundaries.

## Reproducible experiment runner

`experiment.py` is the canonical training entry point. It loads every
node/edge pair through `SocialTrialPackage`, rejects clock or identity
mismatches and node files marked `training_eligible=false`, and fits all
normalization statistics on training trials only. Node channels use valid-node
means and variances. Edge trait statistics use confidence-weighted valid
directed edges; confidence itself stays in `[0,1]` and is never standardized.

The mature five-trial configuration is
`configs/social_gnn_experiment_6dim_v0.yaml`. Before all edge files exist, a
single complete trial can exercise every model and output path without
claiming a valid held-out result:

```powershell
python -m social_gnn.experiment `
  --config configs\social_gnn_experiment_6dim_v0.yaml `
  --smoke-test `
  --device cuda:1
```

Once every configured trial is complete, omit `--smoke-test` for the strict
formal run:

```powershell
python -m social_gnn.experiment `
  --config configs\social_gnn_experiment_6dim_v0.yaml
```

The default experiment compares masked SSL with full edges, node-only input,
and edges shuffled over time within each trial, plus an independent full-edge
patch VAE. The three masked runs start from identical weights and use identical
mask schedules, so their edge condition is the intended difference. Outputs
go to a new timestamped directory and include train-only normalization
statistics, per-epoch JSONL metrics, best/last checkpoints, and one
clock-preserving `g_t` NPZ per trial and split. No run overwrites source files
or shuffles data across trial boundaries.

## Full-trial data loading

Convert an upstream H ZIP (`latents.npz`, `patches.csv`, `meta.json`) into the
canonical node NPZ first. The converter preserves `t_start`/`t_end` exactly,
including overlapping patches, and maps `drop_flag` to `node_mask`. For the
current idtracker identity order, explicitly choose matching output labels:

```powershell
python social_gnn\h_zip_node_converter.py `
  --h-zip D:\input\trial.zip `
  --output-node-npz D:\social_trials\trial_001\node_features.npz `
  --output-identity 0 1
```

Exact duplicate latent streams are rejected as a likely upstream recombination
failure. `--allow-duplicate-nodes` exists only for interface tests; its output
is marked `training_eligible=false`.

For mature per-animal latent files (`downstream [S,D_node]`, `time [S]`, and
`interp_mask [S]`), use the separate fixed-clock converter. It assigns latent
timestamps to half-open 0.16 s patches and computes one validity-weighted mean
per animal and patch. At 25 Hz this normally pools four latent vectors into one
node vector. The approximately one-second receptive field used to produce each
upstream latent does not change the Social-GNN patch duration; it only means
adjacent node observations are not statistically independent.

```powershell
python -m social_gnn.latent_node_converter `
  --trial-id A4_B4 `
  --animal 'A4_1c=Y:\WSB\Computational_Ethology\processed_Data\latents\6dim\A4_B4_A4-1c_latent.npz' `
  --animal 'B4_ab=Y:\WSB\Computational_Ethology\processed_Data\latents\6dim\A4_B4_B4-ab_latent.npz' `
  --output-identity 1 2 `
  --authoritative-clock-npz C:\canonical_nodes\A4_B4_node_features.npz `
  --output-node-npz D:\social_trials\A4_B4\node_features.npz
```

By default non-interpolated samples have weight 1, interpolated samples have
weight 0, and a node remains valid when at least half of the expected sample
weight is present. `--interpolated-weight` and
`--minimum-weight-coverage` make both choices explicit. The node NPZ stores
`node_weight_coverage`, `node_interp_fraction`, and `node_sample_count` as QC
arrays; padding is still performed only later by the DataLoader. Optional
`--clock-start-s`/`--clock-end-s` bounds can restrict conversion to the strict
video/IMU overlap without changing the 0.16 s grid. For replacement latent
deliveries, prefer `--authoritative-clock-npz`: it preserves the existing full
trial clock, `social_step`, identity order, and source-animal order exactly.
Patches outside the delivered latent support remain present with zero values
and `node_mask=false`, so they cannot silently enter training or bridge a
downstream temporal sequence.

The converter permits only sub-microsecond-scale floating timestamp jitter and
extra samples confined to stream boundaries. It trims those boundary extras,
records the dropped counts in the summary, and uses the mean timestamp as the
shared clock. Internal gaps or timestamp offsets above the derived tolerance
raise an error instead of being silently aligned.

`SocialTrialPackage` represents one complete trial. It validates all arrays
before exposing them to PyTorch and raises `SocialTrialValidationError` with
the trial ID when node/edge clocks, identity order, shapes, finite values, or
confidence ranges disagree. Stored NumPy arrays are copied and made read-only.

For large datasets, use `SocialTrialSource` so each node/edge NPZ pair is loaded
lazily by the Dataset worker instead of loading every trial into RAM:

```python
import torch
from social_gnn import (
    SocialV0,
    SocialTrialDataset,
    SocialTrialSource,
    build_social_dataloader,
    compose_edge_inputs,
)

sources = [
    SocialTrialSource(
        trial_id="trial_001",
        node_path=r"D:\social_trials\trial_001\node_features.npz",
        edge_path=r"D:\social_trials\trial_001\trial_001_social_edges.npz",
    ),
    SocialTrialSource(
        trial_id="trial_002",
        node_path=r"D:\social_trials\trial_002\node_features.npz",
        edge_path=r"D:\social_trials\trial_002\trial_002_social_edges.npz",
    ),
]
dataset = SocialTrialDataset(sources)
loader = build_social_dataloader(dataset, batch_size=2, shuffle=True)

model = SocialV0(node_dim=128, edge_dim=16, hidden_dim=64)
for batch in loader:
    edge_features = compose_edge_inputs(
        batch["edge_values"], batch["edge_confidence"]
    )
    g_t = model(
        batch["node_features"],
        edge_features,
        node_mask=batch["node_mask"],
    )
    valid_g_t = g_t[batch["time_mask"]]
```

One Dataset item always equals one full trial. The collate function pads only
the time dimension to the longest trial in the current batch and returns
`time_mask [B,T]`; it never concatenates timelines across trials. `shuffle`
therefore changes only trial order. All padded tensor values and masks are zero,
while `sequence_length`, `identity`, and `trial_id` preserve batch metadata.

The node NPZ must contain `node_features`, `node_mask`, `patch_start_s`,
`patch_end_s`, and `identity`. The edge NPZ may use canonical dense
`edge_values` and `edge_confidence` keys or the existing extractor keys
`edge_value_dense` and `edge_confidence_dense`; it must also contain the same
timestamps and identity order. When an extractor file contains both sparse
`edge_confidence [T,E,8]` and `edge_confidence_dense [T,N,N,8]`, the loader
explicitly selects the dense tensor.

Every Dataset access returns newly allocated tensors. Future SSL code should
keep the returned batch as the unmodified target and create masked/augmented
input clones dynamically, for example:

```python
ssl_node_input = batch["node_features"].clone()
ssl_edge_input = batch["edge_values"].clone()
# Apply random masks to the clones; the original batch remains the target.
```

## Video feature extraction

`video_feature_pipeline.py` launches MouseGPT and idtracker.ai concurrently for
each source video. The two branches intentionally remain identity-independent:

- MouseGPT inference is converted by
  `roughanalysis/mousegpt_head_direction_segments.py` into named keypoints,
  head-direction vectors, and their confidence values.
- idtracker.ai trajectories are converted by `idtracker_export.py` into
  frame-aligned centroid positions, 2-D velocity vectors, speeds, and validity
  masks.
- `idtracker_identity_smoothing.py` can optionally repair a two-animal
  close-contact identity exchange followed by an idtracker.ai recovery jump.
  It swaps the identity channels through the inferred wrong-ID interval and
  interpolates only the short transition windows.

The runnable Windows configuration is
`configs/video_feature_pipeline_v0.yaml`. Review the paths and GUI policy, then
preview every external command without starting inference:

```powershell
C:\Users\DELL\anaconda3\envs\deepof\python.exe -m social_gnn.video_feature_pipeline `
  --config C:\Users\DELL\Desktop\deepOF\configs\video_feature_pipeline_v0.yaml `
  --dry-run
```

Remove `--dry-run` to execute. `idtrackerai.gui_policy` accepts:

- `first_video`: tune the first video in the GUI, automatically snapshot the
  reusable parameters, and run later videos headlessly.
- `every_video`: open the GUI for every video and snapshot each video's actual
  parameters.
- `never`: run headlessly from an existing reusable parameter TOML.

The reusable snapshot deliberately excludes video path, output directory,
session name, and tracking interval. Consequently, a full-video interval and a
new background are computed for every new source video. Cross-tool identity
matching runs after both branches finish.

The extra idtracker.ai repair is disabled by default. Enable it explicitly:

```yaml
idtrackerai:
  smoothing:
    enabled: true
```

When enabled, identity matching automatically uses
`*_idtracker_kinematics_smoothed.csv`. The corresponding NPZ retains raw and
repaired coordinates, while the CSV/JSON event outputs record every swapped or
interpolated frame. This is a two-animal retrospective heuristic, not a
calibrated identity model; inspect representative videos before enabling it for
a large batch.

## Frame-wise identity matching

`identity_matching.py` treats idtracker.ai identity as authoritative and uses
MouseGPT `track_id` only for diagnostics. On every frame it:

1. builds a MouseGPT body anchor from weighted SpineF/SpineG/SpineH/Hip points,
   falling back to the bounding-box centre;
2. computes its distance matrix to all valid idtracker.ai centroids;
3. solves a one-to-one linear assignment independently for that frame; and
4. rejects distant, low-anchor-confidence, or ambiguous assignments.

There is deliberately no temporal identity penalty. After two animals separate,
the next frame immediately follows idtracker.ai's recovered identity. Outputs
include a compact match table, a frame summary, and the complete MouseGPT table
augmented with `idtracker_identity` and `association_confidence`.

`identity_match_visualization.py` is an optional QA-only module. It overlays
idtracker centroids, MouseGPT body anchors, assignment lines, keypoints, bounding
boxes, head direction, distances, and confidence on a scaled copy of the source
video. Disable it for large runs with:

```yaml
identity_matching:
  visualization:
    enabled: false
```

`imu_video_identity.py` provides separate, diagnostic evidence for the
remaining IMU-to-video identity mapping problem. It can estimate a
gravity-orthogonal acceleration signature, compare it to trajectory
acceleration signatures under all one-to-one permutations, and map identities
by the ordering of well-separated entry events. The acceleration estimate is
not a world-coordinate horizontal acceleration without orientation data, and
entry-order evidence must be validated visually before it is used to create a
canonical identity manifest. Neither evidence source automatically rewrites
idtracker.ai identities.

## Sequential-entry identity anchoring

`sequential_entry_identity.py` turns a visually validated sequential-entry
protocol into an auditable identity permutation. For two animals, it detects
or accepts the two IMU impact times, removes configurable margins around both
entries, and inspects the idtracker.ai kinematics rows in the remaining
single-animal interval. The earlier IMU animal becomes canonical animal 1 only
when one original idtracker identity dominates that interval. Minimum window
duration, single-identity frame fraction, selected-ID coverage, competing-ID
coverage, and coverage margin are all fail-closed gates.
The full interval must be present with both identity rows per frame (missing
centroids remain invalid rows). Occupancy here means finite, valid centroid
tracks, not an independent segmentation/blob-count measurement. The arena ROI
must exclude handlers and reflections; inspect the anchor when segmentation
quality is uncertain.

The original idtracker.ai session and IDs are never edited. Outputs are a new
canonical kinematics CSV, a two-row mapping CSV, and a JSON evidence manifest.
The canonical CSV retains `source_idtracker_identity` and adds
`canonical_identity_label` plus the mapping quality score. When this stage is
enabled, MouseGPT matching and edge extraction automatically consume that
canonical CSV.
Canonical numeric labels are `[1, 2]` in entry order. The JSON also records
`canonical_label_order` with the corresponding IMU labels. When converting
nodes, order the actual upstream mouse streams accordingly before assigning
`--output-identity 1 2`; changing labels alone does not reorder latent streams.
Existing node files are not silently relabeled by this stage.

Configure each trial independently because IMU paths and animal labels differ:

```yaml
videos:
  - id: A4_B4
    path: 'Y:\path\to\topview.mp4'
    imu_animals:
      - label: A4_1c
        csv: 'Y:\path\to\A4_B4_A4-1c.csv'
        entry_time_s: 19.923811  # optional visually confirmed override
      - label: B4_ab
        csv: 'Y:\path\to\A4_B4_B4-ab.csv'
        entry_time_s: 57.083811

imu_identity_mapping:
  enabled: true
```

If `entry_time_s` is omitted, the mapper selects the strongest
gravity-compensated acceleration impulse inside the configured early search
window. Manual times are preferable after the synchronized viewer has made the
entry events unambiguous. Ambiguous idtracker windows raise
`SequentialEntryIdentityError`; they are never silently assigned.

Run just this CPU stage against an existing kinematics export:

```powershell
python social_gnn\sequential_entry_identity.py `
  --idtracker-csv D:\trial\A4_B4_idtracker_kinematics.csv `
  --animal 'A4_1c=Y:\WSB\Computational_Ethology\processed_Data\Test\A4_B4_A4-1c.csv' `
  --animal 'B4_ab=Y:\WSB\Computational_Ethology\processed_Data\Test\A4_B4_B4-ab.csv' `
  --entry-time A4_1c=19.923811 --entry-time B4_ab=57.083811 `
  --outdir D:\trial\imu_identity_mapping --output-stem A4_B4
```

With the default margins this uses `[21.923811, 54.083811)` seconds. Remove
both `--entry-time` options to test automatic peak selection. Output confidence
is an anchor quality score, not a calibrated probability or a guarantee that
every later tracking frame has the correct identity.

## Authoritative-clock edge extraction

`edge_extraction.py` uses the one fixed patch grid stored by the node NPZ. A
grid may overlap (for example, 0.32 s duration and 0.16 s stride), but duration
and stride must each remain constant throughout a trial. Variable/mixed patch
granularity and an independently generated video clock are rejected.

Frames are assigned to half-open intervals `[patch_start_s, patch_end_s)`. The
stored contract keeps three arrays separate:

```text
edge_value       [patches, directed_edges, 8]
edge_confidence  [patches, directed_edges, 8]
edge_coverage    [patches, directed_edges, 8]
```

The five state-like traits (three distances, source-facing-target, and head
alignment) use confidence-weighted frame means. Patch confidence is kept as a
separate channel and includes valid-frame coverage.

The three dynamic traits are not averages of frame velocities. Stable
patch-level centroid summaries are built first, then `t -> t+1` finite
differences use

```text
dt = patch_center_s[t+1] - patch_center_s[t]
```

to derive directional source approach speed, symmetric pair closing speed, and
movement-direction alignment. The final patch has no forward transition, so
its dynamic value/confidence/coverage channels are zero. Low displacement
downweights movement-alignment confidence using the configured pixel noise
floor. Missing traits remain zero-confidence; values are never permanently
multiplied by confidence.

The default GNN input remains the 16-channel concatenation
`[edge_value, edge_confidence]`. Coverage and patch centroid summaries are QC
outputs rather than additional V0 model inputs.

Configure each video with its corresponding canonical node clock:

```yaml
videos:
  - id: trial_001
    path: D:\videos\trial_001.mp4
    node_clock_npz: D:\social_trials\trial_001\node_features.npz

edge_extraction:
  enabled: true
  clock_node_npz: null
  minimum_patch_displacement_px: 1.0
```

`edge_extraction.clock_node_npz` is an optional global fallback when all videos
share one clock file. There is no legacy `patch_length_s` mode.
`SocialTrialPackage.from_npz` verifies fixed-grid validity and exact node/edge
identity and patch-time equality before training.
# Relational representation evaluation

Frozen edge sensitivity and information-retention diagnostics are available as
module entry points:

```powershell
python -m social_gnn.evaluate_edge_sensitivity `
  --experiment-config configs/social_gnn_strict_baseline_6dim_v1.yaml `
  --experiment-dir <experiment-directory> `
  --output-dir <output-directory>

python -m social_gnn.evaluate_information_retention `
  --experiment-config configs/social_gnn_strict_baseline_6dim_v1.yaml `
  --experiment-dir <experiment-directory> `
  --validation-trial A4_C4 `
  --test-trial A5_C5 `
  --output-dir <output-directory>
```

The sensitivity evaluator preserves paired directed-edge structure, reports
value and confidence perturbations separately, and normalizes displacement by
the real latent RMS radius. The probe evaluator freezes the encoder, uses only
trial-level splits, counts symmetric edge traits once, and reports both
identity-ordered and permutation-invariant node retention. Neither evaluator
modifies stored `SocialTrialPackage` files.

The message-passing width and exported graph dimension can be controlled
independently. Set `model.hidden_dim` for node/message capacity and
`model.graph_dim` for the final `g_t` width. If `graph_dim` is omitted it
defaults to `hidden_dim`, preserving compatibility with existing checkpoints.

## Graph-level bottleneck SSL

`graph_level_ssl.py` provides the no-bypass objective used by the preliminary
downstream candidate. Its node and edge decoders receive only the exported
`g_t`; they never receive pre-pooling node embeddings. For two animals with
`D_node` channels, the invariant node target has `2 * D_node` channels: the
feature-wise node means plus absolute differences. Edge targets contain
symmetric edge traits once, and direction mean plus absolute asymmetry for
directed traits. `D_node` is inferred from each loaded trial package, so the
current 9D upstream latent and historical 6D packages use the same code path.
Confidence remains a separate loss weight.

The same module contains a capacity-comparable ordered MLP fusion baseline.
It is deliberately retained as a strong diagnostic, but is not suitable as an
identity-permutation-invariant social representation. `SocialV0` also accepts
`readout: soft_role`, an experimental shared-scoring readout that is exactly
invariant to exchanging both animal axes.

Run the validation-selected graph-level mean configuration with:

```powershell
python -m social_gnn.experiment `
  --config configs\social_gnn_selected_graph_mean_6dim_v1.yaml `
  --device cuda:1
```

The selected local checkpoint and complete validation-only interpretation are
recorded in `NEXT_PHASE_REPORT.md`. Information-retention summaries now write
`validation_mean_r2` and `test_mean_r2` separately. Architecture selection
must use the validation column; `A5_C5` is exploratory because it was already
inspected in earlier development.

## Downstream HMM/HSMM state discovery

`social_gnn.downstream` extracts frozen patch-level `g_t` embeddings and fits
continuous-observation diagonal Gaussian HMMs or genuine explicit-duration
Gaussian HSMMs. Trials and invalid temporal runs remain independent, and bout
seconds are derived from patch-center boundaries rather than by summing
possibly overlapping source windows.

```powershell
python -m social_gnn.downstream --help
```

The reproducible command examples and time contract are documented in
`downstream/README.md`. The initial model-selection results, encoder
comparison, synthetic checks, selected checkpoint hashes, and scientific
limitations are in `downstream/DOWNSTREAM_REPORT.md`.

## Node-preserving SSL experiments

`node_preservation_ssl.py` adds three patch-local, permutation-invariant
encoders (social GNN, node-set branch, and fused dual branch), hybrid
full-input/masked objectives, exact two-permutation node-set loss, and
consistent joint node-edge matching. Every reconstruction decoder receives
only final `g_t`; masked node branches receive the corrupted input.

The controlled configurations are:

- `social_gnn_node_preservation_b_pilot_6dim_v1.yaml`: preservation-weight pilot.
- `social_gnn_node_preservation_c_6dim_v1.yaml`: social/node/dual ablation.
- `social_gnn_node_preservation_d_6dim_v1.yaml`: unordered and joint set losses.

Diagnostics are runnable as modules:

```powershell
python -m social_gnn.evaluate_node_stage_retention --help
python -m social_gnn.evaluate_role_binding --help
python -m social_gnn.evaluate_set_reconstruction --help
python -m social_gnn.summarize_node_preservation --help
```

The frozen role-binding evaluator uses a small, predefined set of invariant
cross-trait products and matched real observations. It does not treat these
targets as behavioral labels. The results and selection rationale are in
`NODE_PRESERVATION_REPORT.md`.
