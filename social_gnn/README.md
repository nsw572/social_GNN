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
social_latent = model(h, edges)                 # [8, 100, 64]
```

The V0 implementation uses only PyTorch for message passing. This is
deliberate: PyTorch Geometric is not installed in the current interpreter.
Once the input contract is validated on real data, the layer can be swapped
for `GINEConv` without changing the dataset or downstream representation API.

## Temporal social state

`SocialGNNWithTCN` implements the currently available part of the intended
training path:

```text
Graph sequence -> patch-wise GNN -> causal TCN -> s_t -> future SSL decoder
```

The GNN first produces an independent graph embedding `g_t` for every fixed
social patch. A residual TCN then computes

```text
s_t = TCN(g_1, ..., g_t)
```

using left-only padding, so changing a future patch cannot change a past
state. Dilations grow geometrically across TCN levels. Because each residual
block contains two convolutions, its receptive field in patches is

```text
1 + 2 * (kernel_size - 1) * sum(dilations).
```

The default four levels use dilations `(1, 2, 4, 8)` and kernel size `3`, for a
61-patch receptive field. A `[B,T]` time mask supports padded trials; masked
states are zeroed after every block and cannot inject values into later valid
states.

```python
from social_gnn import SocialGNNWithTCN

encoder = SocialGNNWithTCN(
    node_dim=128,
    edge_dim=16,
    graph_hidden_dim=64,
    temporal_hidden_dim=64,
    tcn_levels=4,
)
s_t = encoder(h, edges, node_mask=node_mask, time_mask=time_mask)
# s_t: [batch, social_patches, 64]
```

The SSL decoder, prediction target, and loss are intentionally outside this
module until the self-supervised objective is selected. They should consume
`s_t` without changing the graph/temporal encoder contract.

## Full-trial data loading

`SocialTrialPackage` represents one complete trial. It validates all arrays
before exposing them to PyTorch and raises `SocialTrialValidationError` with
the trial ID when node/edge clocks, identity order, shapes, finite values, or
confidence ranges disagree. Stored NumPy arrays are copied and made read-only.

For large datasets, use `SocialTrialSource` so each node/edge NPZ pair is loaded
lazily by the Dataset worker instead of loading every trial into RAM:

```python
import torch
from social_gnn import (
    SocialGNNWithTCN,
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

model = SocialGNNWithTCN(node_dim=128, edge_dim=16)
for batch in loader:
    edge_features = compose_edge_inputs(
        batch["edge_values"], batch["edge_confidence"]
    )
    s_t = model(
        batch["node_features"],
        edge_features,
        node_mask=batch["node_mask"],
        time_mask=batch["time_mask"],
    )
```

One Dataset item always equals one full trial. The collate function pads only
the time dimension to the longest trial in the current batch and returns
`time_mask [B,T]`; it never concatenates timelines across trials. `shuffle`
therefore changes only trial order. All padded tensor values and masks are zero,
while `sequence_length`, `identity`, and `trial_id` preserve batch metadata.

The node NPZ must contain `node_features`, `node_mask`, `patch_start_s`,
`patch_end_s`, and `identity`. The edge NPZ may use canonical `edge_values` and
`edge_confidence` keys or the existing extractor keys `edge_value_dense` and
`edge_confidence_dense`; it must also contain the same timestamps and identity
order.

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

## Fixed-clock edge extraction

`edge_extraction.py` computes eight directed frame-level relations and then
aggregates them into fixed social patches. `patch_length_s` is deliberately a
required runtime value: set it to the minimum upstream node-patch duration.
No DeepOF/video window length is used as a substitute.

Frames are assigned to half-open intervals `[patch_start_s, patch_end_s)`. The
stored contract keeps three arrays separate:

```text
edge_value       [patches, directed_edges, 8]
edge_confidence  [patches, directed_edges, 8]
edge_coverage    [patches, directed_edges, 8]
```

Patch values are confidence-weighted means. Patch confidence is the sum of
valid frame confidences divided by all video frames in the patch, and coverage
is the valid-frame fraction. Missing traits have value/confidence/coverage zero;
the physical value is never multiplied by confidence in the saved contract.
Because patch confidence already includes coverage, the default GNN input is
the 16-channel concatenation `[edge_value, edge_confidence]`. The separately
stored coverage tensor is a QC/ablation signal and is not duplicated in V0.

Enable the final pipeline stage only after supplying the upstream duration:

```yaml
edge_extraction:
  enabled: true
  patch_length_s: 0.125  # example only; use the actual upstream value
```

Set `videos[].social_patch_count` when the authoritative number of node
timesteps is available. Otherwise the extractor emits only complete patches
covered by the video and drops an incomplete tail.
