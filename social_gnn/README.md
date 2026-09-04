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
