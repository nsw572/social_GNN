"""Generate exactly the three State Atlas v1 visualization components."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .state_atlas_fingerprint import generate_state_fingerprint
from .state_atlas_pose_validity import generate_pose_validity_matrix
from .state_atlas_timeline import DEFAULT_CHANNELS, generate_timeline_explorer


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("PyYAML is required") from error
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return payload


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _write_readme(output: Path, manifest: dict[str, Any]) -> Path:
    timeline = Path(manifest["components"]["state_timeline_explorer"]["index"])
    fingerprint = manifest["components"]["state_fingerprint"]
    pose = manifest["components"]["state_pose_validity_matrix"]
    content = f"""# State Atlas v1 visualizations

These are three read-only views over the frozen 0.04 s Graph Full -> PCA16 ->
Sticky HMM K=12 State Atlas. They do not retrain or refit anything, do not
require source videos, and do not assign biological behavior names.

## 1. State Timeline Explorer

Open `{timeline.name}` in `timeline_explorer/`, choose a trial, then select a
physical-time window and displayed channels. Each trial page is a standalone
Plotly HTML with checkboxes, zoom/pan, raw scale, per-window min-max scale,
median/IQR robust z-score scale, and optional percentile clipping. HMM states
are categorical strips; missing traits remain gaps; directions remain explicit.

## 2. State Fingerprint

- Summary table: `{Path(fingerprint['summary']).name}`
- Standardized heatmap: `{Path(fingerprint['heatmap']).name}`
- Distribution violins: `{Path(fingerprint['distributions']).name}`

The heatmap color uses each channel's cross-state median/IQR normalization, but
cell text remains the raw state median and the full raw statistics are in CSV.
Distribution violins use deterministic finite-value samples only for drawing;
the CSV statistics use every available observation. Symmetric traits are
counted once and directed traits remain 0->1 / 1->0.

## 3. State × Pose Validity Matrix

- Full aggregate + trial-stratified table: `{Path(pose['summary']).name}`
- Absolute validity matrix: `{Path(pose['absolute']).name}`
- Delta from each channel's global baseline: `{Path(pose['delta_from_global']).name}`
- Confidence among valid observations: `{Path(pose['confidence_on_valid']).name}`

Validity and volatility are kept separate. Lower validity is a diagnostic for
manual review, not proof that a state is artifactual, and this decomposition
does not replace `pose_quality`.

## Reproduce

```powershell
conda run -n deepof python -m social_gnn.downstream state-atlas-viz `
  --config configs\\social_gnn_state_atlas_visualizations_v1.yaml
```

Scaling and clipping in the Timeline Explorer are display-only. No stored raw
trait is filled with zero or multiplied by confidence.
"""
    path = output / "README.md"
    path.write_text(content, encoding="utf-8")
    return path


def generate_state_atlas_visualizations(
    config_path: str | Path,
    *,
    output_override: str | Path | None = None,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    config = _load_yaml(config_file)
    project_root = _resolve(config_file.parent, config.get("project_root", ".."))
    atlas_root = _resolve(project_root, config["atlas_root"])
    output = (
        Path(output_override).resolve()
        if output_override is not None
        else _resolve(project_root, config["output_root"])
    )
    trial_ids = tuple(str(value) for value in config["trial_ids"])
    expected = ("A1_C1", "A3_C3", "A4_B4", "A4_C4")
    if trial_ids != expected or "A5_C5" in trial_ids:
        raise ValueError(f"visualization trial contract must be {expected}")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_file, output / "visualization_config.yaml")

    timeline_config = config.get("timeline_explorer", {})
    timeline = generate_timeline_explorer(
        atlas_root,
        output / "timeline_explorer",
        trial_ids=trial_ids,
        default_window_s=float(timeline_config.get("default_window_s", 120.0)),
        default_channels=tuple(
            str(value)
            for value in timeline_config.get(
                "default_channels",
                DEFAULT_CHANNELS,
            )
        ),
    )
    fingerprint_config = config.get("state_fingerprint", {})
    fingerprint = generate_state_fingerprint(
        atlas_root,
        output / "state_fingerprint",
        trial_ids=trial_ids,
        sample_per_state=int(fingerprint_config.get("violin_sample_per_state", 2500)),
        random_seed=int(fingerprint_config.get("random_seed", 20261102)),
    )
    pose = generate_pose_validity_matrix(
        atlas_root,
        output / "pose_validity_matrix",
        trial_ids=trial_ids,
    )
    manifest = {
        "contract": "social_gnn_state_atlas_visualizations_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "atlas_root": str(atlas_root),
        "trials": list(trial_ids),
        "excluded_trial": "A5_C5",
        "components": {
            "state_timeline_explorer": timeline,
            "state_fingerprint": fingerprint,
            "state_pose_validity_matrix": pose,
        },
        "scientific_constraints": [
            "read-only frozen atlas",
            "missing values remain explicit",
            "raw values are not multiplied by confidence",
            "symmetric summaries are not double-counted",
            "directional channels remain explicit",
            "validity is not interpreted as volatility or causal artifact",
        ],
    }
    readme = _write_readme(output, manifest)
    manifest["readme"] = str(readme.resolve())
    manifest_path = output / "visualization_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = generate_state_atlas_visualizations(
        args.config, output_override=args.output_dir
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["generate_state_atlas_visualizations"]
