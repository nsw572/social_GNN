"""Authoritative node-clock loading for overlapping Social-GNN patches."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np


def validate_fixed_patch_grid(
    patch_start_s: np.ndarray,
    patch_end_s: np.ndarray,
    *,
    social_step: np.ndarray | None = None,
    label: str = "authoritative clock",
) -> dict[str, Any]:
    """Validate one fixed-duration, fixed-stride physical patch grid."""
    starts = np.asarray(patch_start_s, dtype=np.float64)
    ends = np.asarray(patch_end_s, dtype=np.float64)
    if starts.ndim != 1 or ends.ndim != 1:
        raise ValueError(f"{label}: patch_start_s and patch_end_s must be 1-D")
    if not len(starts) or len(starts) != len(ends):
        raise ValueError(f"{label}: patch arrays must have the same non-zero length")
    if not np.isfinite(starts).all() or not np.isfinite(ends).all():
        raise ValueError(f"{label}: patch times must be finite")
    durations = ends - starts
    invalid_duration = np.flatnonzero(durations <= 0)
    if invalid_duration.size:
        index = int(invalid_duration[0])
        raise ValueError(
            f"{label}: patch {index} has non-positive duration "
            f"[{starts[index]}, {ends[index]})"
        )
    if not np.allclose(durations, durations[0], rtol=1e-9, atol=1e-9):
        index = int(np.flatnonzero(~np.isclose(
            durations, durations[0], rtol=1e-9, atol=1e-9
        ))[0])
        raise ValueError(
            f"{label}: variable patch duration at index {index}; "
            "Social-GNN requires one fixed patch granularity"
        )
    centers = (starts + ends) / 2.0
    strides = np.diff(centers)
    if strides.size and np.any(strides <= 0):
        index = int(np.flatnonzero(strides <= 0)[0] + 1)
        raise ValueError(f"{label}: patch centers are not increasing at index {index}")
    if strides.size and not np.allclose(
        strides, strides[0], rtol=1e-9, atol=1e-9
    ):
        index = int(np.flatnonzero(~np.isclose(
            strides, strides[0], rtol=1e-9, atol=1e-9
        ))[0] + 1)
        raise ValueError(
            f"{label}: variable patch stride at index {index}; "
            "Social-GNN requires one fixed patch grid"
        )
    if social_step is not None:
        steps = np.asarray(social_step, dtype=np.int64)
        if steps.ndim != 1 or len(steps) != len(starts):
            raise ValueError(f"{label}: social_step must align with patch arrays")
        if len(steps) > 1 and not np.all(np.diff(steps) == 1):
            raise ValueError(f"{label}: social_step must be consecutive")
    return {
        "patch_duration_s": float(durations[0]),
        "patch_stride_s": float(strides[0]) if strides.size else None,
        "overlapping": bool(strides.size and strides[0] < durations[0]),
    }


def load_node_clock(
    node_npz: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load validated patch intervals without assuming stride equals duration."""
    node_npz = Path(node_npz)
    try:
        with np.load(node_npz, allow_pickle=False) as payload:
            missing = {
                "patch_start_s",
                "patch_end_s",
            } - set(payload.files)
            if missing:
                raise ValueError(f"node NPZ is missing arrays {sorted(missing)}")
            starts = np.array(payload["patch_start_s"], dtype=np.float64, copy=True)
            ends = np.array(payload["patch_end_s"], dtype=np.float64, copy=True)
            steps = (
                np.array(payload["social_step"], dtype=np.int64, copy=True)
                if "social_step" in payload.files
                else np.arange(len(starts), dtype=np.int64)
            )
            trial_id = (
                str(np.asarray(payload["trial_id"]).reshape(-1)[0])
                if "trial_id" in payload.files
                else node_npz.stem
            )
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load authoritative node clock {node_npz}: {exc}") from exc

    grid = validate_fixed_patch_grid(
        starts, ends, social_step=steps, label="authoritative node clock"
    )

    digest = hashlib.sha256()
    digest.update(starts.tobytes(order="C"))
    digest.update(ends.tobytes(order="C"))
    durations = ends - starts
    strides = np.diff(starts)
    metadata = {
        "clock_mode": "authoritative_node_intervals",
        "clock_node_npz": str(node_npz.resolve()),
        "clock_sha256": digest.hexdigest(),
        "clock_source_size": node_npz.stat().st_size,
        "clock_source_mtime_ns": node_npz.stat().st_mtime_ns,
        "trial_id": trial_id,
        "patch_count": int(len(starts)),
        **grid,
        "duration_s_min": float(durations.min()),
        "duration_s_max": float(durations.max()),
        "stride_s_min": float(strides.min()) if strides.size else None,
        "stride_s_max": float(strides.max()) if strides.size else None,
    }
    return steps, starts, ends, metadata
