"""Authoritative node-clock loading for overlapping Social-GNN patches."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np


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

    if starts.ndim != 1 or ends.ndim != 1 or steps.ndim != 1:
        raise ValueError("patch_start_s, patch_end_s and social_step must be 1-D")
    if not len(starts) or not (len(starts) == len(ends) == len(steps)):
        raise ValueError("authoritative clock arrays must have the same non-zero length")
    if not np.isfinite(starts).all() or not np.isfinite(ends).all():
        raise ValueError("authoritative patch times must be finite")
    invalid_duration = np.flatnonzero(ends <= starts)
    if invalid_duration.size:
        index = int(invalid_duration[0])
        raise ValueError(
            f"authoritative patch {index} has non-positive duration "
            f"[{starts[index]}, {ends[index]})"
        )
    if len(starts) > 1 and np.any(np.diff(starts) <= 0):
        index = int(np.flatnonzero(np.diff(starts) <= 0)[0] + 1)
        raise ValueError(f"patch_start_s must be strictly increasing at index {index}")
    if len(ends) > 1 and np.any(np.diff(ends) <= 0):
        index = int(np.flatnonzero(np.diff(ends) <= 0)[0] + 1)
        raise ValueError(f"patch_end_s must be strictly increasing at index {index}")
    if len(np.unique(steps)) != len(steps):
        raise ValueError("social_step values must be unique")

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
        "duration_s_min": float(durations.min()),
        "duration_s_max": float(durations.max()),
        "stride_s_min": float(strides.min()) if strides.size else None,
        "stride_s_max": float(strides.max()) if strides.size else None,
        "overlapping": bool(strides.size and np.any(starts[1:] < ends[:-1])),
    }
    return steps, starts, ends, metadata
