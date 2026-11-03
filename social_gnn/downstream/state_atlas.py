"""State Atlas v1 for the frozen 0.04 s Social-GNN Sticky-HMM pipeline.

The atlas is a read-only characterization/QC layer.  It decodes existing
Graph-Full embeddings with an already fitted scaler/PCA and HMM, aligns the
result to existing edge measurements, and creates quantitative/video-review
artifacts.  It does not fit or modify any model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .embeddings import FrozenEmbeddingTrial
from .evaluation import DecodedTrial, bout_rows, decode_trial
from .gaussian_hmm import GaussianHMM
from .preprocessing import EmbeddingPreprocessor
from .state_atlas_video import (
    attach_clip_paths,
    plan_state_clips,
    render_clip_catalog,
    write_extraction_scripts,
)


TRAIT_NAMES = (
    "centroid_distance",
    "nose_to_nose_distance",
    "source_nose_to_target_centroid_distance",
    "source_faces_target",
    "head_direction_alignment",
    "source_radial_approach_speed",
    "pair_closing_speed",
    "movement_direction_alignment",
)
ATLAS_CHANNEL_NAMES = TRAIT_NAMES + ("dynamic_quality", "pose_quality")
SYMMETRIC_TRAITS = (0, 1, 4, 6, 7)
DIRECTED_TRAITS = (2, 3, 5)
DYNAMIC_TRAITS = (5, 6, 7)
POSE_TRAITS = (1, 2, 3, 4)
DEFAULT_TRIALS = ("A1_C1", "A3_C3", "A4_B4", "A4_C4")
LOCKED_TRIAL = "A5_C5"


@dataclass(frozen=True)
class TrialAtlas:
    trial_id: str
    identity: np.ndarray
    patch_start_s: np.ndarray
    patch_end_s: np.ndarray
    valid_time_mask: np.ndarray
    predicted_state: np.ndarray
    posterior: np.ndarray
    state_confidence: np.ndarray
    bout_id: np.ndarray
    bout_start_s: np.ndarray
    bout_end_s: np.ndarray
    bout_duration_s: np.ndarray
    boundary_adjacent: np.ndarray
    boundary_distance_s: np.ndarray
    edge_values: np.ndarray
    edge_confidence: np.ndarray
    edge_coverage: np.ndarray
    edge_valid: np.ndarray
    atlas_edge_tensor: np.ndarray
    dynamic_quality: np.ndarray
    pose_quality: np.ndarray
    dynamic_sign_flip: np.ndarray
    dynamic_abs_jump: np.ndarray

    def __post_init__(self) -> None:
        trial = self.trial_id or "<unknown>"
        length = len(self.patch_start_s)
        if self.patch_end_s.shape != (length,):
            raise ValueError(f"{trial}: patch clock shape mismatch")
        if self.valid_time_mask.shape != (length,):
            raise ValueError(f"{trial}: valid_time_mask must be [T]")
        if self.predicted_state.shape != (length,):
            raise ValueError(f"{trial}: predicted_state must be [T]")
        if self.posterior.shape[0] != length or self.posterior.ndim != 2:
            raise ValueError(f"{trial}: posterior must be [T,K]")
        number_of_nodes = len(self.identity)
        raw_shape = (length, number_of_nodes, number_of_nodes, len(TRAIT_NAMES))
        for name in ("edge_values", "edge_confidence", "edge_coverage", "edge_valid"):
            if getattr(self, name).shape != raw_shape:
                raise ValueError(f"{trial}: {name} must have shape {raw_shape}")
        if self.atlas_edge_tensor.shape != raw_shape[:-1] + (10,):
            raise ValueError(f"{trial}: atlas_edge_tensor must be [T,N,N,10]")
        if self.dynamic_quality.shape != raw_shape[:-1] or self.pose_quality.shape != raw_shape[:-1]:
            raise ValueError(f"{trial}: quality tensors must be [T,N,N]")
        if self.dynamic_sign_flip.shape != raw_shape[:-1] + (3,):
            raise ValueError(f"{trial}: dynamic_sign_flip must be [T,N,N,3]")
        if self.dynamic_abs_jump.shape != raw_shape[:-1] + (3,):
            raise ValueError(f"{trial}: dynamic_abs_jump must be [T,N,N,3]")
        if np.any((self.edge_confidence < 0) | (self.edge_confidence > 1)):
            raise ValueError(f"{trial}: edge confidence outside [0,1]")
        if np.any((self.edge_coverage < 0) | (self.edge_coverage > 1)):
            raise ValueError(f"{trial}: edge coverage outside [0,1]")

    @property
    def patch_center_s(self) -> np.ndarray:
        return 0.5 * (self.patch_start_s + self.patch_end_s)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - runtime dependency guard
        raise RuntimeError("PyYAML is required for State Atlas configuration") from error
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _path_available(path: str | Path) -> bool:
    try:
        return Path(path).exists()
    except OSError:
        return False


def _safe_quantile(values: np.ndarray, q: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(finite, q)) if finite.size else float("nan")


def _trimmed_mean(values: np.ndarray, fraction: float = 0.1) -> float:
    finite = np.sort(np.asarray(values, dtype=float)[np.isfinite(values)])
    if finite.size == 0:
        return float("nan")
    trim = int(math.floor(fraction * finite.size))
    if 2 * trim >= finite.size:
        return float(np.mean(finite))
    return float(np.mean(finite[trim : finite.size - trim]))


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(frame) < 3 or frame.left.nunique() < 2 or frame.right.nunique() < 2:
        return float("nan")
    return float(frame.left.rank(method="average").corr(frame.right.rank(method="average")))


def compute_quality_channels(
    edge_confidence: np.ndarray,
    edge_coverage: np.ndarray,
    *,
    edge_values: np.ndarray | None = None,
    valid_time_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-trait support, dynamic_quality, and pose_quality.

    Per-trait support is ``sqrt(confidence * coverage)``.  Both summary channels
    are arithmetic means across their declared traits; unavailable traits have
    zero support and therefore reduce quality.  Value magnitude is never used.
    Movement-alignment reliability enters through its existing confidence and
    coverage, which already encode the motion-noise-floor validity logic.
    """

    confidence = np.asarray(edge_confidence, dtype=np.float64)
    coverage = np.asarray(edge_coverage, dtype=np.float64)
    if confidence.shape != coverage.shape or confidence.ndim != 4 or confidence.shape[-1] != 8:
        raise ValueError("confidence and coverage must both have shape [T,N,N,8]")
    if not np.isfinite(confidence).all() or not np.isfinite(coverage).all():
        raise ValueError("confidence/coverage contain nonfinite values")
    if np.any((confidence < 0) | (confidence > 1)) or np.any((coverage < 0) | (coverage > 1)):
        raise ValueError("confidence/coverage must be in [0,1]")
    support = np.sqrt(confidence * coverage)
    if edge_values is not None:
        support = np.where(np.isfinite(edge_values), support, 0.0)
    dynamic = np.mean(support[..., list(DYNAMIC_TRAITS)], axis=-1)
    pose = np.mean(support[..., list(POSE_TRAITS)], axis=-1)
    if valid_time_mask is not None:
        valid = np.asarray(valid_time_mask, dtype=bool)
        if valid.shape != (confidence.shape[0],):
            raise ValueError("valid_time_mask must have shape [T]")
        support[~valid] = 0.0
        dynamic[~valid] = 0.0
        pose[~valid] = 0.0
    number_of_nodes = confidence.shape[1]
    diagonal = np.arange(number_of_nodes)
    support[:, diagonal, diagonal] = 0.0
    dynamic[:, diagonal, diagonal] = np.nan
    pose[:, diagonal, diagonal] = np.nan
    return support, dynamic, pose


def compute_dynamic_diagnostics(
    edge_values: np.ndarray,
    edge_valid: np.ndarray,
    patch_center_s: np.ndarray,
    valid_time_mask: np.ndarray,
    *,
    gap_factor: float = 1.5,
    zero_tolerance: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw sign-flip indicators and absolute jumps for dynamic traits.

    Diagnostics are NaN unless adjacent points are physically contiguous and
    both trait observations are valid.  They are not folded into quality.
    """

    values = np.asarray(edge_values, dtype=np.float64)[..., list(DYNAMIC_TRAITS)]
    observed = np.asarray(edge_valid, dtype=bool)[..., list(DYNAMIC_TRAITS)]
    centers = np.asarray(patch_center_s, dtype=np.float64)
    time_valid = np.asarray(valid_time_mask, dtype=bool)
    output_shape = values.shape
    sign_flip = np.full(output_shape, np.nan, dtype=np.float32)
    abs_jump = np.full(output_shape, np.nan, dtype=np.float32)
    if len(centers) < 2:
        return sign_flip, abs_jump
    stride = float(np.median(np.diff(centers)))
    contiguous = (
        time_valid[1:]
        & time_valid[:-1]
        & (np.diff(centers) <= stride * gap_factor)
    )
    pair_valid = observed[1:] & observed[:-1] & contiguous[:, None, None, None]
    difference = np.abs(values[1:] - values[:-1])
    flips = (
        (values[1:] * values[:-1] < 0)
        & (np.abs(values[1:]) > zero_tolerance)
        & (np.abs(values[:-1]) > zero_tolerance)
    )
    abs_jump[1:] = np.where(pair_valid, difference, np.nan)
    sign_flip[1:] = np.where(pair_valid, flips.astype(np.float32), np.nan)
    return sign_flip, abs_jump


def _bout_fields(decoded: DecodedTrial, boundary_window_s: float) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    length = len(decoded.predicted_state)
    fields = {
        "bout_id": np.full(length, -1, dtype=np.int64),
        "bout_start_s": np.full(length, np.nan),
        "bout_end_s": np.full(length, np.nan),
        "bout_duration_s": np.full(length, np.nan),
    }
    rows = bout_rows(decoded)
    for row in rows:
        index = np.arange(int(row["start_index"]), int(row["end_index_inclusive"]) + 1)
        fields["bout_id"][index] = int(row["bout_id"])
        fields["bout_start_s"][index] = float(row["bout_start_s"])
        fields["bout_end_s"][index] = float(row["bout_end_s"])
        fields["bout_duration_s"][index] = float(row["bout_duration_s"])
    center = decoded.trial.patch_center_s
    boundaries: list[dict[str, Any]] = []
    for run_id, indices in enumerate(decoded.run_indices):
        states = decoded.predicted_state[indices]
        changed = np.flatnonzero(states[1:] != states[:-1])
        for local in changed:
            left = int(indices[local])
            right = int(indices[local + 1])
            boundaries.append(
                {
                    "trial_id": decoded.trial.trial_id,
                    "run_id": run_id,
                    "boundary_index": len(boundaries),
                    "left_patch_index": left,
                    "right_patch_index": right,
                    "boundary_time_s": float(0.5 * (center[left] + center[right])),
                    "from_state": int(decoded.predicted_state[left]),
                    "to_state": int(decoded.predicted_state[right]),
                    "boundary_state_confidence": float(
                        0.5 * (decoded.state_confidence[left] + decoded.state_confidence[right])
                    ),
                }
            )
    distance = np.full(length, np.inf, dtype=np.float64)
    for row in boundaries:
        distance = np.minimum(distance, np.abs(center - float(row["boundary_time_s"])))
    distance[~decoded.trial.valid_time_mask] = np.nan
    fields["boundary_distance_s"] = distance
    fields["boundary_adjacent"] = np.isfinite(distance) & (distance <= boundary_window_s + 1e-12)
    return fields, pd.DataFrame(rows), pd.DataFrame(boundaries)


def build_trial_atlas(
    *,
    decoded: DecodedTrial,
    edge_npz: str | Path,
    boundary_window_s: float,
) -> tuple[TrialAtlas, pd.DataFrame, pd.DataFrame]:
    """Strictly align one decoded trial to its raw edge file."""

    edge_path = Path(edge_npz)
    with np.load(edge_path, allow_pickle=False) as data:
        names = tuple(str(value) for value in data["trait_names"].tolist())
        if names != TRAIT_NAMES:
            raise ValueError(f"{decoded.trial.trial_id}: unexpected trait order {names}")
        start = np.asarray(data["patch_start_s"], dtype=np.float64)
        end = np.asarray(data["patch_end_s"], dtype=np.float64)
        identity = np.asarray(data["identity"])
        values = np.asarray(data["edge_value_dense"], dtype=np.float32)
        confidence = np.asarray(data["edge_confidence_dense"], dtype=np.float32)
        coverage = np.asarray(data["edge_coverage_dense"], dtype=np.float32)
    trial_id = decoded.trial.trial_id
    if not np.array_equal(start, decoded.trial.patch_start_s) or not np.array_equal(end, decoded.trial.patch_end_s):
        raise ValueError(f"{trial_id}: edge and embedding clocks differ")
    if values.shape != (len(start), len(identity), len(identity), 8):
        raise ValueError(f"{trial_id}: dense edge tensor has wrong shape {values.shape}")
    support, dynamic_quality, pose_quality = compute_quality_channels(
        confidence,
        coverage,
        edge_values=values,
        valid_time_mask=decoded.trial.valid_time_mask,
    )
    off_diagonal = ~np.eye(len(identity), dtype=bool)[None, :, :, None]
    edge_valid = (
        np.isfinite(values)
        & (confidence > 0)
        & (coverage > 0)
        & decoded.trial.valid_time_mask[:, None, None, None]
        & off_diagonal
    )
    sign_flip, abs_jump = compute_dynamic_diagnostics(
        values,
        edge_valid,
        decoded.trial.patch_center_s,
        decoded.trial.valid_time_mask,
    )
    atlas = np.full(values.shape[:-1] + (10,), np.nan, dtype=np.float32)
    atlas[..., :8] = np.where(edge_valid, values, np.nan)
    atlas[..., 8] = dynamic_quality
    atlas[..., 9] = pose_quality
    fields, bouts, boundaries = _bout_fields(decoded, boundary_window_s)
    result = TrialAtlas(
        trial_id=trial_id,
        identity=identity,
        patch_start_s=start,
        patch_end_s=end,
        valid_time_mask=np.asarray(decoded.trial.valid_time_mask, dtype=bool),
        predicted_state=np.asarray(decoded.predicted_state, dtype=np.int64),
        posterior=np.asarray(decoded.posterior, dtype=np.float32),
        state_confidence=np.asarray(decoded.state_confidence, dtype=np.float32),
        bout_id=fields["bout_id"],
        bout_start_s=fields["bout_start_s"],
        bout_end_s=fields["bout_end_s"],
        bout_duration_s=fields["bout_duration_s"],
        boundary_adjacent=fields["boundary_adjacent"],
        boundary_distance_s=fields["boundary_distance_s"],
        edge_values=values,
        edge_confidence=confidence,
        edge_coverage=coverage,
        edge_valid=edge_valid,
        atlas_edge_tensor=atlas,
        dynamic_quality=dynamic_quality.astype(np.float32),
        pose_quality=pose_quality.astype(np.float32),
        dynamic_sign_flip=sign_flip,
        dynamic_abs_jump=abs_jump,
    )
    return result, bouts, boundaries


def _summary(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return {
        "valid_count": int(finite.size),
        "mean": float(np.mean(finite)) if finite.size else float("nan"),
        "trimmed_mean_10pct": _trimmed_mean(finite),
        "median": _safe_quantile(finite, 0.5),
        "q25": _safe_quantile(finite, 0.25),
        "q75": _safe_quantile(finite, 0.75),
        "iqr": _safe_quantile(finite, 0.75) - _safe_quantile(finite, 0.25),
        "p10": _safe_quantile(finite, 0.10),
        "p90": _safe_quantile(finite, 0.90),
    }


def _edge_series(
    trial: TrialAtlas,
    state: int | None,
    trait_index: int,
    source: int,
    target: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    mask = trial.valid_time_mask.copy()
    if state is not None:
        mask &= trial.predicted_state == state
    valid = mask & trial.edge_valid[:, source, target, trait_index]
    return (
        trial.edge_values[valid, source, target, trait_index],
        trial.edge_confidence[valid, source, target, trait_index],
        trial.edge_coverage[valid, source, target, trait_index],
        int(mask.sum()),
    )


def state_trait_summary(trials: Sequence[TrialAtlas], n_states: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for state in range(n_states):
        for trait_index, trait_name in enumerate(TRAIT_NAMES):
            edges = [(0, 1)] if trait_index in SYMMETRIC_TRAITS else [(0, 1), (1, 0)]
            for source, target in edges:
                values: list[np.ndarray] = []
                confidences: list[np.ndarray] = []
                coverages: list[np.ndarray] = []
                candidate_count = 0
                for trial in trials:
                    value, confidence, coverage, candidates = _edge_series(
                        trial, state, trait_index, source, target
                    )
                    values.append(value)
                    confidences.append(confidence)
                    coverages.append(coverage)
                    candidate_count += candidates
                merged = np.concatenate(values) if values else np.asarray([])
                record: dict[str, Any] = {
                    "state": state,
                    "trait_index": trait_index,
                    "trait": trait_name,
                    "edge_scope": "symmetric_once" if trait_index in SYMMETRIC_TRAITS else "directed",
                    "source_index": source,
                    "target_index": target,
                    "candidate_points": candidate_count,
                    **_summary(merged),
                }
                record["valid_fraction"] = record["valid_count"] / max(candidate_count, 1)
                record["confidence_median"] = _safe_quantile(np.concatenate(confidences), 0.5)
                record["coverage_median"] = _safe_quantile(np.concatenate(coverages), 0.5)
                rows.append(record)
    return pd.DataFrame(rows)


def _quality_thresholds(trials: Sequence[TrialAtlas]) -> dict[str, float]:
    dynamic: list[np.ndarray] = []
    pose: list[np.ndarray] = []
    for trial in trials:
        mask = trial.valid_time_mask
        dynamic.extend([trial.dynamic_quality[mask, 0, 1], trial.dynamic_quality[mask, 1, 0]])
        pose.extend([trial.pose_quality[mask, 0, 1], trial.pose_quality[mask, 1, 0]])
    dynamic_values = np.concatenate(dynamic)
    pose_values = np.concatenate(pose)
    return {
        "dynamic_low_q20": _safe_quantile(dynamic_values, 0.20),
        "dynamic_high_q80": _safe_quantile(dynamic_values, 0.80),
        "pose_low_q20": _safe_quantile(pose_values, 0.20),
        "pose_high_q80": _safe_quantile(pose_values, 0.80),
    }


def state_quality_summary(
    trials: Sequence[TrialAtlas], n_states: int, thresholds: dict[str, float]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for state in range(n_states):
        for source, target in ((0, 1), (1, 0)):
            state_confidence: list[np.ndarray] = []
            bout_duration: list[np.ndarray] = []
            for quality_name in ("dynamic_quality", "pose_quality"):
                values: list[np.ndarray] = []
                confidence: list[np.ndarray] = []
                duration: list[np.ndarray] = []
                candidate_count = 0
                for trial in trials:
                    mask = trial.valid_time_mask & (trial.predicted_state == state)
                    quality = getattr(trial, quality_name)[mask, source, target]
                    values.append(quality)
                    confidence.append(trial.state_confidence[mask])
                    duration.append(trial.bout_duration_s[mask])
                    candidate_count += int(mask.sum())
                merged = np.concatenate(values)
                merged_conf = np.concatenate(confidence)
                merged_duration = np.concatenate(duration)
                low = thresholds[
                    "dynamic_low_q20" if quality_name == "dynamic_quality" else "pose_low_q20"
                ]
                high = thresholds[
                    "dynamic_high_q80" if quality_name == "dynamic_quality" else "pose_high_q80"
                ]
                finite = np.isfinite(merged)
                record = {
                    "state": state,
                    "quality": quality_name,
                    "source_index": source,
                    "target_index": target,
                    "candidate_points": candidate_count,
                    **_summary(merged),
                    "usable_support_fraction": float(np.mean(merged[finite] > 0)) if finite.any() else 0.0,
                    "low_threshold": low,
                    "high_threshold": high,
                    "low_quality_fraction": float(np.mean(merged[finite] <= low)) if finite.any() else float("nan"),
                    "high_quality_fraction": float(np.mean(merged[finite] >= high)) if finite.any() else float("nan"),
                    "spearman_with_state_confidence": _spearman(merged, merged_conf),
                    "spearman_with_bout_duration": _spearman(merged, merged_duration),
                }
                rows.append(record)
    return pd.DataFrame(rows)


def _enrich_bouts(
    trials: Sequence[TrialAtlas], bout_frames: Sequence[pd.DataFrame]
) -> pd.DataFrame:
    by_trial = {trial.trial_id: trial for trial in trials}
    frames: list[pd.DataFrame] = []
    for frame in bout_frames:
        if frame.empty:
            continue
        value = frame.copy()
        trial = by_trial[str(value.trial_id.iloc[0])]
        dynamic: list[float] = []
        pose: list[float] = []
        for row in value.itertuples():
            index = np.arange(int(row.start_index), int(row.end_index_inclusive) + 1)
            dynamic.append(
                float(np.nanmedian(np.stack([trial.dynamic_quality[index, 0, 1], trial.dynamic_quality[index, 1, 0]])))
            )
            pose.append(
                float(np.nanmedian(np.stack([trial.pose_quality[index, 0, 1], trial.pose_quality[index, 1, 0]])))
            )
        value["median_dynamic_quality"] = dynamic
        value["median_pose_quality"] = pose
        frames.append(value)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def state_bout_summary(
    trials: Sequence[TrialAtlas], bouts: pd.DataFrame, n_states: int, *, short_bout_s: float
) -> pd.DataFrame:
    total_valid_points = sum(int(trial.valid_time_mask.sum()) for trial in trials)
    total_valid_minutes = sum(
        float(np.sum((trial.patch_end_s - trial.patch_start_s)[trial.valid_time_mask]))
        for trial in trials
    ) / 60.0
    rows: list[dict[str, Any]] = []
    for state in range(n_states):
        state_bouts = bouts[bouts.state.astype(int).eq(state)]
        durations = state_bouts.bout_duration_s.to_numpy(float)
        point_confidence = np.concatenate(
            [trial.state_confidence[trial.valid_time_mask & (trial.predicted_state == state)] for trial in trials]
        )
        points = sum(
            int(np.sum(trial.valid_time_mask & (trial.predicted_state == state)))
            for trial in trials
        )
        rows.append(
            {
                "state": state,
                "occupancy_fraction": points / max(total_valid_points, 1),
                "total_valid_points": points,
                "total_bouts": int(len(state_bouts)),
                "bouts_per_recording_minute": len(state_bouts) / max(total_valid_minutes, 1e-12),
                "bouts_per_state_minute": len(state_bouts) / max(float(np.sum(durations)) / 60.0, 1e-12),
                "mean_bout_duration_s": float(np.mean(durations)) if len(durations) else float("nan"),
                "median_bout_duration_s": _safe_quantile(durations, 0.5),
                "bout_duration_p10_s": _safe_quantile(durations, 0.10),
                "bout_duration_p90_s": _safe_quantile(durations, 0.90),
                "short_bout_threshold_s": short_bout_s,
                "short_bout_fraction": float(np.mean(durations <= short_bout_s)) if len(durations) else float("nan"),
                "state_confidence_median": _safe_quantile(point_confidence, 0.5),
                "state_confidence_iqr": _safe_quantile(point_confidence, 0.75) - _safe_quantile(point_confidence, 0.25),
                "state_confidence_p10": _safe_quantile(point_confidence, 0.10),
            }
        )
    return pd.DataFrame(rows)


def trial_state_occupancy(trials: Sequence[TrialAtlas], n_states: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trial in trials:
        total = int(trial.valid_time_mask.sum())
        for state in range(n_states):
            count = int(np.sum(trial.valid_time_mask & (trial.predicted_state == state)))
            rows.append(
                {
                    "trial_id": trial.trial_id,
                    "state": state,
                    "valid_points": count,
                    "occupancy_fraction": count / max(total, 1),
                }
            )
    return pd.DataFrame(rows)


def _boundary_metric_rows(
    trials: Sequence[TrialAtlas], n_states: int, thresholds: dict[str, float]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes: list[int | None] = [None, *range(n_states)]
    for state in scopes:
        state_label = -1 if state is None else state
        for metric_name, trait_index in [*( (name, index) for index, name in enumerate(TRAIT_NAMES) ), ("dynamic_quality", None), ("pose_quality", None)]:
            if trait_index is not None and trait_index in SYMMETRIC_TRAITS:
                edges = [(0, 1)]
            else:
                edges = [(0, 1), (1, 0)]
            for source, target in edges:
                boundary_values: list[np.ndarray] = []
                interior_values: list[np.ndarray] = []
                for trial in trials:
                    base = trial.valid_time_mask.copy()
                    if state is not None:
                        base &= trial.predicted_state == state
                    if trait_index is None:
                        values = getattr(trial, metric_name)[:, source, target]
                        valid = base & np.isfinite(values)
                    else:
                        values = trial.edge_values[:, source, target, trait_index]
                        valid = base & trial.edge_valid[:, source, target, trait_index]
                    boundary_values.append(values[valid & trial.boundary_adjacent])
                    interior_values.append(values[valid & ~trial.boundary_adjacent])
                boundary = np.concatenate(boundary_values)
                interior = np.concatenate(interior_values)
                b = _summary(boundary)
                i = _summary(interior)
                row = {
                    "state": state_label,
                    "metric": metric_name,
                    "source_index": source,
                    "target_index": target,
                    "edge_scope": "symmetric_once" if trait_index in SYMMETRIC_TRAITS else "directed",
                    "boundary_valid_count": b["valid_count"],
                    "interior_valid_count": i["valid_count"],
                    "boundary_median": b["median"],
                    "interior_median": i["median"],
                    "boundary_iqr": b["iqr"],
                    "interior_iqr": i["iqr"],
                    "boundary_minus_interior_median": float(b["median"] - i["median"]),
                }
                if metric_name in {"dynamic_quality", "pose_quality"}:
                    low = thresholds[
                        "dynamic_low_q20" if metric_name == "dynamic_quality" else "pose_low_q20"
                    ]
                    b_low = float(np.mean(boundary <= low)) if boundary.size else float("nan")
                    i_low = float(np.mean(interior <= low)) if interior.size else float("nan")
                    row.update(
                        boundary_low_quality_fraction=b_low,
                        interior_low_quality_fraction=i_low,
                        low_quality_enrichment_ratio=(
                            b_low / i_low if np.isfinite(i_low) and i_low > 0 else float("nan")
                        ),
                    )
                rows.append(row)
        for dynamic_index, trait_index in enumerate(DYNAMIC_TRAITS):
            for diagnostic_name, attribute in (("sign_flip_rate", "dynamic_sign_flip"), ("absolute_jump", "dynamic_abs_jump")):
                diagnostic_edges = (
                    ((0, 1), (1, 0))
                    if trait_index in DIRECTED_TRAITS
                    else ((0, 1),)
                )
                for source, target in diagnostic_edges:
                    boundary_values = []
                    interior_values = []
                    for trial in trials:
                        base = trial.valid_time_mask.copy()
                        if state is not None:
                            base &= trial.predicted_state == state
                        values = getattr(trial, attribute)[:, source, target, dynamic_index]
                        valid = base & np.isfinite(values)
                        boundary_values.append(values[valid & trial.boundary_adjacent])
                        interior_values.append(values[valid & ~trial.boundary_adjacent])
                    boundary = np.concatenate(boundary_values)
                    interior = np.concatenate(interior_values)
                    b = _summary(boundary)
                    i = _summary(interior)
                    rows.append(
                        {
                            "state": state_label,
                            "metric": f"{TRAIT_NAMES[trait_index]}_{diagnostic_name}",
                            "source_index": source,
                            "target_index": target,
                            "edge_scope": "diagnostic_only",
                            "boundary_valid_count": b["valid_count"],
                            "interior_valid_count": i["valid_count"],
                            "boundary_median": b["mean"] if diagnostic_name == "sign_flip_rate" else b["median"],
                            "interior_median": i["mean"] if diagnostic_name == "sign_flip_rate" else i["median"],
                            "boundary_iqr": b["iqr"],
                            "interior_iqr": i["iqr"],
                            "boundary_minus_interior_median": (
                                (b["mean"] - i["mean"]) if diagnostic_name == "sign_flip_rate" else (b["median"] - i["median"])
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def _aligned_long_table(trial: TrialAtlas) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source, target in ((0, 1), (1, 0)):
        data: dict[str, Any] = {
            "trial_id": trial.trial_id,
            "patch_index": np.arange(len(trial.patch_start_s)),
            "patch_start_s": trial.patch_start_s,
            "patch_end_s": trial.patch_end_s,
            "patch_center_s": trial.patch_center_s,
            "valid_time_mask": trial.valid_time_mask,
            "state": trial.predicted_state,
            "state_confidence": trial.state_confidence,
            "bout_id": trial.bout_id,
            "bout_start_s": trial.bout_start_s,
            "bout_end_s": trial.bout_end_s,
            "bout_duration_s": trial.bout_duration_s,
            "boundary_adjacent": trial.boundary_adjacent,
            "boundary_distance_s": trial.boundary_distance_s,
            "source_index": source,
            "target_index": target,
            "source_identity": trial.identity[source],
            "target_identity": trial.identity[target],
            "dynamic_quality": trial.dynamic_quality[:, source, target],
            "pose_quality": trial.pose_quality[:, source, target],
        }
        for trait_index, trait_name in enumerate(TRAIT_NAMES):
            valid = trial.edge_valid[:, source, target, trait_index]
            data[trait_name] = np.where(
                valid, trial.edge_values[:, source, target, trait_index], np.nan
            )
            data[f"{trait_name}__confidence"] = trial.edge_confidence[
                :, source, target, trait_index
            ]
            data[f"{trait_name}__coverage"] = trial.edge_coverage[
                :, source, target, trait_index
            ]
            data[f"{trait_name}__valid"] = valid
        for local, trait_index in enumerate(DYNAMIC_TRAITS):
            name = TRAIT_NAMES[trait_index]
            data[f"{name}__sign_flip"] = trial.dynamic_sign_flip[
                :, source, target, local
            ]
            data[f"{name}__absolute_jump"] = trial.dynamic_abs_jump[
                :, source, target, local
            ]
        frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True)


def _save_trial(trial: TrialAtlas, output: Path) -> tuple[Path, Path]:
    tensor_dir = output / "atlas_edge_tensors"
    table_dir = output / "aligned_trial_tables"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = tensor_dir / f"{trial.trial_id}_atlas_edge_tensor.npz"
    np.savez_compressed(
        tensor_path,
        contract=np.asarray("social_gnn_state_atlas_edge_v1"),
        trial_id=np.asarray(trial.trial_id),
        identity=trial.identity,
        patch_start_s=trial.patch_start_s,
        patch_end_s=trial.patch_end_s,
        patch_center_s=trial.patch_center_s,
        valid_time_mask=trial.valid_time_mask,
        predicted_state=trial.predicted_state,
        state_posterior=trial.posterior,
        state_confidence=trial.state_confidence,
        bout_id=trial.bout_id,
        bout_start_s=trial.bout_start_s,
        bout_end_s=trial.bout_end_s,
        bout_duration_s=trial.bout_duration_s,
        boundary_adjacent=trial.boundary_adjacent,
        boundary_distance_s=trial.boundary_distance_s,
        atlas_channel_names=np.asarray(ATLAS_CHANNEL_NAMES),
        atlas_edge_tensor=trial.atlas_edge_tensor,
        edge_values_raw=trial.edge_values,
        edge_confidence_raw=trial.edge_confidence,
        edge_coverage_raw=trial.edge_coverage,
        edge_valid=trial.edge_valid,
        dynamic_quality=trial.dynamic_quality,
        pose_quality=trial.pose_quality,
        dynamic_trait_names=np.asarray([TRAIT_NAMES[index] for index in DYNAMIC_TRAITS]),
        dynamic_sign_flip=trial.dynamic_sign_flip,
        dynamic_abs_jump=trial.dynamic_abs_jump,
    )
    table_path = table_dir / f"{trial.trial_id}_aligned_atlas.csv.gz"
    _aligned_long_table(trial).to_csv(table_path, index=False, compression="gzip")
    return tensor_path, table_path


def _load_video_metadata(
    *,
    summary_path: Path,
    metadata_path: Path,
    time_offset_path: Path,
    trial_ids: Sequence[str],
) -> pd.DataFrame:
    summary = pd.read_csv(summary_path)
    metadata = pd.read_csv(metadata_path)
    offset_payload = json.loads(time_offset_path.read_text(encoding="utf-8"))
    frame = summary[["trial_id", "source_video", "video_duration_s"]].merge(
        metadata[
            [
                "trial_id",
                "nominal_frame_rate",
                "stream_start_time_s",
                "frame_pts_count",
                "pts_source",
                "probe_status",
            ]
        ],
        on="trial_id",
        how="inner",
        validate="one_to_one",
    )
    frame = frame[frame.trial_id.astype(str).isin(trial_ids)].copy()
    if set(frame.trial_id.astype(str)) != set(trial_ids):
        raise ValueError("video metadata does not cover every atlas trial")
    offsets = offset_payload.get("trials", {})
    frame["video_time_offset_s"] = [
        float(offsets[str(trial)]["video_time_offset_s"])
        for trial in frame.trial_id
    ]
    frame["time_origin_provenance"] = [
        str(offsets[str(trial)]["time_origin_provenance"])
        for trial in frame.trial_id
    ]
    frame["source_available"] = [_path_available(str(path)) for path in frame.source_video]
    return frame.sort_values("trial_id").reset_index(drop=True)


def _attach_boundary_quality(
    boundaries: pd.DataFrame, trials: Sequence[TrialAtlas]
) -> pd.DataFrame:
    if boundaries.empty:
        return boundaries
    by_trial = {trial.trial_id: trial for trial in trials}
    dynamic: list[float] = []
    pose: list[float] = []
    for row in boundaries.itertuples():
        trial = by_trial[str(row.trial_id)]
        index = np.asarray([int(row.left_patch_index), int(row.right_patch_index)])
        dynamic.append(
            float(np.nanmedian(trial.dynamic_quality[index][:, [0, 1], [1, 0]]))
        )
        pose.append(float(np.nanmedian(trial.pose_quality[index][:, [0, 1], [1, 0]])))
    result = boundaries.copy()
    result["boundary_dynamic_quality"] = dynamic
    result["boundary_pose_quality"] = pose
    return result


def _write_figures(
    output: Path,
    trait_summary: pd.DataFrame,
    quality_summary: pd.DataFrame,
    bout_summary: pd.DataFrame,
    boundary_summary: pd.DataFrame,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    trait = trait_summary.copy()
    trait["column"] = trait.apply(
        lambda row: (
            str(row.trait)
            if row.edge_scope == "symmetric_once"
            else f"{row.trait}:{int(row.source_index)}->{int(row.target_index)}"
        ),
        axis=1,
    )
    matrix = trait.pivot(index="state", columns="column", values="median")
    robust = matrix.copy()
    for column in robust.columns:
        values = robust[column]
        median = values.median()
        iqr = values.quantile(0.75) - values.quantile(0.25)
        robust[column] = (values - median) / (iqr if iqr > 1e-12 else 1.0)
    fig, axis = plt.subplots(figsize=(15, 6))
    image = axis.imshow(robust.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    axis.set_yticks(range(len(robust.index)), [str(value) for value in robust.index])
    axis.set_xticks(range(len(robust.columns)), robust.columns, rotation=65, ha="right", fontsize=7)
    axis.set_ylabel("HMM state")
    axis.set_title("State median edge traits (columnwise robust scaling for display only)")
    fig.colorbar(image, ax=axis, label="(state median - across-state median) / IQR")
    fig.tight_layout()
    path = figure_dir / "state_by_trait_robust_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path.resolve()))

    quality = quality_summary.groupby(["state", "quality"], as_index=False).median(numeric_only=True)
    qmatrix = quality.pivot(index="state", columns="quality", values="median")
    low_matrix = quality.pivot(index="state", columns="quality", values="low_quality_fraction")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(bout_summary.state, bout_summary.occupancy_fraction)
    axes[0].set(xlabel="HMM state", ylabel="occupancy fraction", title="State occupancy")
    axes[1].plot(
        qmatrix.index,
        qmatrix["pose_quality"],
        marker="o",
        label="median pose_quality",
    )
    axes[1].plot(
        low_matrix.index,
        low_matrix["dynamic_quality"],
        marker="s",
        label="low dynamic-support fraction",
    )
    axes[1].set(
        xlabel="HMM state",
        ylabel="quality proxy / fraction",
        title="Pose availability and dynamic support",
    )
    axes[1].legend()
    fig.tight_layout()
    path = figure_dir / "state_occupancy_and_quality.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path.resolve()))

    global_quality = boundary_summary[
        boundary_summary.state.eq(-1)
        & boundary_summary.metric.isin(["dynamic_quality", "pose_quality"])
    ]
    fig, axis = plt.subplots(figsize=(8, 4))
    labels = [f"{row.metric} {int(row.source_index)}->{int(row.target_index)}" for row in global_quality.itertuples()]
    axis.bar(labels, global_quality.boundary_minus_interior_median)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("boundary median - interior median")
    axis.set_title("Global boundary-adjacent quality difference")
    axis.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    path = figure_dir / "boundary_quality_delta.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path.resolve()))
    return paths


def _state_flags(
    state: int,
    quality_summary: pd.DataFrame,
    boundary_summary: pd.DataFrame,
) -> list[str]:
    flags: list[str] = []
    quality = quality_summary[quality_summary.state.eq(state)]
    pose = quality[quality.quality.eq("pose_quality")]
    dynamic = quality[quality.quality.eq("dynamic_quality")]
    if not pose.empty and float(pose.low_quality_fraction.mean()) >= 0.30:
        flags.append("elevated_low_pose_quality_fraction")
    if not dynamic.empty and float(dynamic.low_quality_fraction.mean()) >= 0.30:
        flags.append("elevated_low_dynamic_support_fraction")
    boundary = boundary_summary[
        boundary_summary.state.eq(state)
        & boundary_summary.metric.isin(["pose_quality", "dynamic_quality"])
    ]
    enrichment = pd.to_numeric(boundary.get("low_quality_enrichment_ratio"), errors="coerce")
    if np.isfinite(enrichment).any() and float(np.nanmax(enrichment)) >= 1.25:
        flags.append("boundary_low_quality_enrichment_ge_1p25")
    return flags or ["no_predefined_qc_flag_triggered"]


def _write_state_cards(
    output: Path,
    *,
    bout_summary: pd.DataFrame,
    trait_summary: pd.DataFrame,
    quality_summary: pd.DataFrame,
    boundary_summary: pd.DataFrame,
    clip_catalog: pd.DataFrame,
    n_states: int,
) -> list[str]:
    directory = output / "state_cards"
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for state in range(n_states):
        bout = bout_summary[bout_summary.state.eq(state)].iloc[0].to_dict()
        traits = trait_summary[trait_summary.state.eq(state)].to_dict("records")
        quality = quality_summary[quality_summary.state.eq(state)].to_dict("records")
        clips = clip_catalog[clip_catalog.state.eq(state)].to_dict("records")
        flags = _state_flags(state, quality_summary, boundary_summary)
        payload = {
            "contract": "social_gnn_state_card_v1",
            "state_id": state,
            "biological_name": None,
            "bout_summary": bout,
            "trait_summary": traits,
            "quality_summary": quality,
            "qc_flags": flags,
            "clip_records": clips,
            "interpretation_warning": "Descriptive state ID only; no biological behavior label assigned.",
        }
        json_path = directory / f"state_{state:02d}.json"
        _write_json(json_path, payload)
        markdown = [
            f"# State {state:02d}",
            "",
            "Provisional Sticky-HMM state ID only. No biological name is assigned.",
            "",
            f"- Occupancy: {float(bout['occupancy_fraction']):.3%}",
            f"- Bouts: {int(bout['total_bouts'])}",
            f"- Median duration: {float(bout['median_bout_duration_s']):.3f} s",
            f"- Median posterior confidence: {float(bout['state_confidence_median']):.3f}",
            f"- QC flags: {', '.join(flags)}",
            "",
            f"Clip records: {len(clips)} (rendering status is recorded per row in the global clip catalog).",
        ]
        md_path = directory / f"state_{state:02d}.md"
        md_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
        paths.extend([str(json_path.resolve()), str(md_path.resolve())])
    return paths


def _write_report(
    output: Path,
    *,
    trials: Sequence[TrialAtlas],
    bout_summary: pd.DataFrame,
    trait_summary: pd.DataFrame,
    quality_summary: pd.DataFrame,
    boundary_summary: pd.DataFrame,
    clip_catalog: pd.DataFrame,
    video_metadata: pd.DataFrame,
    thresholds: dict[str, float],
    representative_assignment_exact: bool,
) -> Path:
    pose = (
        quality_summary[quality_summary.quality.eq("pose_quality")]
        .groupby("state", as_index=False)
        .agg(median=("median", "mean"), low_fraction=("low_quality_fraction", "mean"))
        .sort_values("median")
    )
    dynamic = (
        quality_summary[quality_summary.quality.eq("dynamic_quality")]
        .groupby("state", as_index=False)
        .agg(median=("median", "mean"), low_fraction=("low_quality_fraction", "mean"))
        .sort_values("low_fraction", ascending=False)
    )
    global_boundary = boundary_summary[
        boundary_summary.state.eq(-1)
        & boundary_summary.metric.isin(["dynamic_quality", "pose_quality"])
    ]
    low_pose_states = ", ".join(
        f"{int(row.state)} ({row.median:.3f})" for row in pose.head(3).itertuples()
    )
    high_low_dynamic_states = ", ".join(
        f"{int(row.state)} ({row.low_fraction:.1%} low-support)"
        for row in dynamic.head(3).itertuples()
    )
    boundary_lines = []
    for row in global_boundary.itertuples():
        enrichment = getattr(row, "low_quality_enrichment_ratio", float("nan"))
        boundary_lines.append(
            f"- `{row.metric}` {int(row.source_index)}->{int(row.target_index)}: "
            f"boundary median minus interior median = {row.boundary_minus_interior_median:.4f}; "
            f"low-quality enrichment = {enrichment:.3f}."
        )
    unavailable = video_metadata[~video_metadata.source_available.astype(bool)].trial_id.astype(str).tolist()
    clips_by_category = clip_catalog.groupby("category").size().to_dict()
    total_valid = sum(int(trial.valid_time_mask.sum()) for trial in trials)
    state_overview = bout_summary[[
        "state",
        "occupancy_fraction",
        "total_bouts",
        "median_bout_duration_s",
        "short_bout_fraction",
        "state_confidence_median",
    ]].merge(
        pose.rename(columns={"median": "pose_quality_median"})[["state", "pose_quality_median"]],
        on="state",
    ).merge(
        dynamic.rename(columns={"low_fraction": "dynamic_low_support_fraction"})[["state", "dynamic_low_support_fraction"]],
        on="state",
    ).sort_values("state")
    overview_lines = [
        "| State | Occupancy | Bouts | Median bout s | Short-bout fraction | Posterior median | Pose quality median | Dynamic low-support |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in state_overview.itertuples():
        overview_lines.append(
            f"| {int(row.state)} | {row.occupancy_fraction:.2%} | {int(row.total_bouts)} | "
            f"{row.median_bout_duration_s:.3f} | {row.short_bout_fraction:.1%} | "
            f"{row.state_confidence_median:.4f} | {row.pose_quality_median:.3f} | "
            f"{row.dynamic_low_support_fraction:.1%} |"
        )
    trait_lines = [
        "| Trait / edge scope | Lowest-median state | Highest-median state |",
        "| --- | ---: | ---: |",
    ]
    trait_groups = trait_summary.copy()
    trait_groups["scope"] = trait_groups.apply(
        lambda row: (
            str(row.trait)
            if row.edge_scope == "symmetric_once"
            else f"{row.trait} {int(row.source_index)}->{int(row.target_index)}"
        ),
        axis=1,
    )
    for scope, rows in trait_groups.groupby("scope", sort=False):
        ordered = rows.sort_values("median")
        low = ordered.iloc[0]
        high = ordered.iloc[-1]
        trait_lines.append(
            f"| `{scope}` | {int(low.state)} ({float(low['median']):.3f}) | "
            f"{int(high.state)} ({float(high['median']):.3f}) |"
        )
    sign_flip = boundary_summary[
        boundary_summary.state.eq(-1)
        & boundary_summary.metric.str.endswith("_sign_flip_rate")
    ].sort_values("boundary_minus_interior_median", ascending=False)
    sign_flip_lines = [
        f"- Largest global boundary sign-flip-rate delta: `{row.metric}` "
        f"{int(row.source_index)}->{int(row.target_index)} = "
        f"{row.boundary_minus_interior_median:+.4f}."
        for row in sign_flip.head(3).itertuples()
    ]
    report = f"""# State Atlas v1 for the 0.04 s Social-GNN Sticky-HMM Pipeline

## What this is

State Atlas v1 is a read-only quantitative, QC, and video-review backend for
the existing state system. It is **not** a new model, an interpreter, or an
automatic behavior-labeling system. State IDs remain provisional numerical
labels.

## Frozen contract

`Graph Full (64D g_t) -> train-fitted scaler -> train-fitted PCA16 -> Sticky HMM K=12`

- Clock: 0.04 s pointwise physical intervals (25 Hz state grid).
- Included trials: A1_C1, A3_C3, A4_B4, A4_C4.
- Excluded/locked: A5_C5.
- No encoder, scaler, PCA, or HMM parameter was fitted or changed here.
- Decoding the saved A4_C4 representative assignment was exact: `{representative_assignment_exact}`.

## Directed 10D edge atlas

Every off-diagonal directed edge has `[8 raw traits, dynamic_quality,
pose_quality]`, giving `[T,N,N,10]`. A->B and B->A are never collapsed in the
stored tensor. Symmetric traits are stored on both directed edges for contract
consistency, but state summaries count only one direction to avoid duplication.
The raw confidence and coverage arrays remain separate; values are never
multiplied by confidence.

Missing raw traits are represented as `NaN` in the analysis tensor and remain
explicit in `edge_valid`. The original finite raw arrays are retained alongside
their confidence and coverage metadata.

## Quality definitions

For every raw trait, support is `sqrt(confidence * coverage)` in [0,1].
`dynamic_quality` is the arithmetic mean of support for radial approach speed,
pair closing speed, and movement-direction alignment. `pose_quality` is the
arithmetic mean of support for nose distance, source-nose-to-target-centroid,
source-faces-target, and head-direction alignment. Missing support contributes
zero, so missingness lowers the summary. Movement-alignment's existing
motion-noise-floor validity enters through its raw confidence/coverage.

Neither quality proxy uses trait magnitude. Real fast motion is therefore not
classified as bad merely because it is large. Dynamic sign flips and absolute
jumps are exported separately and are not folded into `dynamic_quality`.
These are transparent QC proxies, not calibrated probabilities.

Global data-derived thresholds (for descriptive fractions only):

- dynamic low/high: {thresholds['dynamic_low_q20']:.4f} / {thresholds['dynamic_high_q80']:.4f}
- pose low/high: {thresholds['pose_low_q20']:.4f} / {thresholds['pose_high_q80']:.4f}

## Confirmed descriptive findings

- The atlas preserves {total_valid:,} valid 0.04 s points across four trials.
- Lowest median pose-quality states: {low_pose_states}.
- Highest low-dynamic-support fractions: {high_low_dynamic_states}.
- Dynamic-quality medians are 1.0 for every state; this proxy is saturated at the median and should be read through low-support fractions and the separately exported sign-flip/jump diagnostics, not as a fine ranking.
- Occupancy ranges from {bout_summary.occupancy_fraction.min():.3%} to {bout_summary.occupancy_fraction.max():.3%}; all 12 selected-state IDs are retained in the tables.
- State-level distributions include median, IQR, P10/P90, trimmed mean, usable support, posterior confidence, and bout-duration summaries; means alone are not used as the profile.

### State-by-state overview

{chr(10).join(overview_lines)}

### Trait pattern extrema

This compact table reports descriptive state medians. Full distributions and
support are in `summary_tables/state_trait_summary.csv`.

{chr(10).join(trait_lines)}

## Boundary QC

Boundary-adjacent means within the configured physical-time window around a
Viterbi state change; invalid gaps and trial boundaries are not bridged.

{chr(10).join(boundary_lines)}

{chr(10).join(sign_flip_lines)}

These differences are descriptive. They may identify review priorities but do
not prove that a transition is an artifact or that a state is biologically
invalid.

## Video atlas

The deterministic catalog contains {len(clip_catalog)} planned clips:
{json.dumps(clips_by_category, ensure_ascii=False)}. Categories are clean
high-confidence examples, deterministic random representatives, state-relative
short bouts, and transition-adjacent intervals. Each interval includes physical
context and records the exact rule `video_pts = social_time + offset` plus a
reference frame span derived from the previously packet-PTS-validated CFR grid.
It does not assume one state step equals one video frame.

Source availability at generation time: unavailable trials =
`{', '.join(unavailable) if unavailable else 'none'}`. Individual clips were not
fabricated. `video_sampling_tables/extract_clips.ps1` and `.sh` can render the
catalog when the source videos are mounted again. No montage was emitted, so
the gray-separator requirement is not applicable to the current individual-clip
artifact set.

## QC suspicions for manual review

- States with elevated low-pose fractions should be reviewed for MouseGPT
  availability confounding before biological naming.
- States whose boundary rows show enriched low quality or dynamic sign flips
  deserve transition-focused video review.
- High dynamic volatility remains potentially biological; the atlas deliberately
  keeps magnitude, support, sign flips, and jumps as separate evidence.

## Open questions

1. Do clean and random clips from the same state show a coherent visual motif?
2. Are short bouts meaningful brief events or transition uncertainty?
3. Do states with low pose availability remain coherent when reviewed using
   centroid/body movement alone?
4. Are boundary-adjacent changes visually meaningful across multiple trials?

## Limitations

- This is descriptive downstream analysis of a development state system.
- The edge traits already contributed to the upstream Graph Full representation,
  so state-trait associations are not independent validation.
- A4_C4 is reused development data, not an untouched biological test.
- Quality proxies are not causal proof of measurement artifact.
- No biological behavior names are assigned.
"""
    path = output / "state_atlas_v1_report.md"
    path.write_text(report, encoding="utf-8")
    return path


def build_state_atlas(
    config_path: str | Path,
    *,
    output_override: str | Path | None = None,
    render_override: bool | None = None,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    config = _load_yaml(config_file)
    root = _resolve(config_file.parent, config.get("project_root", ".."))
    output = (
        Path(output_override).resolve()
        if output_override is not None
        else _resolve(root, config["output_dir"])
    )
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_file, output / "atlas_config.yaml")

    trials_requested = tuple(str(value) for value in config.get("trial_ids", DEFAULT_TRIALS))
    if trials_requested != DEFAULT_TRIALS:
        raise ValueError(f"State Atlas v1 requires exact trials {DEFAULT_TRIALS}, got {trials_requested}")
    if LOCKED_TRIAL in trials_requested:
        raise ValueError(f"locked trial {LOCKED_TRIAL} must not enter State Atlas v1")
    frozen = config["frozen_pipeline"]
    checkpoint = _resolve(root, frozen["hmm_checkpoint"])
    preprocessor_path = _resolve(root, frozen["preprocessor"])
    embedding_dir = _resolve(root, frozen["embedding_dir"])
    edge_dir = _resolve(root, frozen["edge_dir"])
    saved_assignment = _resolve(root, frozen["representative_assignment"])

    model, model_metadata = GaussianHMM.load(checkpoint)
    if model.n_states != 12 or not np.isclose(model.timestep_s, 0.04):
        raise ValueError("frozen HMM must be K=12 on the 0.04 s clock")
    preprocessor = EmbeddingPreprocessor.load(preprocessor_path)
    if preprocessor.output_dim != 16:
        raise ValueError("frozen preprocessor must output PCA16")
    expected_preprocessor_hash = str(model_metadata.get("preprocessor_sha256", "")).upper()
    actual_preprocessor_hash = _sha256(preprocessor_path)
    if expected_preprocessor_hash and expected_preprocessor_hash != actual_preprocessor_hash:
        raise ValueError("preprocessor hash differs from frozen HMM metadata")
    if int(model_metadata.get("K", 12)) != 12 or str(model_metadata.get("architecture", "graph_full")) != "graph_full":
        raise ValueError("checkpoint metadata does not describe selected Graph Full K=12")

    decoded: list[DecodedTrial] = []
    frozen_trials: list[FrozenEmbeddingTrial] = []
    for trial_id in trials_requested:
        trial = FrozenEmbeddingTrial.load(embedding_dir / f"{trial_id}_g_t.npz")
        if trial.trial_id != trial_id:
            raise ValueError(f"{trial_id}: embedding trial ID mismatch")
        durations = trial.patch_end_s - trial.patch_start_s
        if not np.allclose(durations, 0.04, rtol=1e-7, atol=1e-9):
            raise ValueError(f"{trial_id}: not a 0.04 s pointwise clock")
        frozen_trials.append(trial)
        decoded.append(decode_trial(model, preprocessor, trial))

    with np.load(saved_assignment, allow_pickle=False) as data:
        saved_trial_id = str(data["trial_id"].item())
        saved_state = np.asarray(data["predicted_state"], dtype=np.int64)
        saved_posterior = np.asarray(data["state_posterior"], dtype=np.float32)
        saved_start = np.asarray(data["patch_start_s"], dtype=np.float64)
        saved_end = np.asarray(data["patch_end_s"], dtype=np.float64)
    reference = next(value for value in decoded if value.trial.trial_id == saved_trial_id)
    representative_assignment_exact = bool(
        np.array_equal(saved_start, reference.trial.patch_start_s)
        and np.array_equal(saved_end, reference.trial.patch_end_s)
        and np.array_equal(saved_state, reference.predicted_state)
        and np.allclose(saved_posterior, reference.posterior, rtol=2e-5, atol=2e-6)
    )
    if not representative_assignment_exact:
        raise ValueError("frozen decode does not reproduce representative assignment")

    analysis = config.get("analysis", {})
    boundary_window_s = float(analysis.get("boundary_window_s", 0.20))
    short_bout_s = float(analysis.get("short_bout_s", 0.40))
    atlases: list[TrialAtlas] = []
    bout_frames: list[pd.DataFrame] = []
    boundary_frames: list[pd.DataFrame] = []
    tensor_files: list[str] = []
    table_files: list[str] = []
    for value in decoded:
        trial_id = value.trial.trial_id
        atlas, bouts, boundaries = build_trial_atlas(
            decoded=value,
            edge_npz=edge_dir / f"{trial_id}_social_edges.npz",
            boundary_window_s=boundary_window_s,
        )
        atlases.append(atlas)
        bout_frames.append(bouts)
        boundary_frames.append(boundaries)
        tensor_path, table_path = _save_trial(atlas, output)
        tensor_files.append(str(tensor_path.resolve()))
        table_files.append(str(table_path.resolve()))

    bouts = _enrich_bouts(atlases, bout_frames)
    boundaries = _attach_boundary_quality(
        pd.concat(boundary_frames, ignore_index=True), atlases
    )
    thresholds = _quality_thresholds(atlases)
    trait_summary = state_trait_summary(atlases, model.n_states)
    quality_summary = state_quality_summary(atlases, model.n_states, thresholds)
    bout_summary = state_bout_summary(
        atlases, bouts, model.n_states, short_bout_s=short_bout_s
    )
    occupancy = trial_state_occupancy(atlases, model.n_states)
    boundary_summary = _boundary_metric_rows(atlases, model.n_states, thresholds)

    summary_dir = output / "summary_tables"
    qc_dir = output / "qc_reports"
    profile_dir = output / "state_profiles"
    for directory in (summary_dir, qc_dir, profile_dir):
        directory.mkdir(parents=True, exist_ok=True)
    bouts.to_csv(profile_dir / "all_bouts.csv", index=False)
    boundaries.to_csv(qc_dir / "state_boundaries.csv", index=False)
    trait_summary.to_csv(summary_dir / "state_trait_summary.csv", index=False)
    quality_summary.to_csv(summary_dir / "state_quality_summary.csv", index=False)
    bout_summary.to_csv(summary_dir / "state_bout_summary.csv", index=False)
    occupancy.to_csv(summary_dir / "trial_state_occupancy.csv", index=False)
    boundary_summary.to_csv(summary_dir / "state_boundary_qc.csv", index=False)

    trait_wide = trait_summary.copy()
    trait_wide["feature"] = trait_wide.apply(
        lambda row: str(row.trait) if row.edge_scope == "symmetric_once" else f"{row.trait}_{int(row.source_index)}to{int(row.target_index)}",
        axis=1,
    )
    trait_wide.pivot(index="state", columns="feature", values="median").to_csv(
        summary_dir / "state_by_trait_median.csv"
    )
    quality_wide = quality_summary.groupby(["state", "quality"], as_index=False).median(numeric_only=True)
    quality_wide.pivot(index="state", columns="quality", values="median").to_csv(
        summary_dir / "state_by_quality_median.csv"
    )
    occupancy.pivot(index="trial_id", columns="state", values="occupancy_fraction").to_csv(
        summary_dir / "trial_by_state_occupancy_matrix.csv"
    )
    profile = bout_summary.merge(
        quality_wide.pivot(index="state", columns="quality", values="median").reset_index(),
        on="state",
        how="left",
    )
    profile.to_csv(summary_dir / "state_profile_summary.csv", index=False)

    video = config["video_mapping"]
    video_metadata = _load_video_metadata(
        summary_path=_resolve(root, video["trial_alignment_summary"]),
        metadata_path=_resolve(root, video["video_metadata"]),
        time_offset_path=_resolve(root, video["time_offsets"]),
        trial_ids=trials_requested,
    )
    video_dir = output / "video_sampling_tables"
    clip_root = output / "extracted_clips"
    video_dir.mkdir(parents=True, exist_ok=True)
    clip_root.mkdir(parents=True, exist_ok=True)
    clip_config = config.get("clips", {})
    clip_catalog = plan_state_clips(
        bouts=bouts,
        boundaries=boundaries,
        video_metadata=video_metadata,
        n_states=model.n_states,
        clips_per_category=int(clip_config.get("clips_per_category", 3)),
        context_s=float(clip_config.get("context_s", 1.5)),
        max_focal_s=float(clip_config.get("max_focal_s", 2.0)),
        boundary_focal_s=float(clip_config.get("boundary_focal_s", 0.4)),
        max_overlap_fraction=float(clip_config.get("max_overlap_fraction", 0.25)),
        random_seed=int(clip_config.get("random_seed", 20261101)),
    )
    clip_catalog = attach_clip_paths(clip_catalog, clip_root)
    render = bool(clip_config.get("render", False)) if render_override is None else render_override
    if render:
        clip_catalog = render_clip_catalog(
            clip_catalog,
            ffmpeg=_resolve(root, clip_config.get("ffmpeg", r"D:\ffmpeg-7.1-essentials_build\bin\ffmpeg.exe")),
            overwrite=bool(clip_config.get("overwrite", False)),
        )
    clip_catalog_path = video_dir / "clip_catalog.csv"
    clip_catalog.to_csv(clip_catalog_path, index=False)
    selection_metadata_path = video_dir / "clip_selection_metadata.csv"
    pd.DataFrame(
        [
            {
                "contract": "social_gnn_state_atlas_clip_selection_v1",
                "clips_per_category": int(clip_config.get("clips_per_category", 3)),
                "context_s": float(clip_config.get("context_s", 1.5)),
                "max_focal_s": float(clip_config.get("max_focal_s", 2.0)),
                "boundary_focal_s": float(clip_config.get("boundary_focal_s", 0.4)),
                "max_overlap_fraction": float(clip_config.get("max_overlap_fraction", 0.25)),
                "random_seed": int(clip_config.get("random_seed", 20261101)),
                "high_posterior_threshold": float(clip_catalog.selection_threshold_posterior.iloc[0]),
                "high_pose_quality_threshold": float(clip_catalog.selection_threshold_pose_quality.iloc[0]),
                "acceptable_dynamic_quality_threshold": float(clip_catalog.selection_threshold_dynamic_quality.iloc[0]),
                "physical_time_rule": "video_pts=social_time+video_time_offset_s",
                "sampling_note": "round-robin across trials when possible; overlap constrained within state/category",
            }
        ]
    ).to_csv(selection_metadata_path, index=False)
    for (state, category), rows in clip_catalog.groupby(["state", "category"]):
        directory = video_dir / f"state_{int(state):02d}"
        directory.mkdir(parents=True, exist_ok=True)
        rows.to_csv(directory / f"{category}.csv", index=False)
    write_extraction_scripts(
        clip_catalog_path,
        powershell_path=video_dir / "extract_clips.ps1",
        bash_path=video_dir / "extract_clips.sh",
    )
    video_metadata.to_csv(video_dir / "video_sources_and_time_mapping.csv", index=False)

    state_cards = _write_state_cards(
        output,
        bout_summary=bout_summary,
        trait_summary=trait_summary,
        quality_summary=quality_summary,
        boundary_summary=boundary_summary,
        clip_catalog=clip_catalog,
        n_states=model.n_states,
    )
    figures = _write_figures(
        output, trait_summary, quality_summary, bout_summary, boundary_summary
    )
    report_path = _write_report(
        output,
        trials=atlases,
        bout_summary=bout_summary,
        trait_summary=trait_summary,
        quality_summary=quality_summary,
        boundary_summary=boundary_summary,
        clip_catalog=clip_catalog,
        video_metadata=video_metadata,
        thresholds=thresholds,
        representative_assignment_exact=representative_assignment_exact,
    )

    artifact_files = [
        *tensor_files,
        *table_files,
        str((profile_dir / "all_bouts.csv").resolve()),
        str((qc_dir / "state_boundaries.csv").resolve()),
        str((summary_dir / "state_trait_summary.csv").resolve()),
        str((summary_dir / "state_quality_summary.csv").resolve()),
        str((summary_dir / "state_bout_summary.csv").resolve()),
        str((summary_dir / "trial_state_occupancy.csv").resolve()),
        str((summary_dir / "state_boundary_qc.csv").resolve()),
        str(clip_catalog_path.resolve()),
        str(selection_metadata_path.resolve()),
        *state_cards,
        *figures,
        str(report_path.resolve()),
    ]
    manifest = {
        "contract": "social_gnn_state_atlas_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Read-only quantitative/QC/video-review atlas; no fitting and no biological labels.",
        "frozen_pipeline": {
            "encoder": "graph_full",
            "embedding_dim": 64,
            "preprocessor_output_dim": preprocessor.output_dim,
            "hmm_states": model.n_states,
            "timestep_s": model.timestep_s,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "preprocessor": str(preprocessor_path),
            "preprocessor_sha256": actual_preprocessor_hash,
            "model_metadata": model_metadata,
            "representative_assignment_exact": representative_assignment_exact,
        },
        "trials": list(trials_requested),
        "excluded_trial": LOCKED_TRIAL,
        "atlas_tensor_contract": {
            "shape": "[T,N,N,10]",
            "channels": list(ATLAS_CHANNEL_NAMES),
            "directionality": "off-diagonal directed edges retained separately",
            "missing_value_rule": "NaN in analysis tensor plus explicit edge_valid; raw arrays retained",
        },
        "quality_definition": {
            "per_trait_support": "sqrt(edge_confidence * edge_coverage)",
            "dynamic_quality_traits": [TRAIT_NAMES[index] for index in DYNAMIC_TRAITS],
            "pose_quality_traits": [TRAIT_NAMES[index] for index in POSE_TRAITS],
            "aggregation": "arithmetic mean over declared traits; missing support contributes zero",
            "magnitude_used": False,
            "diagnostics_not_folded_into_quality": ["dynamic_sign_flip", "dynamic_abs_jump"],
            "global_descriptive_thresholds": thresholds,
        },
        "video": {
            "render_requested": render,
            "rendered_clips": int(clip_catalog.render_status.eq("rendered").sum()),
            "source_unavailable_clips": int(clip_catalog.render_status.eq("source_unavailable").sum()),
            "mapping_rule": "video_pts = social_time + video_time_offset_s",
            "state_step_is_not_video_frame": True,
        },
        "artifact_files": artifact_files,
        "limitations": [
            "descriptive development atlas, not an interpreter",
            "A4_C4 is reused development data",
            "edge traits contributed to the upstream representation",
            "quality proxies are not calibrated probabilities or causal proof",
            "state IDs have no automatic biological names",
        ],
    }
    _write_json(output / "atlas_manifest.json", manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--render-clips", action="store_true", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_state_atlas(
        args.config,
        output_override=args.output_dir,
        render_override=args.render_clips,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
