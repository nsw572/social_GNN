"""Shared read-only loaders and definitions for the three State Atlas views."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .state_atlas import (
    ATLAS_CHANNEL_NAMES,
    DEFAULT_TRIALS,
    DYNAMIC_TRAITS,
    TRAIT_NAMES,
)


@dataclass(frozen=True)
class AtlasViewTrial:
    trial_id: str
    identity: np.ndarray
    patch_start_s: np.ndarray
    patch_end_s: np.ndarray
    valid_time_mask: np.ndarray
    predicted_state: np.ndarray
    state_confidence: np.ndarray
    bout_id: np.ndarray
    boundary_adjacent: np.ndarray
    atlas_edge_tensor: np.ndarray
    edge_values: np.ndarray
    edge_confidence: np.ndarray
    edge_coverage: np.ndarray
    edge_valid: np.ndarray
    dynamic_quality: np.ndarray
    pose_quality: np.ndarray
    dynamic_sign_flip: np.ndarray
    dynamic_abs_jump: np.ndarray

    def __post_init__(self) -> None:
        length = len(self.patch_start_s)
        if self.patch_end_s.shape != (length,):
            raise ValueError(f"{self.trial_id}: invalid patch clock")
        if self.atlas_edge_tensor.shape != (length, 2, 2, 10):
            raise ValueError(
                f"{self.trial_id}: expected atlas [T,2,2,10], got "
                f"{self.atlas_edge_tensor.shape}"
            )
        if self.edge_values.shape != (length, 2, 2, 8):
            raise ValueError(f"{self.trial_id}: expected edge values [T,2,2,8]")
        for value in (self.edge_confidence, self.edge_coverage):
            if value.shape != self.edge_values.shape:
                raise ValueError(f"{self.trial_id}: edge metadata shape mismatch")
            if np.any((value < 0) | (value > 1)):
                raise ValueError(f"{self.trial_id}: edge metadata outside [0,1]")
        if tuple(ATLAS_CHANNEL_NAMES) != (
            "centroid_distance",
            "nose_to_nose_distance",
            "source_nose_to_target_centroid_distance",
            "source_faces_target",
            "head_direction_alignment",
            "source_radial_approach_speed",
            "pair_closing_speed",
            "movement_direction_alignment",
            "dynamic_quality",
            "pose_quality",
        ):
            raise RuntimeError("State Atlas channel contract changed")

    @property
    def patch_center_s(self) -> np.ndarray:
        return 0.5 * (self.patch_start_s + self.patch_end_s)


def load_atlas_trial(path: str | Path) -> AtlasViewTrial:
    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        trial_id = str(data["trial_id"].item())
        channel_names = tuple(str(value) for value in data["atlas_channel_names"])
        if channel_names != tuple(ATLAS_CHANNEL_NAMES):
            raise ValueError(f"{trial_id}: atlas channel order differs from v1 contract")
        return AtlasViewTrial(
            trial_id=trial_id,
            identity=np.asarray(data["identity"]),
            patch_start_s=np.asarray(data["patch_start_s"], dtype=np.float64),
            patch_end_s=np.asarray(data["patch_end_s"], dtype=np.float64),
            valid_time_mask=np.asarray(data["valid_time_mask"], dtype=bool),
            predicted_state=np.asarray(data["predicted_state"], dtype=np.int64),
            state_confidence=np.asarray(data["state_confidence"], dtype=np.float32),
            bout_id=np.asarray(data["bout_id"], dtype=np.int64),
            boundary_adjacent=np.asarray(data["boundary_adjacent"], dtype=bool),
            atlas_edge_tensor=np.asarray(data["atlas_edge_tensor"], dtype=np.float32),
            edge_values=np.asarray(data["edge_values_raw"], dtype=np.float32),
            edge_confidence=np.asarray(data["edge_confidence_raw"], dtype=np.float32),
            edge_coverage=np.asarray(data["edge_coverage_raw"], dtype=np.float32),
            edge_valid=np.asarray(data["edge_valid"], dtype=bool),
            dynamic_quality=np.asarray(data["dynamic_quality"], dtype=np.float32),
            pose_quality=np.asarray(data["pose_quality"], dtype=np.float32),
            dynamic_sign_flip=np.asarray(data["dynamic_sign_flip"], dtype=np.float32),
            dynamic_abs_jump=np.asarray(data["dynamic_abs_jump"], dtype=np.float32),
        )


def iter_atlas_trials(
    atlas_root: str | Path,
    trial_ids: tuple[str, ...] = DEFAULT_TRIALS,
) -> Iterator[AtlasViewTrial]:
    root = Path(atlas_root)
    tensor_root = root / "atlas_edge_tensors"
    for trial_id in trial_ids:
        yield load_atlas_trial(tensor_root / f"{trial_id}_atlas_edge_tensor.npz")


def mean_directions(values: np.ndarray, valid_time_mask: np.ndarray) -> np.ndarray:
    """Mean 0->1 and 1->0 without converting invalid timesteps to zero."""

    pair = np.stack([values[:, 0, 1], values[:, 1, 0]], axis=1).astype(np.float64)
    result = np.nanmean(pair, axis=1)
    result[~np.asarray(valid_time_mask, dtype=bool)] = np.nan
    return result


def transition_times(trial: AtlasViewTrial) -> np.ndarray:
    center = trial.patch_center_s
    if len(center) < 2:
        return np.asarray([], dtype=np.float64)
    nominal = float(np.median(np.diff(center)))
    adjacent = (
        trial.valid_time_mask[:-1]
        & trial.valid_time_mask[1:]
        & (np.diff(center) <= 1.5 * nominal)
        & (trial.predicted_state[:-1] != trial.predicted_state[1:])
    )
    index = np.flatnonzero(adjacent)
    return 0.5 * (center[index] + center[index + 1])


FINGERPRINT_CHANNELS = (
    ("centroid_distance", 0, 0, 1, "symmetric"),
    ("nose_to_nose_distance", 1, 0, 1, "symmetric"),
    ("head_direction_alignment", 4, 0, 1, "symmetric"),
    ("pair_closing_speed", 6, 0, 1, "symmetric"),
    ("movement_direction_alignment", 7, 0, 1, "symmetric"),
    ("source_nose_to_target_centroid_distance 0->1", 2, 0, 1, "directed"),
    ("source_nose_to_target_centroid_distance 1->0", 2, 1, 0, "directed"),
    ("source_faces_target 0->1", 3, 0, 1, "directed"),
    ("source_faces_target 1->0", 3, 1, 0, "directed"),
    ("source_radial_approach_speed 0->1", 5, 0, 1, "directed"),
    ("source_radial_approach_speed 1->0", 5, 1, 0, "directed"),
    ("dynamic_quality", 8, -1, -1, "quality"),
    ("pose_quality", 9, -1, -1, "quality"),
)


POSE_VALIDITY_CHANNELS = (
    ("nose_to_nose_distance", 1, 0, 1, "symmetric"),
    ("head_direction_alignment", 4, 0, 1, "symmetric"),
    ("source_nose_to_target_centroid_distance 0->1", 2, 0, 1, "directed"),
    ("source_nose_to_target_centroid_distance 1->0", 2, 1, 0, "directed"),
    ("source_faces_target 0->1", 3, 0, 1, "directed"),
    ("source_faces_target 1->0", 3, 1, 0, "directed"),
)


def raw_channel(
    trial: AtlasViewTrial,
    trait_index: int,
    source: int,
    target: int,
) -> np.ndarray:
    values = trial.edge_values[:, source, target, trait_index].astype(np.float64)
    valid = trial.edge_valid[:, source, target, trait_index] & trial.valid_time_mask
    return np.where(valid, values, np.nan)


def fingerprint_channel_values(
    trial: AtlasViewTrial,
    channel: tuple[str, int, int, int, str],
) -> np.ndarray:
    _label, trait_index, source, target, kind = channel
    if kind == "quality":
        values = trial.dynamic_quality if trait_index == 8 else trial.pose_quality
        return mean_directions(values, trial.valid_time_mask)
    return raw_channel(trial, trait_index, source, target)


def robust_stats(values: np.ndarray) -> dict[str, float | int]:
    finite = np.sort(np.asarray(values, dtype=np.float64))
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "median": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "iqr": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
            "trimmed_mean_10pct": float("nan"),
        }
    q10, q25, median, q75, q90 = np.quantile(finite, [0.1, 0.25, 0.5, 0.75, 0.9])
    trim = int(np.floor(0.1 * len(finite)))
    trimmed = finite[trim : len(finite) - trim] if 2 * trim < len(finite) else finite
    return {
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "p10": float(q10),
        "p90": float(q90),
        "trimmed_mean_10pct": float(np.mean(trimmed)),
    }


__all__ = [
    "AtlasViewTrial",
    "FINGERPRINT_CHANNELS",
    "POSE_VALIDITY_CHANNELS",
    "fingerprint_channel_values",
    "iter_atlas_trials",
    "load_atlas_trial",
    "mean_directions",
    "raw_channel",
    "robust_stats",
    "transition_times",
]
