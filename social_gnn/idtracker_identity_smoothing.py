"""Optional retrospective repair for two-animal idtracker.ai trajectories.

The repair is deliberately separate from idtracker.ai.  It detects a close-contact
trajectory crossing followed by a later speed discontinuity, swaps the two identity
channels between those events, and interpolates only the short entry/exit windows.
It also repairs isolated paired coordinate spikes.  Raw coordinates are retained in
the output contract so every change remains auditable.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TrajectorySmoothingResult:
    raw_trajectories: np.ndarray
    smoothed_trajectories: np.ndarray
    raw_speed_px_s: np.ndarray
    smoothed_speed_px_s: np.ndarray
    artifact_seed_mask: np.ndarray
    artifact_mask: np.ndarray
    segment_id: np.ndarray
    identity_swap_mask: np.ndarray
    paired_interval_mask: np.ndarray
    swap_pair_id: np.ndarray
    centroid_confidence: np.ndarray
    robust_mean_speed_px_s: np.ndarray
    parameters: dict[str, Any]
    events: list[dict[str, Any]]
    swap_pairs: list[dict[str, Any]]


def compute_velocity_px_s(
    trajectories: np.ndarray, fps: float, method: str = "backward"
) -> tuple[np.ndarray, np.ndarray]:
    """Compute frame-aligned velocity without filling missing coordinates."""
    trajectories = np.asarray(trajectories, dtype=float)
    if trajectories.ndim != 3 or trajectories.shape[-1] != 2:
        raise ValueError("trajectories must have shape [frames, animals, 2]")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    if method not in {"backward", "central"}:
        raise ValueError("velocity method must be 'backward' or 'central'")

    velocity = np.full_like(trajectories, np.nan, dtype=float)
    valid = np.zeros(trajectories.shape[:2], dtype=bool)
    if trajectories.shape[0] < 2:
        return velocity, valid

    if method == "backward":
        pair_valid = np.isfinite(trajectories[1:]).all(axis=-1) & np.isfinite(
            trajectories[:-1]
        ).all(axis=-1)
        delta = (trajectories[1:] - trajectories[:-1]) * fps
        velocity[1:][pair_valid] = delta[pair_valid]
        valid[1:] = pair_valid
        return velocity, valid

    first_valid = np.isfinite(trajectories[0]).all(axis=-1) & np.isfinite(
        trajectories[1]
    ).all(axis=-1)
    last_valid = np.isfinite(trajectories[-2]).all(axis=-1) & np.isfinite(
        trajectories[-1]
    ).all(axis=-1)
    velocity[0][first_valid] = (
        trajectories[1][first_valid] - trajectories[0][first_valid]
    ) * fps
    velocity[-1][last_valid] = (
        trajectories[-1][last_valid] - trajectories[-2][last_valid]
    ) * fps
    valid[0] = first_valid
    valid[-1] = last_valid
    if trajectories.shape[0] > 2:
        central_valid = np.isfinite(trajectories[2:]).all(axis=-1) & np.isfinite(
            trajectories[:-2]
        ).all(axis=-1)
        delta = (trajectories[2:] - trajectories[:-2]) * (fps / 2.0)
        velocity[1:-1][central_valid] = delta[central_valid]
        valid[1:-1] = central_valid
    return velocity, valid


def compute_frame_speed_px_s(trajectories: np.ndarray, fps: float) -> np.ndarray:
    velocity, valid = compute_velocity_px_s(trajectories, fps, "backward")
    speed = np.linalg.norm(velocity, axis=-1)
    speed[~valid] = np.nan
    return speed


def _boolean_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(end - 1)) for start, end in changes.reshape((-1, 2))]


def _merge_short_gaps(
    runs: list[tuple[int, int]], max_gap_frames: int
) -> list[tuple[int, int]]:
    if not runs:
        return []
    merged = [runs[0]]
    for start, end in runs[1:]:
        previous_start, previous_end = merged[-1]
        if start - previous_end - 1 <= max_gap_frames:
            merged[-1] = (previous_start, end)
        else:
            merged.append((start, end))
    return merged


def detect_and_smooth_contact_speed_artifacts(
    trajectories: np.ndarray,
    *,
    fps: float,
    body_length_px: float,
    speed_multiplier: float = 5.0,
    robust_mean_upper_quantile: float = 0.99,
    minimum_speed_similarity_ratio: float = 0.80,
    maximum_displacement_cosine: float = -0.80,
    contact_distance_body_lengths: float = 0.35,
    contact_window_frames: int = 2,
    max_seed_gap_frames: int = 2,
    identity_contact_distance_body_lengths: float = 0.20,
    maximum_cross_identity_endpoint_ratio: float = 0.25,
    maximum_swap_pair_gap_seconds: float = 5.0,
    transition_speed_cap_multiplier: float = 5.0,
    maximum_transition_expansion_seconds: float = 1.0,
    interpolation_confidence: float = 0.05,
) -> TrajectorySmoothingResult:
    """Repair local spikes and contact-to-recovery identity exchanges."""
    raw = np.asarray(trajectories, dtype=float)
    if raw.ndim != 3 or raw.shape[1:] != (2, 2):
        raise ValueError("Identity smoothing currently requires [T, 2, 2] tracks")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not np.isfinite(body_length_px) or body_length_px <= 0:
        raise ValueError("body_length_px must be a positive finite value")
    if not 0 < robust_mean_upper_quantile <= 1:
        raise ValueError("robust_mean_upper_quantile must be in (0, 1]")
    if not 0 <= minimum_speed_similarity_ratio <= 1:
        raise ValueError("minimum_speed_similarity_ratio must be in [0, 1]")
    if not -1 <= maximum_displacement_cosine <= 1:
        raise ValueError("maximum_displacement_cosine must be in [-1, 1]")
    if not 0 <= maximum_cross_identity_endpoint_ratio <= 1:
        raise ValueError("maximum_cross_identity_endpoint_ratio must be in [0, 1]")
    if maximum_swap_pair_gap_seconds <= 0:
        raise ValueError("maximum_swap_pair_gap_seconds must be positive")
    if identity_contact_distance_body_lengths <= 0:
        raise ValueError("identity_contact_distance_body_lengths must be positive")
    if transition_speed_cap_multiplier <= 0:
        raise ValueError("transition_speed_cap_multiplier must be positive")
    if maximum_transition_expansion_seconds < 0:
        raise ValueError("maximum_transition_expansion_seconds must be non-negative")
    if not 0 <= interpolation_confidence <= 1:
        raise ValueError("interpolation_confidence must be in [0, 1]")

    raw_speed = compute_frame_speed_px_s(raw, fps)
    robust_means = np.full(2, np.nan, dtype=float)
    for animal in range(2):
        finite = raw_speed[:, animal][np.isfinite(raw_speed[:, animal])]
        if not finite.size:
            raise ValueError(f"No finite speeds for identity {animal + 1}")
        cutoff = float(np.quantile(finite, robust_mean_upper_quantile))
        robust_means[animal] = float(np.mean(finite[finite <= cutoff]))

    displacement = np.full_like(raw, np.nan, dtype=float)
    displacement[1:] = raw[1:] - raw[:-1]
    displacement_norm = np.linalg.norm(displacement, axis=2)
    denominator = displacement_norm[:, 0] * displacement_norm[:, 1]
    displacement_cosine = np.divide(
        np.einsum("ij,ij->i", displacement[:, 0], displacement[:, 1]),
        denominator,
        out=np.full(raw.shape[0], np.nan, dtype=float),
        where=denominator > 0,
    )
    minimum_speed = np.minimum(raw_speed[:, 0], raw_speed[:, 1])
    maximum_speed = np.maximum(raw_speed[:, 0], raw_speed[:, 1])
    speed_similarity = np.divide(
        minimum_speed,
        maximum_speed,
        out=np.full(raw.shape[0], np.nan, dtype=float),
        where=maximum_speed > 0,
    )
    both_high = np.all(raw_speed > speed_multiplier * robust_means[None, :], axis=1)
    pair_distance = np.linalg.norm(raw[:, 0] - raw[:, 1], axis=1)
    local_minimum_pair_distance = np.full(raw.shape[0], np.nan, dtype=float)
    contact_window_frames = max(0, int(contact_window_frames))
    for frame in range(raw.shape[0]):
        start = max(0, frame - contact_window_frames)
        end = min(raw.shape[0], frame + contact_window_frames + 1)
        finite = pair_distance[start:end]
        finite = finite[np.isfinite(finite)]
        if finite.size:
            local_minimum_pair_distance[frame] = float(np.min(finite))

    artifact_seed_mask = (
        both_high
        & (speed_similarity >= minimum_speed_similarity_ratio)
        & (displacement_cosine <= maximum_displacement_cosine)
        & (
            local_minimum_pair_distance
            <= contact_distance_body_lengths * body_length_px
        )
    )
    expanded_runs: list[tuple[int, int]] = []
    for start, end in _merge_short_gaps(
        _boolean_runs(artifact_seed_mask), max(0, int(max_seed_gap_frames))
    ):
        while start > 0 and both_high[start - 1]:
            start -= 1
        while end + 1 < raw.shape[0] and both_high[end + 1]:
            end += 1
        expanded_runs.append((start, end))
    expanded_runs = _merge_short_gaps(expanded_runs, max(0, int(max_seed_gap_frames)))

    smoothed = raw.copy()
    artifact_mask = np.zeros(raw.shape[0], dtype=bool)
    segment_id = np.zeros(raw.shape[0], dtype=int)
    identity_swap_mask = np.zeros(raw.shape[0], dtype=bool)
    paired_interval_mask = np.zeros(raw.shape[0], dtype=bool)
    swap_pair_id = np.zeros(raw.shape[0], dtype=int)
    events: list[dict[str, Any]] = []
    swap_pairs: list[dict[str, Any]] = []

    def frame_is_valid(frame: int) -> bool:
        return 0 <= frame < raw.shape[0] and bool(np.isfinite(raw[frame]).all())

    def assigned_points(frame: int, swapped: bool) -> np.ndarray:
        return raw[frame, ::-1] if swapped else raw[frame]

    def nearest_valid_left(frame: int, lower_bound: int = 0) -> int:
        while frame >= lower_bound and not frame_is_valid(frame):
            frame -= 1
        return frame

    def nearest_valid_right(frame: int, upper_bound: int) -> int:
        while frame <= upper_bound and not frame_is_valid(frame):
            frame += 1
        return frame

    speed_caps = transition_speed_cap_multiplier * robust_means
    maximum_expansion = int(round(maximum_transition_expansion_seconds * fps))

    def transition_anchors(
        start: int,
        end: int,
        *,
        pre_swapped: bool,
        post_swapped: bool,
        lower_bound: int = 0,
        upper_bound: int | None = None,
    ) -> tuple[int, int, np.ndarray]:
        upper_bound = raw.shape[0] - 1 if upper_bound is None else int(upper_bound)
        left = nearest_valid_left(start - 1, lower_bound)
        right = nearest_valid_right(end + 1, upper_bound)
        if left < lower_bound or right > upper_bound or right <= left + 1:
            raise ValueError("No valid transition anchors")
        expansions = 0
        while True:
            required = (
                np.linalg.norm(
                    assigned_points(right, post_swapped)
                    - assigned_points(left, pre_swapped),
                    axis=1,
                )
                * fps
                / (right - left)
            )
            if np.all(required <= speed_caps) or expansions >= maximum_expansion:
                return left, right, required
            next_left = nearest_valid_left(left - 1, lower_bound)
            next_right = nearest_valid_right(right + 1, upper_bound)
            changed = False
            if next_left >= lower_bound and next_left < left:
                left = next_left
                changed = True
            if next_right <= upper_bound and next_right > right:
                right = next_right
                changed = True
            if not changed:
                return left, right, required
            expansions += 1

    def make_event(
        *,
        source_start: int,
        source_end: int,
        left: int,
        right: int,
        event_type: str,
        repair_kind: str,
        pair_number: int,
        paired_segment_id: int = 0,
        required_speed: np.ndarray | None = None,
    ) -> dict[str, Any]:
        same_cost = float(np.linalg.norm(raw[right] - raw[left], axis=1).sum())
        cross_cost = float(np.linalg.norm(raw[right, ::-1] - raw[left], axis=1).sum())
        peak = np.nanmax(raw_speed[source_start : source_end + 1], axis=0)
        event = {
            "segment_id": len(events) + 1,
            "start_frame": int(source_start),
            "end_frame": int(source_end),
            "left_anchor_frame": int(left),
            "right_anchor_frame": int(right),
            "duration_frames": int(source_end - source_start + 1),
            "duration_s": (source_end - source_start + 1) / fps,
            "minimum_pair_distance_px": float(
                np.nanmin(pair_distance[source_start : source_end + 1])
            ),
            "blue_peak_raw_px_s": float(peak[0]),
            "red_peak_raw_px_s": float(peak[1]),
            "same_identity_endpoint_cost_px": same_cost,
            "cross_identity_endpoint_cost_px": cross_cost,
            "cross_over_same_endpoint_cost": (
                float(cross_cost / same_cost) if same_cost > 0 else None
            ),
            "event_type": event_type,
            "repair_kind": repair_kind,
            "swap_pair_id": int(pair_number),
            "paired_transition_segment_id": int(paired_segment_id),
            "blue_repaired_transition_speed_px_s": (
                float(required_speed[0]) if required_speed is not None else None
            ),
            "red_repaired_transition_speed_px_s": (
                float(required_speed[1]) if required_speed is not None else None
            ),
        }
        events.append(event)
        return event

    def interpolate_transition(
        event: dict[str, Any], left_points: np.ndarray, right_points: np.ndarray
    ) -> None:
        left = int(event["left_anchor_frame"])
        right = int(event["right_anchor_frame"])
        for frame in range(left + 1, right):
            alpha = (frame - left) / (right - left)
            smoothed[frame] = left_points + alpha * (right_points - left_points)
        artifact_mask[left + 1 : right] = True
        segment_id[left + 1 : right] = int(event["segment_id"])

    any_high = np.any(raw_speed > speed_multiplier * robust_means[None, :], axis=1)
    high_runs = _merge_short_gaps(
        _boolean_runs(any_high), max(0, int(max_seed_gap_frames))
    )
    contact_runs = _merge_short_gaps(
        _boolean_runs(
            pair_distance <= identity_contact_distance_body_lengths * body_length_px
        ),
        max(0, int(max_seed_gap_frames)),
    )
    maximum_pair_gap = int(round(maximum_swap_pair_gap_seconds * fps))
    used_high_runs: set[int] = set()
    occupied_until = -1
    for contact_start, contact_end in contact_runs:
        if contact_start <= occupied_until:
            continue
        exit_index = next(
            (
                index
                for index, (speed_start, _speed_end) in enumerate(high_runs)
                if index not in used_high_runs
                and speed_start > contact_end
                and speed_start - contact_end <= maximum_pair_gap
            ),
            None,
        )
        if exit_index is None:
            continue
        exit_start, exit_end = high_runs[exit_index]
        pair_number = len(swap_pairs) + 1
        try:
            entry_left, entry_right, entry_speed = transition_anchors(
                contact_start,
                contact_end,
                pre_swapped=False,
                post_swapped=True,
                upper_bound=exit_start - 1,
            )
        except ValueError:
            continue
        entry_same = float(np.linalg.norm(raw[entry_right] - raw[entry_left], axis=1).sum())
        entry_cross = float(
            np.linalg.norm(raw[entry_right, ::-1] - raw[entry_left], axis=1).sum()
        )
        if entry_same <= 0 or entry_cross / entry_same > maximum_cross_identity_endpoint_ratio:
            continue
        try:
            exit_left, exit_right, exit_speed = transition_anchors(
                exit_start,
                exit_end,
                pre_swapped=True,
                post_swapped=False,
                lower_bound=entry_right,
            )
        except ValueError:
            continue
        if entry_right > exit_left:
            continue

        entry_event = make_event(
            source_start=contact_start,
            source_end=contact_end,
            left=entry_left,
            right=entry_right,
            event_type="close_contact_identity_entry",
            repair_kind="paired_entry_transition_interpolation",
            pair_number=pair_number,
            required_speed=entry_speed,
        )
        exit_event = make_event(
            source_start=exit_start,
            source_end=exit_end,
            left=exit_left,
            right=exit_right,
            event_type="speed_discontinuity_identity_recovery",
            repair_kind="paired_exit_transition_interpolation",
            pair_number=pair_number,
            paired_segment_id=int(entry_event["segment_id"]),
            required_speed=exit_speed,
        )
        entry_event["paired_transition_segment_id"] = int(exit_event["segment_id"])
        interpolate_transition(
            entry_event, assigned_points(entry_left, False), assigned_points(entry_right, True)
        )
        smoothed[entry_right : exit_left + 1] = raw[entry_right : exit_left + 1, ::-1]
        identity_swap_mask[entry_right : exit_left + 1] = True
        interpolate_transition(
            exit_event, assigned_points(exit_left, True), assigned_points(exit_right, False)
        )
        paired_interval_mask[entry_left + 1 : exit_right] = True
        swap_pair_id[entry_left + 1 : exit_right] = pair_number
        artifact_seed_mask[contact_start : contact_end + 1] = True
        artifact_seed_mask[exit_start : exit_end + 1] = True
        swap_pairs.append(
            {
                "swap_pair_id": pair_number,
                "entry_segment_id": int(entry_event["segment_id"]),
                "exit_segment_id": int(exit_event["segment_id"]),
                "entry_contact_start_frame": int(contact_start),
                "entry_contact_end_frame": int(contact_end),
                "entry_contact_minimum_frame": int(
                    contact_start
                    + np.nanargmin(pair_distance[contact_start : contact_end + 1])
                ),
                "entry_left_anchor_frame": int(entry_left),
                "entry_right_anchor_frame": int(entry_right),
                "identity_swap_start_frame": int(entry_right),
                "identity_swap_end_frame": int(exit_left),
                "exit_speed_start_frame": int(exit_start),
                "exit_speed_end_frame": int(exit_end),
                "exit_left_anchor_frame": int(exit_left),
                "exit_right_anchor_frame": int(exit_right),
                "duration_frames": int(exit_right - entry_left - 1),
                "duration_s": (exit_right - entry_left - 1) / fps,
            }
        )
        used_high_runs.add(exit_index)
        occupied_until = exit_right - 1

    for start, end in expanded_runs:
        if paired_interval_mask[start : end + 1].any():
            continue
        left = nearest_valid_left(start - 1)
        right = nearest_valid_right(end + 1, raw.shape[0] - 1)
        if left < 0 or right >= raw.shape[0] or right <= left + 1:
            continue
        same_cost = float(np.linalg.norm(raw[right] - raw[left], axis=1).sum())
        cross_cost = float(np.linalg.norm(raw[right, ::-1] - raw[left], axis=1).sum())
        if same_cost > 0 and cross_cost / same_cost <= maximum_cross_identity_endpoint_ratio:
            continue
        event = make_event(
            source_start=start,
            source_end=end,
            left=left,
            right=right,
            event_type="local_coordinate_spike",
            repair_kind="local_linear_interpolation",
            pair_number=0,
        )
        interpolate_transition(event, smoothed[left], smoothed[right])

    smoothed_speed = compute_frame_speed_px_s(smoothed, fps)
    confidence = np.where(np.isfinite(raw).all(axis=2), 1.0, 0.0).astype(float)
    confidence[artifact_mask, :] = float(interpolation_confidence)
    parameters = {
        "speed_multiplier": float(speed_multiplier),
        "robust_mean_upper_quantile": float(robust_mean_upper_quantile),
        "minimum_speed_similarity_ratio": float(minimum_speed_similarity_ratio),
        "maximum_displacement_cosine": float(maximum_displacement_cosine),
        "contact_distance_body_lengths": float(contact_distance_body_lengths),
        "contact_window_frames": int(contact_window_frames),
        "max_seed_gap_frames": int(max_seed_gap_frames),
        "identity_contact_distance_body_lengths": float(
            identity_contact_distance_body_lengths
        ),
        "maximum_cross_identity_endpoint_ratio": float(
            maximum_cross_identity_endpoint_ratio
        ),
        "maximum_swap_pair_gap_seconds": float(maximum_swap_pair_gap_seconds),
        "transition_speed_cap_multiplier": float(transition_speed_cap_multiplier),
        "maximum_transition_expansion_seconds": float(
            maximum_transition_expansion_seconds
        ),
        "interpolation_confidence": float(interpolation_confidence),
        "body_length_px": float(body_length_px),
        "fps": float(fps),
        "method": "contact_to_recovery_identity_swap_with_transition_interpolation",
    }
    return TrajectorySmoothingResult(
        raw_trajectories=raw,
        smoothed_trajectories=smoothed,
        raw_speed_px_s=raw_speed,
        smoothed_speed_px_s=smoothed_speed,
        artifact_seed_mask=artifact_seed_mask,
        artifact_mask=artifact_mask,
        segment_id=segment_id,
        identity_swap_mask=identity_swap_mask,
        paired_interval_mask=paired_interval_mask,
        swap_pair_id=swap_pair_id,
        centroid_confidence=confidence,
        robust_mean_speed_px_s=robust_means,
        parameters=parameters,
        events=events,
        swap_pairs=swap_pairs,
    )


def _finite_or_blank(value: float) -> float | str:
    return float(value) if np.isfinite(value) else ""


def _velocity_confidence(
    centroid_confidence: np.ndarray, velocity_valid: np.ndarray, method: str
) -> np.ndarray:
    result = np.zeros_like(centroid_confidence, dtype=float)
    if centroid_confidence.shape[0] < 2:
        return result
    if method == "backward":
        result[1:] = np.minimum(centroid_confidence[1:], centroid_confidence[:-1])
    else:
        result[0] = np.minimum(centroid_confidence[0], centroid_confidence[1])
        result[-1] = np.minimum(centroid_confidence[-2], centroid_confidence[-1])
        if centroid_confidence.shape[0] > 2:
            result[1:-1] = np.minimum(
                centroid_confidence[:-2], centroid_confidence[2:]
            )
    result[~velocity_valid] = 0.0
    return result


def load_idtracker_npz(path: Path) -> dict[str, np.ndarray | float]:
    with np.load(path, allow_pickle=False) as payload:
        required = {"frame_id", "identity", "fps", "centroids_px"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"Missing idtracker NPZ keys: {sorted(missing)}")
        return {
            "frame_id": np.asarray(payload["frame_id"], dtype=np.int64),
            "identity": np.asarray(payload["identity"], dtype=np.int64),
            "fps": float(np.asarray(payload["fps"]).reshape(())),
            "centroids_px": np.asarray(payload["centroids_px"], dtype=float),
            "centroid_confidence_v0": (
                np.asarray(payload["centroid_confidence_v0"], dtype=float)
                if "centroid_confidence_v0" in payload.files
                else np.isfinite(payload["centroids_px"]).all(axis=-1).astype(float)
            ),
        }


def load_body_length_px(session_dir: Path) -> float:
    session_json = session_dir / "session.json"
    if not session_json.is_file():
        raise FileNotFoundError(session_json)
    metadata = json.loads(session_json.read_text(encoding="utf-8"))
    value = float(metadata.get("median_body_length") or 0.0)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"session.json has no positive median_body_length: {session_json}")
    return value


def write_smoothed_contract(
    result: TrajectorySmoothingResult,
    *,
    source_npz: Path,
    session_dir: Path,
    output_dir: Path,
    output_stem: str,
    frame_ids: np.ndarray,
    identities: np.ndarray,
    input_centroid_confidence: np.ndarray,
    velocity_method: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fps = float(result.parameters["fps"])
    centroid_valid = np.isfinite(result.smoothed_trajectories).all(axis=-1)
    centroid_confidence = np.minimum(
        np.asarray(input_centroid_confidence, dtype=float), result.centroid_confidence
    )
    centroid_confidence[~centroid_valid] = 0.0
    velocity, velocity_valid = compute_velocity_px_s(
        result.smoothed_trajectories, fps, velocity_method
    )
    speed = np.linalg.norm(velocity, axis=-1)
    raw_velocity, raw_velocity_valid = compute_velocity_px_s(
        result.raw_trajectories, fps, velocity_method
    )
    raw_speed = np.linalg.norm(raw_velocity, axis=-1)
    velocity_confidence = _velocity_confidence(
        centroid_confidence, velocity_valid, velocity_method
    )

    csv_path = output_dir / f"{output_stem}_idtracker_kinematics_smoothed.csv"
    npz_path = output_dir / f"{output_stem}_idtracker_kinematics_smoothed.npz"
    events_path = output_dir / f"{output_stem}_idtracker_smoothing_events.csv"
    summary_path = output_dir / f"{output_stem}_idtracker_smoothing_summary.json"
    fieldnames = [
        "frame_id", "time_s", "identity", "centroid_x_px", "centroid_y_px",
        "centroid_valid", "centroid_confidence_v0", "velocity_x_px_s",
        "velocity_y_px_s", "speed_px_s", "velocity_valid",
        "velocity_confidence_v0", "raw_centroid_x_px", "raw_centroid_y_px",
        "raw_velocity_x_px_s", "raw_velocity_y_px_s", "raw_speed_px_s",
        "identity_channels_swapped", "transition_interpolated",
        "paired_swap_repair", "swap_pair_id", "smoothing_segment_id",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for frame_index, frame_id in enumerate(frame_ids):
            for animal_index, identity in enumerate(identities):
                writer.writerow(
                    {
                        "frame_id": int(frame_id),
                        "time_s": int(frame_id) / fps,
                        "identity": int(identity),
                        "centroid_x_px": _finite_or_blank(result.smoothed_trajectories[frame_index, animal_index, 0]),
                        "centroid_y_px": _finite_or_blank(result.smoothed_trajectories[frame_index, animal_index, 1]),
                        "centroid_valid": int(centroid_valid[frame_index, animal_index]),
                        "centroid_confidence_v0": float(centroid_confidence[frame_index, animal_index]),
                        "velocity_x_px_s": _finite_or_blank(velocity[frame_index, animal_index, 0]),
                        "velocity_y_px_s": _finite_or_blank(velocity[frame_index, animal_index, 1]),
                        "speed_px_s": _finite_or_blank(speed[frame_index, animal_index]),
                        "velocity_valid": int(velocity_valid[frame_index, animal_index]),
                        "velocity_confidence_v0": float(velocity_confidence[frame_index, animal_index]),
                        "raw_centroid_x_px": _finite_or_blank(result.raw_trajectories[frame_index, animal_index, 0]),
                        "raw_centroid_y_px": _finite_or_blank(result.raw_trajectories[frame_index, animal_index, 1]),
                        "raw_velocity_x_px_s": _finite_or_blank(raw_velocity[frame_index, animal_index, 0]),
                        "raw_velocity_y_px_s": _finite_or_blank(raw_velocity[frame_index, animal_index, 1]),
                        "raw_speed_px_s": _finite_or_blank(raw_speed[frame_index, animal_index]),
                        "identity_channels_swapped": int(result.identity_swap_mask[frame_index]),
                        "transition_interpolated": int(result.artifact_mask[frame_index]),
                        "paired_swap_repair": int(result.paired_interval_mask[frame_index]),
                        "swap_pair_id": int(result.swap_pair_id[frame_index]),
                        "smoothing_segment_id": int(result.segment_id[frame_index]),
                    }
                )

    np.savez_compressed(
        npz_path,
        frame_id=frame_ids,
        identity=identities,
        fps=np.asarray(fps, dtype=np.float64),
        centroids_px=result.smoothed_trajectories,
        centroid_valid=centroid_valid,
        centroid_confidence_v0=centroid_confidence.astype(np.float32),
        velocity_px_s=velocity,
        speed_px_s=speed,
        velocity_valid=velocity_valid,
        velocity_confidence_v0=velocity_confidence.astype(np.float32),
        raw_centroids_px=result.raw_trajectories,
        raw_velocity_px_s=raw_velocity,
        raw_speed_px_s=raw_speed,
        raw_velocity_valid=raw_velocity_valid,
        artifact_seed_mask=result.artifact_seed_mask,
        transition_interpolated=result.artifact_mask,
        smoothing_segment_id=result.segment_id,
        identity_channels_swapped=result.identity_swap_mask,
        paired_swap_repair=result.paired_interval_mask,
        swap_pair_id=result.swap_pair_id,
    )
    if result.events:
        with events_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result.events[0].keys()))
            writer.writeheader()
            writer.writerows(result.events)
    else:
        events_path.write_text("", encoding="utf-8-sig")

    summary: dict[str, Any] = {
        "contract_version": "idtracker_identity_smoothing_v0",
        "source_npz": str(source_npz.resolve()),
        "session_dir": str(session_dir.resolve()),
        "fps": fps,
        "frame_count": int(result.raw_trajectories.shape[0]),
        "number_of_animals": int(result.raw_trajectories.shape[1]),
        "smoothing_applied": bool(result.events),
        "parameters": result.parameters,
        "robust_mean_speed_px_s": result.robust_mean_speed_px_s.tolist(),
        "transition_interpolated_frames": int(result.artifact_mask.sum()),
        "identity_swapped_frames": int(result.identity_swap_mask.sum()),
        "events": result.events,
        "swap_pairs": result.swap_pairs,
        "confidence_note": (
            "Only interpolated coordinates are assigned the configured low centroid "
            "confidence. Swapped-channel coordinates remain observed coordinates; "
            "identity_channels_swapped records the heuristic identity edit separately."
        ),
        "limitations": [
            "This optional heuristic is implemented only for exactly two animals.",
            "A smooth trajectory is not proof of biological identity during occlusion.",
            "Raw coordinates and modification masks are retained for audit and rollback.",
        ],
        "outputs": {
            "csv": str(csv_path),
            "npz": str(npz_path),
            "events_csv": str(events_path),
            "summary_json": str(summary_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def smooth_exported_idtracker_kinematics(
    *,
    input_npz: Path,
    session_dir: Path,
    output_dir: Path,
    output_stem: str,
    velocity_method: str = "backward",
    **smoothing_parameters: Any,
) -> dict[str, Any]:
    payload = load_idtracker_npz(input_npz)
    result = detect_and_smooth_contact_speed_artifacts(
        np.asarray(payload["centroids_px"]),
        fps=float(payload["fps"]),
        body_length_px=load_body_length_px(session_dir),
        **smoothing_parameters,
    )
    return write_smoothed_contract(
        result,
        source_npz=input_npz,
        session_dir=session_dir,
        output_dir=output_dir,
        output_stem=output_stem,
        frame_ids=np.asarray(payload["frame_id"]),
        identities=np.asarray(payload["identity"]),
        input_centroid_confidence=np.asarray(payload["centroid_confidence_v0"]),
        velocity_method=velocity_method,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optionally repair two-animal idtracker.ai identity exchanges."
    )
    parser.add_argument("--input-npz", required=True, type=Path)
    parser.add_argument("--session-dir", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--output-stem", required=True)
    parser.add_argument("--velocity-method", choices=("backward", "central"), default="backward")
    parser.add_argument("--speed-multiplier", type=float, default=5.0)
    parser.add_argument("--robust-mean-upper-quantile", type=float, default=0.99)
    parser.add_argument("--minimum-speed-similarity-ratio", type=float, default=0.80)
    parser.add_argument("--maximum-displacement-cosine", type=float, default=-0.80)
    parser.add_argument("--contact-distance-body-lengths", type=float, default=0.35)
    parser.add_argument("--contact-window-frames", type=int, default=2)
    parser.add_argument("--max-seed-gap-frames", type=int, default=2)
    parser.add_argument("--identity-contact-distance-body-lengths", type=float, default=0.20)
    parser.add_argument("--maximum-cross-identity-endpoint-ratio", type=float, default=0.25)
    parser.add_argument("--maximum-swap-pair-gap-seconds", type=float, default=5.0)
    parser.add_argument("--transition-speed-cap-multiplier", type=float, default=5.0)
    parser.add_argument("--maximum-transition-expansion-seconds", type=float, default=1.0)
    parser.add_argument("--interpolation-confidence", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    parameter_names = (
        "speed_multiplier", "robust_mean_upper_quantile",
        "minimum_speed_similarity_ratio", "maximum_displacement_cosine",
        "contact_distance_body_lengths", "contact_window_frames",
        "max_seed_gap_frames", "identity_contact_distance_body_lengths",
        "maximum_cross_identity_endpoint_ratio", "maximum_swap_pair_gap_seconds",
        "transition_speed_cap_multiplier", "maximum_transition_expansion_seconds",
        "interpolation_confidence",
    )
    summary = smooth_exported_idtracker_kinematics(
        input_npz=args.input_npz,
        session_dir=args.session_dir,
        output_dir=args.outdir,
        output_stem=args.output_stem,
        velocity_method=args.velocity_method,
        **{name: getattr(args, name) for name in parameter_names},
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
