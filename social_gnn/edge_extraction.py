"""Build confidence-aware social edge features from aligned video outputs.

Inputs are the frame-aligned idtracker.ai kinematics CSV and the MouseGPT table
already matched to idtracker.ai identities. Every output graph uses one interval
from the authoritative node NPZ clock.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:  # Support both ``python -m social_gnn...`` and direct script execution.
    from .authoritative_clock import load_node_clock, validate_fixed_patch_grid
except ImportError:  # pragma: no cover - exercised by Windows pipeline execution
    from authoritative_clock import load_node_clock, validate_fixed_patch_grid


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

TRAIT_UNITS = (
    "px",
    "px",
    "px",
    "unitless",
    "unitless",
    "px/s",
    "px/s",
    "unitless",
)

STATE_TRAIT_INDICES = (0, 1, 2, 3, 4)
DYNAMIC_TRAIT_INDICES = (5, 6, 7)


@dataclass
class FrameInputs:
    frame_ids: np.ndarray
    frame_time_s: np.ndarray
    identities: np.ndarray
    position_px: np.ndarray
    position_confidence: np.ndarray
    velocity_px_s: np.ndarray
    velocity_confidence: np.ndarray
    nose_px: np.ndarray
    nose_confidence: np.ndarray
    head_direction: np.ndarray
    head_direction_confidence: np.ndarray


@dataclass
class EdgeExtractionResult:
    frame_ids: np.ndarray
    frame_time_s: np.ndarray
    identities: np.ndarray
    edge_index: np.ndarray
    edge_identity: np.ndarray
    frame_edge_value: np.ndarray
    frame_edge_confidence: np.ndarray
    social_step: np.ndarray
    patch_start_s: np.ndarray
    patch_end_s: np.ndarray
    patch_center_s: np.ndarray
    patch_delta_t_s: np.ndarray
    patch_frame_count: np.ndarray
    patch_centroid_px: np.ndarray
    patch_centroid_confidence: np.ndarray
    patch_centroid_coverage: np.ndarray
    edge_value: np.ndarray
    edge_confidence: np.ndarray
    edge_coverage: np.ndarray
    edge_valid_frame_count: np.ndarray
    parameters: dict[str, Any]


def _numeric(data: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(data[column], errors="coerce").to_numpy(float)


def _boolean(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(bool)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).to_numpy(float) != 0
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "yes", "y"})
        .to_numpy(bool)
    )


def _clip_confidence(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return float(np.clip(value, 0.0, 1.0)) if np.isfinite(value) else 0.0


def _prepare_idtracker(idtracker: pd.DataFrame) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    required = {
        "frame_id",
        "time_s",
        "identity",
        "centroid_x_px",
        "centroid_y_px",
        "centroid_valid",
    }
    missing = sorted(required.difference(idtracker.columns))
    if missing:
        raise ValueError(f"idtracker CSV lacks required columns: {missing}")

    data = idtracker.copy()
    data["frame_id"] = pd.to_numeric(data["frame_id"], errors="raise").astype(int)
    data["identity"] = pd.to_numeric(data["identity"], errors="raise").astype(int)
    if data.duplicated(["frame_id", "identity"]).any():
        raise ValueError("idtracker CSV contains duplicate (frame_id, identity) rows")

    frame_ids = np.sort(data["frame_id"].unique().astype(np.int64))
    identities = np.sort(data["identity"].unique().astype(np.int64))
    if len(identities) < 2:
        raise ValueError("Social edge extraction requires at least two identities")
    frame_lookup = {int(value): index for index, value in enumerate(frame_ids)}
    identity_lookup = {int(value): index for index, value in enumerate(identities)}
    n_frames, n_animals = len(frame_ids), len(identities)

    frame_time_s = np.full(n_frames, np.nan, dtype=float)
    position = np.full((n_frames, n_animals, 2), np.nan, dtype=float)
    position_confidence = np.zeros((n_frames, n_animals), dtype=float)
    input_velocity_confidence = np.ones((n_frames, n_animals), dtype=float)
    centroid_valid = _boolean(data["centroid_valid"])
    has_position_confidence = "centroid_confidence_v0" in data
    has_velocity_confidence = "velocity_confidence_v0" in data

    for row_number, row in data.reset_index(drop=True).iterrows():
        frame_index = frame_lookup[int(row["frame_id"])]
        animal_index = identity_lookup[int(row["identity"])]
        time_s = float(row["time_s"])
        if not np.isfinite(time_s):
            raise ValueError(f"Non-finite time_s at idtracker row {row_number}")
        if np.isfinite(frame_time_s[frame_index]) and not math.isclose(
            frame_time_s[frame_index], time_s, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(f"Identities disagree on time_s for frame {row['frame_id']}")
        frame_time_s[frame_index] = time_s

        point = np.asarray([row["centroid_x_px"], row["centroid_y_px"]], dtype=float)
        if centroid_valid[row_number] and np.isfinite(point).all():
            position[frame_index, animal_index] = point
            confidence = (
                _clip_confidence(row["centroid_confidence_v0"])
                if has_position_confidence
                else 1.0
            )
            position_confidence[frame_index, animal_index] = confidence
        if has_velocity_confidence:
            input_velocity_confidence[frame_index, animal_index] = _clip_confidence(
                row["velocity_confidence_v0"]
            )

    if not np.isfinite(frame_time_s).all():
        raise ValueError("Could not determine time_s for every idtracker frame")
    if np.any(np.diff(frame_time_s) <= 0):
        raise ValueError("idtracker frame timestamps must be strictly increasing")

    velocity = np.full_like(position, np.nan, dtype=float)
    velocity_confidence = np.zeros((n_frames, n_animals), dtype=float)
    for frame_index in range(1, n_frames):
        if frame_ids[frame_index] != frame_ids[frame_index - 1] + 1:
            continue
        delta_t = frame_time_s[frame_index] - frame_time_s[frame_index - 1]
        if not np.isfinite(delta_t) or delta_t <= 0:
            continue
        valid = (
            np.isfinite(position[frame_index]).all(axis=-1)
            & np.isfinite(position[frame_index - 1]).all(axis=-1)
            & (position_confidence[frame_index] > 0)
            & (position_confidence[frame_index - 1] > 0)
        )
        velocity[frame_index, valid] = (
            position[frame_index, valid] - position[frame_index - 1, valid]
        ) / delta_t
        velocity_confidence[frame_index, valid] = np.minimum.reduce(
            [
                position_confidence[frame_index, valid],
                position_confidence[frame_index - 1, valid],
                input_velocity_confidence[frame_index, valid],
            ]
        )
    return (
        frame_ids,
        frame_time_s,
        identities,
        position,
        position_confidence,
        velocity,
        velocity_confidence,
    )


def _prepare_mousegpt_pose(
    mousegpt: pd.DataFrame,
    frame_ids: np.ndarray,
    identities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = {
        "frame_id",
        "idtracker_identity",
        "association_valid",
        "association_confidence",
        "Snout_x",
        "Snout_y",
        "Snout_score",
        "direction_x",
        "direction_y",
        "direction_confidence",
    }
    missing = sorted(required.difference(mousegpt.columns))
    if missing:
        raise ValueError(f"Matched MouseGPT CSV lacks required columns: {missing}")

    frame_lookup = {int(value): index for index, value in enumerate(frame_ids)}
    identity_lookup = {int(value): index for index, value in enumerate(identities)}
    n_frames, n_animals = len(frame_ids), len(identities)
    nose = np.full((n_frames, n_animals, 2), np.nan, dtype=float)
    nose_confidence = np.zeros((n_frames, n_animals), dtype=float)
    direction = np.full((n_frames, n_animals, 2), np.nan, dtype=float)
    direction_confidence = np.zeros((n_frames, n_animals), dtype=float)

    data = mousegpt.copy().reset_index(drop=True)
    valid_association = _boolean(data["association_valid"])
    frame_values = pd.to_numeric(data["frame_id"], errors="coerce")
    identity_values = pd.to_numeric(data["idtracker_identity"], errors="coerce")
    order = np.argsort(-np.nan_to_num(_numeric(data, "association_confidence"), nan=-1.0))
    occupied: set[tuple[int, int]] = set()
    for row_index in order:
        if not valid_association[row_index]:
            continue
        frame_value = frame_values.iloc[row_index]
        identity_value = identity_values.iloc[row_index]
        if not np.isfinite(frame_value) or not np.isfinite(identity_value):
            continue
        frame_id, identity = int(frame_value), int(identity_value)
        if frame_id not in frame_lookup or identity not in identity_lookup:
            continue
        key = frame_id, identity
        if key in occupied:
            continue
        occupied.add(key)
        frame_index = frame_lookup[frame_id]
        animal_index = identity_lookup[identity]
        row = data.iloc[row_index]
        association_confidence = _clip_confidence(row["association_confidence"])

        nose_point = np.asarray([row["Snout_x"], row["Snout_y"]], dtype=float)
        nose_score = _clip_confidence(row["Snout_score"])
        if association_confidence > 0 and nose_score > 0 and np.isfinite(nose_point).all():
            nose[frame_index, animal_index] = nose_point
            nose_confidence[frame_index, animal_index] = min(
                association_confidence, nose_score
            )

        vector = np.asarray([row["direction_x"], row["direction_y"]], dtype=float)
        vector_norm = float(np.linalg.norm(vector))
        head_score = _clip_confidence(row["direction_confidence"])
        if (
            association_confidence > 0
            and head_score > 0
            and np.isfinite(vector).all()
            and vector_norm > 1e-12
        ):
            direction[frame_index, animal_index] = vector / vector_norm
            direction_confidence[frame_index, animal_index] = min(
                association_confidence, head_score
            )
    return nose, nose_confidence, direction, direction_confidence


def load_frame_inputs(
    idtracker_csv: Path | str, matched_mousegpt_csv: Path | str
) -> FrameInputs:
    idtracker_path = Path(idtracker_csv).resolve()
    mousegpt_path = Path(matched_mousegpt_csv).resolve()
    if not idtracker_path.is_file():
        raise FileNotFoundError(idtracker_path)
    if not mousegpt_path.is_file():
        raise FileNotFoundError(mousegpt_path)
    idtracker = pd.read_csv(idtracker_path)
    mousegpt = pd.read_csv(mousegpt_path)
    (
        frame_ids,
        frame_time_s,
        identities,
        position,
        position_confidence,
        velocity,
        velocity_confidence,
    ) = _prepare_idtracker(idtracker)
    nose, nose_confidence, direction, direction_confidence = _prepare_mousegpt_pose(
        mousegpt, frame_ids, identities
    )
    return FrameInputs(
        frame_ids=frame_ids,
        frame_time_s=frame_time_s,
        identities=identities,
        position_px=position,
        position_confidence=position_confidence,
        velocity_px_s=velocity,
        velocity_confidence=velocity_confidence,
        nose_px=nose,
        nose_confidence=nose_confidence,
        head_direction=direction,
        head_direction_confidence=direction_confidence,
    )


def complete_directed_edges(identities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sources: list[int] = []
    targets: list[int] = []
    for source in range(len(identities)):
        for target in range(len(identities)):
            if source != target:
                sources.append(source)
                targets.append(target)
    edge_index = np.asarray([sources, targets], dtype=np.int64)
    edge_identity = np.asarray(
        [[identities[source], identities[target]] for source, target in zip(sources, targets)],
        dtype=np.int64,
    ).T
    return edge_index, edge_identity


def compute_frame_edge_traits(
    inputs: FrameInputs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the five state-like traits at frame resolution.

    Dynamic trait channels remain zero here and are derived only after stable
    patch-level centroid aggregation.
    """
    edge_index, edge_identity = complete_directed_edges(inputs.identities)
    n_frames, n_edges = len(inputs.frame_ids), edge_index.shape[1]
    values = np.zeros((n_frames, n_edges, len(TRAIT_NAMES)), dtype=np.float32)
    confidence = np.zeros_like(values)

    for edge_number, (source, target) in enumerate(edge_index.T):
        p_source = inputs.position_px[:, source]
        p_target = inputs.position_px[:, target]
        q_p_source = inputs.position_confidence[:, source]
        q_p_target = inputs.position_confidence[:, target]
        pair_vector = p_target - p_source
        pair_distance = np.linalg.norm(pair_vector, axis=1)
        pair_valid = (
            np.isfinite(pair_vector).all(axis=1)
            & (pair_distance > 1e-12)
            & (q_p_source > 0)
            & (q_p_target > 0)
        )
        pair_direction = np.zeros_like(pair_vector)
        pair_direction[pair_valid] = (
            pair_vector[pair_valid] / pair_distance[pair_valid, None]
        )
        values[pair_valid, edge_number, 0] = pair_distance[pair_valid]
        confidence[pair_valid, edge_number, 0] = np.minimum(
            q_p_source[pair_valid], q_p_target[pair_valid]
        )

        nose_source = inputs.nose_px[:, source]
        nose_target = inputs.nose_px[:, target]
        q_nose_source = inputs.nose_confidence[:, source]
        q_nose_target = inputs.nose_confidence[:, target]
        nose_pair_valid = (
            np.isfinite(nose_source).all(axis=1)
            & np.isfinite(nose_target).all(axis=1)
            & (q_nose_source > 0)
            & (q_nose_target > 0)
        )
        values[nose_pair_valid, edge_number, 1] = np.linalg.norm(
            nose_target[nose_pair_valid] - nose_source[nose_pair_valid], axis=1
        )
        confidence[nose_pair_valid, edge_number, 1] = np.minimum(
            q_nose_source[nose_pair_valid], q_nose_target[nose_pair_valid]
        )

        nose_to_target_valid = (
            np.isfinite(nose_source).all(axis=1)
            & np.isfinite(p_target).all(axis=1)
            & (q_nose_source > 0)
            & (q_p_target > 0)
        )
        values[nose_to_target_valid, edge_number, 2] = np.linalg.norm(
            p_target[nose_to_target_valid] - nose_source[nose_to_target_valid],
            axis=1,
        )
        confidence[nose_to_target_valid, edge_number, 2] = np.minimum(
            q_nose_source[nose_to_target_valid], q_p_target[nose_to_target_valid]
        )

        head_source = inputs.head_direction[:, source]
        head_target = inputs.head_direction[:, target]
        q_head_source = inputs.head_direction_confidence[:, source]
        q_head_target = inputs.head_direction_confidence[:, target]
        faces_valid = pair_valid & np.isfinite(head_source).all(axis=1) & (q_head_source > 0)
        values[faces_valid, edge_number, 3] = np.clip(
            np.einsum(
                "ij,ij->i", head_source[faces_valid], pair_direction[faces_valid]
            ),
            -1.0,
            1.0,
        )
        confidence[faces_valid, edge_number, 3] = np.minimum.reduce(
            [
                q_head_source[faces_valid],
                q_p_source[faces_valid],
                q_p_target[faces_valid],
            ]
        )

        head_alignment_valid = (
            np.isfinite(head_source).all(axis=1)
            & np.isfinite(head_target).all(axis=1)
            & (q_head_source > 0)
            & (q_head_target > 0)
        )
        values[head_alignment_valid, edge_number, 4] = np.clip(
            np.einsum(
                "ij,ij->i",
                head_source[head_alignment_valid],
                head_target[head_alignment_valid],
            ),
            -1.0,
            1.0,
        )
        confidence[head_alignment_valid, edge_number, 4] = np.minimum(
            q_head_source[head_alignment_valid], q_head_target[head_alignment_valid]
        )

    return edge_index, edge_identity, values, confidence


def aggregate_fixed_patches(
    *,
    frame_time_s: np.ndarray,
    frame_edge_value: np.ndarray,
    frame_edge_confidence: np.ndarray,
    patch_start_s: np.ndarray,
    patch_end_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate with weighted means and separate confidence/coverage channels.

    Patch confidence is ``sum(frame confidence) / number of frames in patch``.
    Equivalently it is valid-frame coverage times mean confidence on valid frames.
    This function is used for state-like traits. Dynamic channels are computed
    later from adjacent patch summaries and are not frame-averaged.
    """
    times = np.asarray(frame_time_s, dtype=float)
    values = np.asarray(frame_edge_value, dtype=float)
    confidence = np.asarray(frame_edge_confidence, dtype=float)
    if values.shape != confidence.shape or values.ndim != 3:
        raise ValueError("frame edge value/confidence must share shape [F, E, R]")
    n_patches = len(patch_start_s)
    n_edges, n_traits = values.shape[1:]
    patch_value = np.zeros((n_patches, n_edges, n_traits), dtype=np.float32)
    patch_confidence = np.zeros_like(patch_value)
    patch_coverage = np.zeros_like(patch_value)
    valid_frame_count = np.zeros((n_patches, n_edges, n_traits), dtype=np.int32)
    patch_frame_count = np.zeros(n_patches, dtype=np.int32)

    for patch in range(n_patches):
        start = int(np.searchsorted(times, patch_start_s[patch], side="left"))
        end = int(np.searchsorted(times, patch_end_s[patch], side="left"))
        count = end - start
        patch_frame_count[patch] = count
        if count <= 0:
            continue
        local_values = values[start:end]
        local_confidence = confidence[start:end]
        for edge in range(n_edges):
            for trait in range(n_traits):
                q = local_confidence[:, edge, trait]
                x = local_values[:, edge, trait]
                valid = np.isfinite(x) & np.isfinite(q) & (q > 0)
                valid_count = int(valid.sum())
                valid_frame_count[patch, edge, trait] = valid_count
                patch_coverage[patch, edge, trait] = valid_count / count
                if not valid_count:
                    continue
                weight_sum = float(q[valid].sum())
                if weight_sum <= 0:
                    continue
                patch_value[patch, edge, trait] = float(
                    np.dot(x[valid], q[valid]) / weight_sum
                )
                patch_confidence[patch, edge, trait] = weight_sum / count
    return (
        patch_frame_count,
        patch_value,
        patch_confidence,
        patch_coverage,
        valid_frame_count,
    )


def aggregate_patch_centroids(
    *,
    frame_time_s: np.ndarray,
    position_px: np.ndarray,
    position_confidence: np.ndarray,
    patch_start_s: np.ndarray,
    patch_end_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build stable confidence-weighted centroid summaries for every patch."""
    times = np.asarray(frame_time_s, dtype=float)
    positions = np.asarray(position_px, dtype=float)
    confidence = np.asarray(position_confidence, dtype=float)
    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError("position_px must have shape [F,N,2]")
    if confidence.shape != positions.shape[:2] or len(times) != len(positions):
        raise ValueError("position arrays and frame_time_s are not aligned")

    n_patches = len(patch_start_s)
    n_animals = positions.shape[1]
    patch_frame_count = np.zeros(n_patches, dtype=np.int32)
    centroid = np.zeros((n_patches, n_animals, 2), dtype=np.float32)
    centroid_confidence = np.zeros((n_patches, n_animals), dtype=np.float32)
    centroid_coverage = np.zeros_like(centroid_confidence)
    centroid_valid_count = np.zeros((n_patches, n_animals), dtype=np.int32)

    for patch in range(n_patches):
        start = int(np.searchsorted(times, patch_start_s[patch], side="left"))
        end = int(np.searchsorted(times, patch_end_s[patch], side="left"))
        count = end - start
        patch_frame_count[patch] = count
        if count <= 0:
            continue
        for animal in range(n_animals):
            q = confidence[start:end, animal]
            points = positions[start:end, animal]
            valid = np.isfinite(points).all(axis=1) & np.isfinite(q) & (q > 0)
            valid_count = int(valid.sum())
            centroid_valid_count[patch, animal] = valid_count
            centroid_coverage[patch, animal] = valid_count / count
            if not valid_count:
                continue
            weight_sum = float(q[valid].sum())
            if weight_sum <= 0:
                continue
            centroid[patch, animal] = np.average(
                points[valid], axis=0, weights=q[valid]
            )
            centroid_confidence[patch, animal] = weight_sum / count
    return (
        patch_frame_count,
        centroid,
        centroid_confidence,
        centroid_coverage,
        centroid_valid_count,
    )


def compute_patch_dynamic_traits(
    *,
    patch_center_s: np.ndarray,
    edge_index: np.ndarray,
    patch_centroid_px: np.ndarray,
    patch_centroid_confidence: np.ndarray,
    patch_centroid_coverage: np.ndarray,
    patch_centroid_valid_count: np.ndarray,
    state_edge_value: np.ndarray,
    state_edge_confidence: np.ndarray,
    state_edge_coverage: np.ndarray,
    state_edge_valid_count: np.ndarray,
    minimum_patch_displacement_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derive the three dynamic traits from adjacent patch summaries.

    Dynamic values at index ``t`` use the forward transition ``t -> t+1``.
    The final patch has no forward transition and therefore retains zero value,
    confidence, coverage, and valid-count outputs.
    """
    if not np.isfinite(minimum_patch_displacement_px) or minimum_patch_displacement_px < 0:
        raise ValueError("minimum_patch_displacement_px must be finite and non-negative")
    centers = np.asarray(patch_center_s, dtype=float)
    dt = np.zeros(len(centers), dtype=np.float64)
    if len(centers) > 1:
        dt[:-1] = np.diff(centers)
        if np.any(dt[:-1] <= 0):
            raise ValueError("patch centers must be strictly increasing")

    n_patches = len(centers)
    n_edges = edge_index.shape[1]
    values = np.zeros((n_patches, n_edges, 3), dtype=np.float32)
    confidence = np.zeros_like(values)
    coverage = np.zeros_like(values)
    valid_count = np.zeros((n_patches, n_edges, 3), dtype=np.int32)
    eps = 1e-12

    for patch in range(n_patches - 1):
        delta_t = dt[patch]
        for edge, (source, target) in enumerate(edge_index.T):
            source_now = patch_centroid_px[patch, source]
            source_next = patch_centroid_px[patch + 1, source]
            target_now = patch_centroid_px[patch, target]
            target_next = patch_centroid_px[patch + 1, target]
            q_source_now = float(patch_centroid_confidence[patch, source])
            q_source_next = float(patch_centroid_confidence[patch + 1, source])
            q_target_now = float(patch_centroid_confidence[patch, target])
            q_target_next = float(patch_centroid_confidence[patch + 1, target])

            distance_q = min(
                float(state_edge_confidence[patch, edge, 0]),
                float(state_edge_confidence[patch + 1, edge, 0]),
            )
            if distance_q > 0:
                values[patch, edge, 1] = (
                    state_edge_value[patch, edge, 0]
                    - state_edge_value[patch + 1, edge, 0]
                ) / delta_t
                confidence[patch, edge, 1] = distance_q
                coverage[patch, edge, 1] = min(
                    state_edge_coverage[patch, edge, 0],
                    state_edge_coverage[patch + 1, edge, 0],
                )
                valid_count[patch, edge, 1] = min(
                    state_edge_valid_count[patch, edge, 0],
                    state_edge_valid_count[patch + 1, edge, 0],
                )

            pair_vector = target_now - source_now
            pair_distance = float(np.linalg.norm(pair_vector))
            radial_q = min(q_source_now, q_source_next, q_target_now)
            if radial_q > 0 and pair_distance > eps:
                source_velocity = (source_next - source_now) / delta_t
                values[patch, edge, 0] = float(
                    np.dot(source_velocity, pair_vector / pair_distance)
                )
                confidence[patch, edge, 0] = radial_q
                coverage[patch, edge, 0] = min(
                    patch_centroid_coverage[patch, source],
                    patch_centroid_coverage[patch + 1, source],
                    patch_centroid_coverage[patch, target],
                )
                valid_count[patch, edge, 0] = min(
                    patch_centroid_valid_count[patch, source],
                    patch_centroid_valid_count[patch + 1, source],
                    patch_centroid_valid_count[patch, target],
                )

            delta_source = source_next - source_now
            delta_target = target_next - target_now
            source_displacement = float(np.linalg.norm(delta_source))
            target_displacement = float(np.linalg.norm(delta_target))
            alignment_q = min(
                q_source_now, q_source_next, q_target_now, q_target_next
            )
            if (
                alignment_q > 0
                and source_displacement > eps
                and target_displacement > eps
            ):
                values[patch, edge, 2] = float(
                    np.clip(
                        np.dot(delta_source, delta_target)
                        / (source_displacement * target_displacement),
                        -1.0,
                        1.0,
                    )
                )
                reliability = 1.0
                if minimum_patch_displacement_px > 0:
                    reliability = min(
                        1.0,
                        source_displacement / minimum_patch_displacement_px,
                        target_displacement / minimum_patch_displacement_px,
                    )
                confidence[patch, edge, 2] = alignment_q * reliability
                coverage[patch, edge, 2] = min(
                    patch_centroid_coverage[patch, source],
                    patch_centroid_coverage[patch + 1, source],
                    patch_centroid_coverage[patch, target],
                    patch_centroid_coverage[patch + 1, target],
                )
                valid_count[patch, edge, 2] = min(
                    patch_centroid_valid_count[patch, source],
                    patch_centroid_valid_count[patch + 1, source],
                    patch_centroid_valid_count[patch, target],
                    patch_centroid_valid_count[patch + 1, target],
                )
    return dt, values, confidence, coverage, valid_count


def extract_social_edges_on_clock(
    inputs: FrameInputs,
    *,
    social_step: np.ndarray,
    patch_start_s: np.ndarray,
    patch_end_s: np.ndarray,
    minimum_patch_displacement_px: float = 1.0,
    clock_parameters: dict[str, Any] | None = None,
) -> EdgeExtractionResult:
    """Aggregate edges on caller-supplied physical time intervals."""
    social_step = np.asarray(social_step, dtype=np.int64)
    patch_start_s = np.asarray(patch_start_s, dtype=np.float64)
    patch_end_s = np.asarray(patch_end_s, dtype=np.float64)
    validate_fixed_patch_grid(
        patch_start_s,
        patch_end_s,
        social_step=social_step,
        label="edge extraction clock",
    )
    edge_index, edge_identity, frame_value, frame_confidence = (
        compute_frame_edge_traits(inputs)
    )
    (
        patch_frame_count,
        edge_value,
        edge_confidence,
        edge_coverage,
        valid_frame_count,
    ) = aggregate_fixed_patches(
        frame_time_s=inputs.frame_time_s,
        frame_edge_value=frame_value,
        frame_edge_confidence=frame_confidence,
        patch_start_s=patch_start_s,
        patch_end_s=patch_end_s,
    )
    (
        centroid_frame_count,
        patch_centroid,
        patch_centroid_confidence,
        patch_centroid_coverage,
        patch_centroid_valid_count,
    ) = aggregate_patch_centroids(
        frame_time_s=inputs.frame_time_s,
        position_px=inputs.position_px,
        position_confidence=inputs.position_confidence,
        patch_start_s=patch_start_s,
        patch_end_s=patch_end_s,
    )
    if not np.array_equal(patch_frame_count, centroid_frame_count):
        raise AssertionError("edge and centroid patch frame counts disagree")
    patch_center_s = (patch_start_s + patch_end_s) / 2.0
    (
        patch_delta_t_s,
        dynamic_value,
        dynamic_confidence,
        dynamic_coverage,
        dynamic_valid_count,
    ) = compute_patch_dynamic_traits(
        patch_center_s=patch_center_s,
        edge_index=edge_index,
        patch_centroid_px=patch_centroid,
        patch_centroid_confidence=patch_centroid_confidence,
        patch_centroid_coverage=patch_centroid_coverage,
        patch_centroid_valid_count=patch_centroid_valid_count,
        state_edge_value=edge_value,
        state_edge_confidence=edge_confidence,
        state_edge_coverage=edge_coverage,
        state_edge_valid_count=valid_frame_count,
        minimum_patch_displacement_px=minimum_patch_displacement_px,
    )
    edge_value[:, :, DYNAMIC_TRAIT_INDICES] = dynamic_value
    edge_confidence[:, :, DYNAMIC_TRAIT_INDICES] = dynamic_confidence
    edge_coverage[:, :, DYNAMIC_TRAIT_INDICES] = dynamic_coverage
    valid_frame_count[:, :, DYNAMIC_TRAIT_INDICES] = dynamic_valid_count
    parameters = dict(clock_parameters or {})
    parameters.update(
        {
            "patch_count": int(len(social_step)),
            "minimum_patch_displacement_px": float(
                minimum_patch_displacement_px
            ),
            "patch_interval_convention": "[patch_start_s, patch_end_s)",
            "dynamic_difference_policy": "forward t_to_t_plus_1",
            "final_patch_dynamic_policy": "zero_value_confidence_coverage",
            "dynamic_dt_source": "adjacent_patch_centers",
        }
    )
    return EdgeExtractionResult(
        frame_ids=inputs.frame_ids,
        frame_time_s=inputs.frame_time_s,
        identities=inputs.identities,
        edge_index=edge_index,
        edge_identity=edge_identity,
        frame_edge_value=frame_value,
        frame_edge_confidence=frame_confidence,
        social_step=social_step,
        patch_start_s=patch_start_s,
        patch_end_s=patch_end_s,
        patch_center_s=patch_center_s,
        patch_delta_t_s=patch_delta_t_s,
        patch_frame_count=patch_frame_count,
        patch_centroid_px=patch_centroid,
        patch_centroid_confidence=patch_centroid_confidence,
        patch_centroid_coverage=patch_centroid_coverage,
        edge_value=edge_value,
        edge_confidence=edge_confidence,
        edge_coverage=edge_coverage,
        edge_valid_frame_count=valid_frame_count,
        parameters=parameters,
    )


def _dense_edge_array(
    compact: np.ndarray, edge_index: np.ndarray, number_of_animals: int
) -> np.ndarray:
    dense = np.zeros(
        (compact.shape[0], number_of_animals, number_of_animals, compact.shape[-1]),
        dtype=compact.dtype,
    )
    for edge, (source, target) in enumerate(edge_index.T):
        dense[:, source, target] = compact[:, edge]
    return dense


def write_edge_outputs(
    result: EdgeExtractionResult,
    *,
    output_dir: Path | str,
    output_stem: str,
    idtracker_csv: Path | str,
    matched_mousegpt_csv: Path | str,
    save_frame_level: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{output_stem}_social_edges.csv"
    npz_path = output_dir / f"{output_stem}_social_edges.npz"
    summary_path = output_dir / f"{output_stem}_social_edges_summary.json"

    fieldnames = [
        "social_step",
        "patch_start_s",
        "patch_end_s",
        "patch_center_s",
        "patch_delta_t_s",
        "source_identity",
        "target_identity",
        "patch_frame_count",
    ]
    for trait in TRAIT_NAMES:
        fieldnames.extend(
            [
                f"{trait}_value",
                f"{trait}_confidence",
                f"{trait}_coverage",
                f"{trait}_valid_frames",
            ]
        )
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for patch, social_step in enumerate(result.social_step):
            for edge in range(result.edge_index.shape[1]):
                row: dict[str, Any] = {
                    "social_step": int(social_step),
                    "patch_start_s": float(result.patch_start_s[patch]),
                    "patch_end_s": float(result.patch_end_s[patch]),
                    "patch_center_s": float(result.patch_center_s[patch]),
                    "patch_delta_t_s": float(result.patch_delta_t_s[patch]),
                    "source_identity": int(result.edge_identity[0, edge]),
                    "target_identity": int(result.edge_identity[1, edge]),
                    "patch_frame_count": int(result.patch_frame_count[patch]),
                }
                for trait_index, trait in enumerate(TRAIT_NAMES):
                    row[f"{trait}_value"] = float(
                        result.edge_value[patch, edge, trait_index]
                    )
                    row[f"{trait}_confidence"] = float(
                        result.edge_confidence[patch, edge, trait_index]
                    )
                    row[f"{trait}_coverage"] = float(
                        result.edge_coverage[patch, edge, trait_index]
                    )
                    row[f"{trait}_valid_frames"] = int(
                        result.edge_valid_frame_count[patch, edge, trait_index]
                    )
                writer.writerow(row)

    number_of_animals = len(result.identities)
    payload: dict[str, Any] = {
        "contract_version": np.asarray("social_edge_features_v1"),
        "trait_names": np.asarray(TRAIT_NAMES),
        "trait_units": np.asarray(TRAIT_UNITS),
        "social_step": result.social_step,
        "patch_start_s": result.patch_start_s,
        "patch_end_s": result.patch_end_s,
        "patch_center_s": result.patch_center_s,
        "patch_delta_t_s": result.patch_delta_t_s,
        "patch_frame_count": result.patch_frame_count,
        "patch_centroid_px": result.patch_centroid_px,
        "patch_centroid_confidence": result.patch_centroid_confidence,
        "patch_centroid_coverage": result.patch_centroid_coverage,
        "identity": result.identities,
        "edge_index": result.edge_index,
        "edge_identity": result.edge_identity,
        "edge_value": result.edge_value,
        "edge_confidence": result.edge_confidence,
        "edge_coverage": result.edge_coverage,
        "edge_valid_frame_count": result.edge_valid_frame_count,
        "edge_value_dense": _dense_edge_array(
            result.edge_value, result.edge_index, number_of_animals
        ),
        "edge_confidence_dense": _dense_edge_array(
            result.edge_confidence, result.edge_index, number_of_animals
        ),
        "edge_coverage_dense": _dense_edge_array(
            result.edge_coverage, result.edge_index, number_of_animals
        ),
    }
    if save_frame_level:
        payload.update(
            {
                "frame_id": result.frame_ids,
                "frame_time_s": result.frame_time_s,
                "frame_edge_value": result.frame_edge_value,
                "frame_edge_confidence": result.frame_edge_confidence,
            }
        )
    np.savez_compressed(npz_path, **payload)

    summary: dict[str, Any] = {
        "contract_version": "social_edge_features_v1",
        "idtracker_csv": str(Path(idtracker_csv).resolve()),
        "matched_mousegpt_csv": str(Path(matched_mousegpt_csv).resolve()),
        "parameters": result.parameters,
        "trait_names": list(TRAIT_NAMES),
        "trait_units": dict(zip(TRAIT_NAMES, TRAIT_UNITS)),
        "shapes": {
            "edge_value": list(result.edge_value.shape),
            "edge_confidence": list(result.edge_confidence.shape),
            "edge_coverage": list(result.edge_coverage.shape),
            "edge_index": list(result.edge_index.shape),
        },
        "aggregation": {
            "state_traits": [TRAIT_NAMES[index] for index in STATE_TRAIT_INDICES],
            "dynamic_traits": [TRAIT_NAMES[index] for index in DYNAMIC_TRAIT_INDICES],
            "state_value": "sum(q_frame * value_frame) / sum(q_frame)",
            "state_confidence": "sum(q_frame) / patch_frame_count",
            "dynamic_source": "finite differences between adjacent patch summaries",
            "dynamic_dt": "patch_center_s[t+1] - patch_center_s[t]",
            "frame_level_dynamic_channels": "zero_not_computed",
            "coverage": "valid_frame_count / patch_frame_count",
            "missing_value": 0.0,
            "missing_confidence": 0.0,
            "note": (
                "Trait values, confidence, and coverage are separate arrays. "
                "Values are never multiplied by confidence in the stored contract."
            ),
        },
        "outputs": {
            "csv": str(csv_path),
            "npz": str(npz_path),
            "summary_json": str(summary_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def run_edge_extraction(
    *,
    idtracker_csv: Path,
    matched_mousegpt_csv: Path,
    output_dir: Path,
    output_stem: str,
    clock_node_npz: Path,
    minimum_patch_displacement_px: float = 1.0,
    save_frame_level: bool = True,
) -> dict[str, Any]:
    inputs = load_frame_inputs(idtracker_csv, matched_mousegpt_csv)
    social_step, patch_start_s, patch_end_s, clock_meta = load_node_clock(
        clock_node_npz
    )
    result = extract_social_edges_on_clock(
        inputs,
        social_step=social_step,
        patch_start_s=patch_start_s,
        patch_end_s=patch_end_s,
        minimum_patch_displacement_px=minimum_patch_displacement_px,
        clock_parameters=clock_meta,
    )
    return write_edge_outputs(
        result,
        output_dir=output_dir,
        output_stem=output_stem,
        idtracker_csv=idtracker_csv,
        matched_mousegpt_csv=matched_mousegpt_csv,
        save_frame_level=save_frame_level,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract eight confidence-aware social edge traits on the fixed "
            "authoritative node patch grid."
        )
    )
    parser.add_argument("--idtracker-csv", required=True, type=Path)
    parser.add_argument("--matched-mousegpt-csv", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--output-stem", required=True)
    parser.add_argument(
        "--clock-node-npz",
        required=True,
        type=Path,
        help=(
            "Canonical node NPZ whose patch_start_s/patch_end_s arrays define "
            "the authoritative edge aggregation intervals."
        ),
    )
    parser.add_argument(
        "--minimum-patch-displacement-px",
        type=float,
        default=1.0,
        help=(
            "Motion-noise floor used to downweight patch displacement direction."
        ),
    )
    parser.add_argument("--no-frame-level", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_edge_extraction(
        idtracker_csv=args.idtracker_csv,
        matched_mousegpt_csv=args.matched_mousegpt_csv,
        output_dir=args.outdir,
        output_stem=args.output_stem,
        clock_node_npz=args.clock_node_npz,
        minimum_patch_displacement_px=args.minimum_patch_displacement_px,
        save_frame_level=not args.no_frame_level,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
