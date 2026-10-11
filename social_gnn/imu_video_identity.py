"""Diagnostic evidence for mapping IMU identities to video identities.

This module does not rewrite idtracker identities automatically. It produces
two independent evidence sources: gravity-compensated IMU motion and the
ordering of unmistakable entry/impact events.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class HorizontalAcceleration:
    gravity_xyz: np.ndarray
    dynamic_xyz: np.ndarray
    horizontal_xyz: np.ndarray
    horizontal_magnitude: np.ndarray


@dataclass(frozen=True)
class MotionIdentityEvidence:
    imu_to_video: tuple[int, ...]
    correlation_matrix: np.ndarray
    total_score: float
    assignment_margin: float


@dataclass(frozen=True)
class EntryOrderEvidence:
    imu_to_video: tuple[int, ...]
    imu_event_times_s: tuple[float, ...]
    video_event_times_s: tuple[float, ...]
    minimum_separation_s: float
    confidence: float


def _moving_average(values: np.ndarray, window_samples: int) -> np.ndarray:
    if window_samples <= 1:
        return values.copy()
    if window_samples % 2 == 0:
        window_samples += 1
    radius = window_samples // 2
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    kernel = np.ones(window_samples, dtype=float) / float(window_samples)
    return np.stack(
        [np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(values.shape[1])],
        axis=1,
    )


def estimate_horizontal_acceleration(
    acceleration_xyz: np.ndarray,
    time_s: np.ndarray,
    *,
    gravity_window_s: float = 0.8,
) -> HorizontalAcceleration:
    """Estimate acceleration in the plane orthogonal to low-pass gravity.

    Without orientation/magnetometer data this is not a world-coordinate
    horizontal acceleration. It is a gravity-orthogonal dynamic component,
    useful as a motion signature but requiring validation on real recordings.
    """

    acceleration_xyz = np.asarray(acceleration_xyz, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    if acceleration_xyz.ndim != 2 or acceleration_xyz.shape[1] != 3:
        raise ValueError("acceleration_xyz must have shape [T,3]")
    if time_s.ndim != 1 or len(time_s) != len(acceleration_xyz):
        raise ValueError("time_s must align with acceleration_xyz")
    if len(time_s) < 2 or not np.all(np.diff(time_s) > 0):
        raise ValueError("time_s must be strictly increasing with at least two samples")
    if gravity_window_s <= 0:
        raise ValueError("gravity_window_s must be positive")
    dt = float(np.median(np.diff(time_s)))
    window = max(1, int(round(gravity_window_s / dt)))
    gravity = _moving_average(acceleration_xyz, window)
    dynamic = acceleration_xyz - gravity
    gravity_norm = np.linalg.norm(gravity, axis=1, keepdims=True)
    gravity_unit = np.divide(
        gravity,
        gravity_norm,
        out=np.zeros_like(gravity),
        where=gravity_norm > 1e-8,
    )
    vertical = np.sum(dynamic * gravity_unit, axis=1, keepdims=True) * gravity_unit
    horizontal = dynamic - vertical
    magnitude = np.linalg.norm(horizontal, axis=1)
    invalid = ~np.isfinite(acceleration_xyz).all(axis=1) | ~np.isfinite(time_s)
    gravity[invalid] = np.nan
    dynamic[invalid] = np.nan
    horizontal[invalid] = np.nan
    magnitude[invalid] = np.nan
    return HorizontalAcceleration(gravity, dynamic, horizontal, magnitude)


def trajectory_motion_signature(
    centroids_xy: np.ndarray,
    time_s: np.ndarray,
    *,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Return pixel acceleration magnitude as a video motion signature."""

    positions = np.asarray(centroids_xy, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("centroids_xy must have shape [T,2]")
    if time_s.ndim != 1 or len(time_s) != len(positions) or len(time_s) < 3:
        raise ValueError("time_s must align with at least three centroids")
    if not np.all(np.diff(time_s) > 0):
        raise ValueError("time_s must be strictly increasing")
    velocity = np.gradient(positions, time_s, axis=0)
    acceleration = np.gradient(velocity, time_s, axis=0)
    signature = np.linalg.norm(acceleration, axis=1)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != (len(positions),):
            raise ValueError("valid must have shape [T]")
        signature[~valid] = np.nan
    return signature


def detect_impulse_event_times(
    signal: np.ndarray,
    time_s: np.ndarray,
    *,
    threshold_z: float = 6.0,
    refractory_s: float = 0.5,
    max_events: int | None = 2,
) -> np.ndarray:
    """Detect strong acceleration/impact-like events for entry diagnostics.

    This is intentionally conservative: it returns candidate event times and
    does not claim that every peak is an arena-entry event.
    """

    signal = np.asarray(signal, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    if signal.ndim != 1 or time_s.ndim != 1 or len(signal) != len(time_s):
        raise ValueError("signal and time_s must be aligned one-dimensional arrays")
    if len(signal) < 3 or not np.all(np.diff(time_s) > 0):
        raise ValueError("time_s must be increasing and contain at least three samples")
    if threshold_z <= 0 or refractory_s <= 0:
        raise ValueError("threshold_z and refractory_s must be positive")
    z = _robust_zscore(signal)
    candidates = np.flatnonzero(
        np.isfinite(z[1:-1])
        & (z[1:-1] >= threshold_z)
        & (z[1:-1] >= z[:-2])
        & (z[1:-1] >= z[2:])
    ) + 1
    ordered = candidates[np.argsort(z[candidates])[::-1]]
    selected: list[int] = []
    for index in ordered:
        if all(abs(float(time_s[index] - time_s[other])) >= refractory_s for other in selected):
            selected.append(int(index))
            if max_events is not None and len(selected) >= max_events:
                break
    return np.asarray(sorted(selected), dtype=int)


def first_consecutive_valid_time(
    time_s: np.ndarray,
    valid: np.ndarray,
    *,
    consecutive_samples: int = 5,
) -> float:
    """Return the first time with a sustained valid video/arena observation."""

    time_s = np.asarray(time_s, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if time_s.ndim != 1 or valid.shape != time_s.shape or len(time_s) == 0:
        raise ValueError("time_s and valid must be aligned one-dimensional arrays")
    if consecutive_samples < 1:
        raise ValueError("consecutive_samples must be >= 1")
    run = 0
    for index, is_valid in enumerate(valid):
        run = run + 1 if is_valid else 0
        if run >= consecutive_samples:
            return float(time_s[index - consecutive_samples + 1])
    return float("nan")


def _robust_zscore(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    median = np.nanmedian(signal)
    mad = np.nanmedian(np.abs(signal - median))
    scale = max(1.4826 * mad, 1e-8)
    return (signal - median) / scale


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return float("nan")
    aa = _robust_zscore(a[valid])
    bb = _robust_zscore(b[valid])
    denominator = np.linalg.norm(aa) * np.linalg.norm(bb)
    if denominator <= 1e-8:
        return 0.0
    return float(np.dot(aa, bb) / denominator)


def score_motion_identity_permutations(
    imu_signatures: np.ndarray,
    video_signatures: np.ndarray,
) -> MotionIdentityEvidence:
    """Score the best one-to-one IMU/video identity assignment.

    Both inputs must already be sampled on the same time grid. The returned
    tuple is indexed by IMU identity and contains its proposed video identity.
    """

    imu = np.asarray(imu_signatures, dtype=float)
    video = np.asarray(video_signatures, dtype=float)
    if imu.ndim != 2 or video.ndim != 2 or imu.shape != video.shape:
        raise ValueError("imu_signatures and video_signatures must share shape [T,N]")
    n = imu.shape[1]
    correlation = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            correlation[i, j] = _correlation(imu[:, i], video[:, j])
    finite = np.where(np.isfinite(correlation), correlation, -1.0)
    rows, cols = linear_sum_assignment(-finite)
    assignment = np.full(n, -1, dtype=int)
    assignment[rows] = cols
    best = float(finite[rows, cols].sum())
    alternatives = []
    for row, col in zip(rows, cols):
        alternative = finite.copy()
        alternative[row, col] = -np.inf
        alt_rows, alt_cols = linear_sum_assignment(-alternative)
        if len(alt_rows) == n and np.isfinite(alternative[alt_rows, alt_cols]).all():
            alternatives.append(float(alternative[alt_rows, alt_cols].sum()))
    second = max(alternatives) if alternatives else float("nan")
    margin = best - second if np.isfinite(second) else float("nan")
    return MotionIdentityEvidence(tuple(int(x) for x in assignment), correlation, best, margin)


def match_entry_order(
    imu_event_times_s: np.ndarray,
    video_event_times_s: np.ndarray,
    *,
    minimum_separation_s: float = 1.0,
) -> EntryOrderEvidence:
    """Map identities by entry order when two entry events are well separated."""

    imu = np.asarray(imu_event_times_s, dtype=float)
    video = np.asarray(video_event_times_s, dtype=float)
    if imu.ndim != 1 or video.ndim != 1 or len(imu) != len(video) or len(imu) < 1:
        raise ValueError("event time arrays must be non-empty and have equal length")
    if not np.isfinite(imu).all() or not np.isfinite(video).all():
        raise ValueError("event times must be finite")
    if minimum_separation_s <= 0:
        raise ValueError("minimum_separation_s must be positive")
    imu_order = np.argsort(imu)
    video_order = np.argsort(video)
    mapping = np.full(len(imu), -1, dtype=int)
    mapping[imu_order] = video_order
    imu_sep = float(np.min(np.diff(np.sort(imu)))) if len(imu) > 1 else float("inf")
    video_sep = float(np.min(np.diff(np.sort(video)))) if len(video) > 1 else float("inf")
    separation = min(imu_sep, video_sep)
    confidence = 1.0 if separation >= minimum_separation_s else separation / minimum_separation_s
    return EntryOrderEvidence(
        tuple(int(x) for x in mapping),
        tuple(float(x) for x in imu),
        tuple(float(x) for x in video),
        separation,
        float(np.clip(confidence, 0.0, 1.0)),
    )
