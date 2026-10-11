"""Anchor idtracker.ai identities to IMU identities by sequential arena entry.

The idtracker.ai identity numbers are arbitrary but internally useful.  This
module therefore never modifies an idtracker.ai session.  It derives a
post-hoc permutation from the interval in which only the first animal is in
the arena, writes a canonicalized kinematics CSV, and records the complete
evidence in a JSON manifest.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

try:  # Support module and direct-script execution.
    from .imu_video_identity import (
        detect_impulse_event_times,
        estimate_horizontal_acceleration,
    )
except ImportError:  # pragma: no cover - Windows pipeline direct execution
    from imu_video_identity import (  # type: ignore
        detect_impulse_event_times,
        estimate_horizontal_acceleration,
    )


class SequentialEntryIdentityError(ValueError):
    """Raised when the single-animal anchor is absent or ambiguous."""


@dataclass(frozen=True)
class IMUAnimalSource:
    label: str
    csv_path: Path
    entry_time_s: float | None = None


@dataclass(frozen=True)
class IMUEntryEvent:
    label: str
    time_s: float
    source: str
    imu_csv: str
    peak_robust_z: float | None


@dataclass(frozen=True)
class SequentialEntryMapping:
    trial_id: str
    entry_events: tuple[IMUEntryEvent, ...]
    anchor_start_s: float
    anchor_end_s: float
    anchor_frame_count: int
    source_to_canonical: dict[int, int]
    source_to_label: dict[int, str]
    coverage_by_source_identity: dict[int, float]
    single_valid_identity_fraction: float
    selected_source_identity: int
    competing_identity_coverage: float
    coverage_margin: float
    confidence_v0: float


def _fail(trial_id: str, message: str) -> SequentialEntryIdentityError:
    return SequentialEntryIdentityError(
        f"Sequential-entry identity mapping failed for trial {trial_id!r}: {message}"
    )


def _robust_zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    scale = max(1.4826 * float(mad), 1e-8)
    return (values - median) / scale


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


def detect_entry_event_from_imu(
    imu: pd.DataFrame,
    *,
    label: str,
    search_start_s: float = 0.0,
    search_end_s: float = 120.0,
    gravity_window_s: float = 0.8,
    threshold_z: float = 6.0,
) -> tuple[float, float]:
    """Return the strongest impact-like IMU event in an early search window.

    Acceleration is converted to a gravity-compensated dynamic magnitude.  The
    automatic result is deliberately exposed as evidence rather than assumed
    to be ground truth; callers may supply a visually confirmed entry time.
    """

    required = {"time", "ax", "ay", "az"}
    missing = sorted(required.difference(imu.columns))
    if missing:
        raise ValueError(f"IMU {label!r} lacks required columns: {missing}")
    time_s = pd.to_numeric(imu["time"], errors="coerce").to_numpy(float)
    acceleration = imu[["ax", "ay", "az"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    if (
        len(time_s) < 3
        or not np.isfinite(time_s).all()
        or not np.isfinite(acceleration).all()
        or np.any(np.diff(time_s) <= 0)
    ):
        raise ValueError(
            f"IMU {label!r} must contain finite, strictly increasing time and acceleration"
        )
    if not math.isfinite(search_start_s) or not math.isfinite(search_end_s):
        raise ValueError("entry search bounds must be finite")
    if search_end_s <= search_start_s:
        raise ValueError("entry_search_end_s must exceed entry_search_start_s")

    horizontal = estimate_horizontal_acceleration(
        acceleration, time_s, gravity_window_s=gravity_window_s
    )
    dynamic_magnitude = np.linalg.norm(horizontal.dynamic_xyz, axis=1)
    valid = (
        (time_s >= search_start_s)
        & (time_s < search_end_s)
        & np.isfinite(dynamic_magnitude)
    )
    if "interp_mask" in imu:
        valid &= ~_boolean(imu["interp_mask"])
    indices = np.flatnonzero(valid)
    if len(indices) < 3:
        raise ValueError(f"IMU {label!r} has fewer than three valid search-window samples")

    local_signal = dynamic_magnitude[indices]
    local_time = time_s[indices]
    candidates = detect_impulse_event_times(
        local_signal,
        local_time,
        threshold_z=threshold_z,
        refractory_s=0.5,
        max_events=1,
    )
    if len(candidates) != 1:
        raise ValueError(
            f"IMU {label!r} has no impact candidate above robust z={threshold_z:g} "
            f"inside [{search_start_s:g}, {search_end_s:g}) s"
        )
    local_index = int(candidates[0])
    robust_z = _robust_zscore(local_signal)
    return float(local_time[local_index]), float(robust_z[local_index])


def resolve_entry_events(
    animals: Sequence[IMUAnimalSource],
    *,
    search_start_s: float = 0.0,
    search_end_s: float = 120.0,
    gravity_window_s: float = 0.8,
    threshold_z: float = 6.0,
) -> tuple[IMUEntryEvent, ...]:
    """Resolve manual or automatically detected entry events for two animals."""

    if len(animals) != 2:
        raise ValueError("sequential-entry V0 requires exactly two IMU animals")
    labels = [animal.label.strip() for animal in animals]
    if any(not label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("IMU animal labels must be non-empty and unique")

    events: list[IMUEntryEvent] = []
    for animal, label in zip(animals, labels):
        path = animal.csv_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if animal.entry_time_s is not None:
            event_time = float(animal.entry_time_s)
            if not math.isfinite(event_time):
                raise ValueError(f"manual entry time for {label!r} must be finite")
            peak_z: float | None = None
            source = "manual_confirmed"
        else:
            imu = pd.read_csv(path)
            event_time, peak_z = detect_entry_event_from_imu(
                imu,
                label=label,
                search_start_s=search_start_s,
                search_end_s=search_end_s,
                gravity_window_s=gravity_window_s,
                threshold_z=threshold_z,
            )
            source = "automatic_dynamic_acceleration_peak"
        events.append(
            IMUEntryEvent(
                label=label,
                time_s=event_time,
                source=source,
                imu_csv=str(path),
                peak_robust_z=peak_z,
            )
        )
    return tuple(sorted(events, key=lambda event: event.time_s))


def infer_sequential_entry_mapping(
    idtracker: pd.DataFrame,
    entry_events: Sequence[IMUEntryEvent],
    *,
    trial_id: str,
    after_first_entry_margin_s: float = 0.5,
    before_second_entry_margin_s: float = 0.5,
    minimum_anchor_duration_s: float = 1.0,
    minimum_anchor_frames: int = 10,
    minimum_single_identity_fraction: float = 0.80,
    minimum_selected_coverage: float = 0.80,
    maximum_competing_coverage: float = 0.20,
    minimum_coverage_margin: float = 0.60,
) -> SequentialEntryMapping:
    """Infer the idtracker permutation from a stable one-animal interval."""

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
        raise _fail(trial_id, f"idtracker CSV lacks columns {missing}")
    if len(entry_events) != 2:
        raise _fail(trial_id, "V0 requires exactly two entry events")
    events = tuple(sorted(entry_events, key=lambda event: event.time_s))
    if not all(math.isfinite(event.time_s) for event in events):
        raise _fail(trial_id, "entry event times must be finite")
    if events[0].label == events[1].label:
        raise _fail(trial_id, "entry event labels must be unique")
    for value, name in (
        (after_first_entry_margin_s, "after_first_entry_margin_s"),
        (before_second_entry_margin_s, "before_second_entry_margin_s"),
        (minimum_anchor_duration_s, "minimum_anchor_duration_s"),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if minimum_anchor_frames < 1:
        raise ValueError("minimum_anchor_frames must be >= 1")
    for value, name in (
        (minimum_single_identity_fraction, "minimum_single_identity_fraction"),
        (minimum_selected_coverage, "minimum_selected_coverage"),
        (maximum_competing_coverage, "maximum_competing_coverage"),
        (minimum_coverage_margin, "minimum_coverage_margin"),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0,1]")

    data = idtracker.copy().reset_index(drop=True)
    for column in ("frame_id", "identity"):
        numeric = pd.to_numeric(data[column], errors="coerce").to_numpy(float)
        if not np.isfinite(numeric).all() or np.any(numeric != np.floor(numeric)):
            raise _fail(trial_id, f"{column} must contain finite integers")
        data[column] = numeric.astype(np.int64)
    if (data["frame_id"] < 0).any() or (data["identity"] < 1).any():
        raise _fail(trial_id, "frame IDs must be non-negative and identity IDs positive")
    data["time_s"] = pd.to_numeric(data["time_s"], errors="coerce")
    data["centroid_x_px"] = pd.to_numeric(
        data["centroid_x_px"], errors="coerce"
    )
    data["centroid_y_px"] = pd.to_numeric(
        data["centroid_y_px"], errors="coerce"
    )
    if data.duplicated(["frame_id", "identity"]).any():
        raise _fail(trial_id, "duplicate (frame_id, identity) rows")
    identities = tuple(sorted(int(value) for value in data["identity"].unique()))
    if len(identities) != 2:
        raise _fail(
            trial_id,
            f"expected exactly two idtracker identities, found {list(identities)}",
        )
    per_frame_time_count = data.groupby("frame_id")["time_s"].nunique(dropna=False)
    if (per_frame_time_count != 1).any() or not np.isfinite(data["time_s"]).all():
        raise _fail(trial_id, "identities disagree on finite time_s within a frame")
    frame_clock = data.groupby("frame_id")["time_s"].first().sort_index()
    frame_times = frame_clock.to_numpy(float)
    if len(frame_times) < 2 or np.any(np.diff(frame_times) <= 0):
        raise _fail(trial_id, "frame timestamps must increase with frame_id")
    if not (data.groupby("frame_id").size() == 2).all():
        raise _fail(trial_id, "each frame must retain both identity rows, including missing centroids")

    search_start = events[0].time_s + after_first_entry_margin_s
    search_end = events[1].time_s - before_second_entry_margin_s
    if search_end - search_start < minimum_anchor_duration_s:
        raise _fail(
            trial_id,
            f"safe single-animal interval [{search_start:.3f}, {search_end:.3f}) s "
            f"is shorter than {minimum_anchor_duration_s:.3f} s",
        )

    # Derive sampling from timestamps; do not turn a short surviving fragment
    # into 100% coverage of a much longer requested interval. The complete
    # between-entry search interval must be present, but identity evidence is
    # taken from its best short stable window. This fits rapid sequential entry:
    # we only need enough frames to identify the first animal, not several
    # seconds of flawless segmentation before the second animal enters.
    frame_step = float(np.median(np.diff(frame_times)))
    if frame_times[0] > search_start or frame_times[-1] + frame_step < search_end:
        raise _fail(trial_id, "tracking does not cover the complete anchor interval")

    in_search = (data["time_s"] >= search_start) & (data["time_s"] < search_end)
    search = data.loc[in_search].copy()
    search_clock = search.groupby("frame_id")["time_s"].first().sort_index()
    if (
        np.any(np.diff(search_clock.index.to_numpy()) != 1)
        or np.any(np.diff(search_clock.to_numpy()) > 1.5 * frame_step)
    ):
        raise _fail(trial_id, "missing frames or timestamp gaps inside anchor interval")

    candidate_starts = search_clock.to_numpy(float)
    latest_start = search_end - minimum_anchor_duration_s
    candidate_starts = candidate_starts[
        candidate_starts <= latest_start + 0.5 * frame_step
    ]
    candidate_starts = np.unique(
        np.concatenate(([search_start], candidate_starts, [latest_start]))
    )
    candidates: list[dict[str, Any]] = []
    for candidate_start in candidate_starts:
        candidate_end = candidate_start + minimum_anchor_duration_s
        in_window = (search["time_s"] >= candidate_start) & (
            search["time_s"] < candidate_end
        )
        window = search.loc[in_window].copy()
        frame_count = int(window["frame_id"].nunique())
        if frame_count < minimum_anchor_frames:
            continue

        centroid_valid = (
            _boolean(window["centroid_valid"])
            & np.isfinite(window["centroid_x_px"].to_numpy(float))
            & np.isfinite(window["centroid_y_px"].to_numpy(float))
        )
        window["_anchor_valid"] = centroid_valid
        validity = (
            window.pivot(index="frame_id", columns="identity", values="_anchor_valid")
            .reindex(columns=identities, fill_value=False)
            .fillna(False)
            .astype(bool)
        )
        valid_count = validity.sum(axis=1).to_numpy(int)
        single_fraction = float(np.mean(valid_count == 1))
        coverage = {
            int(identity): float(validity[identity].mean()) for identity in identities
        }
        ordered = sorted(coverage, key=coverage.get, reverse=True)
        selected = int(ordered[0])
        competitor = int(ordered[1])
        selected_coverage = coverage[selected]
        competitor_coverage = coverage[competitor]
        margin = selected_coverage - competitor_coverage

        failures: list[str] = []
        if single_fraction < minimum_single_identity_fraction:
            failures.append(
                f"single-valid-identity fraction {single_fraction:.3f} < "
                f"{minimum_single_identity_fraction:.3f}"
            )
        if selected_coverage < minimum_selected_coverage:
            failures.append(
                f"selected identity coverage {selected_coverage:.3f} < "
                f"{minimum_selected_coverage:.3f}"
            )
        if competitor_coverage > maximum_competing_coverage:
            failures.append(
                f"competing identity coverage {competitor_coverage:.3f} > "
                f"{maximum_competing_coverage:.3f}"
            )
        if margin < minimum_coverage_margin:
            failures.append(
                f"coverage margin {margin:.3f} < {minimum_coverage_margin:.3f}"
            )
        confidence = float(
            np.clip(
                min(
                    single_fraction,
                    selected_coverage,
                    1.0 - competitor_coverage,
                    max(0.0, margin),
                ),
                0.0,
                1.0,
            )
        )
        candidates.append(
            {
                "start": float(candidate_start),
                "end": float(candidate_end),
                "frame_count": frame_count,
                "coverage": coverage,
                "single_fraction": single_fraction,
                "selected": selected,
                "competitor": competitor,
                "selected_coverage": selected_coverage,
                "competitor_coverage": competitor_coverage,
                "margin": margin,
                "confidence": confidence,
                "failures": failures,
            }
        )

    if not candidates:
        raise _fail(
            trial_id,
            f"no {minimum_anchor_duration_s:.3f} s candidate anchor contains "
            f"at least {minimum_anchor_frames} frames",
        )

    valid_candidates = [candidate for candidate in candidates if not candidate["failures"]]
    if not valid_candidates:
        best = max(
            candidates,
            key=lambda item: (
                item["confidence"],
                item["margin"],
                item["single_fraction"],
                -item["start"],
            ),
        )
        raise _fail(
            trial_id,
            "; ".join(best["failures"])
            + f"; best coverage={best['coverage']}, candidate="
            f"[{best['start']:.3f}, {best['end']:.3f}), searched="
            f"[{search_start:.3f}, {search_end:.3f})",
        )

    selected_identities = {candidate["selected"] for candidate in valid_candidates}
    if len(selected_identities) != 1:
        raise _fail(
            trial_id,
            "different stable anchor windows select different idtracker identities: "
            f"{sorted(selected_identities)}",
        )
    best = max(
        valid_candidates,
        key=lambda item: (
            item["confidence"],
            item["margin"],
            item["single_fraction"],
            -item["start"],
        ),
    )
    anchor_start = best["start"]
    anchor_end = best["end"]
    frame_count = best["frame_count"]
    coverage = best["coverage"]
    single_fraction = best["single_fraction"]
    selected = best["selected"]
    competitor = best["competitor"]
    competitor_coverage = best["competitor_coverage"]
    margin = best["margin"]
    confidence = best["confidence"]

    source_to_canonical = {selected: 1, competitor: 2}
    source_to_label = {selected: events[0].label, competitor: events[1].label}
    return SequentialEntryMapping(
        trial_id=trial_id,
        entry_events=events,
        anchor_start_s=float(anchor_start),
        anchor_end_s=float(anchor_end),
        anchor_frame_count=frame_count,
        source_to_canonical=source_to_canonical,
        source_to_label=source_to_label,
        coverage_by_source_identity=coverage,
        single_valid_identity_fraction=single_fraction,
        selected_source_identity=selected,
        competing_identity_coverage=competitor_coverage,
        coverage_margin=margin,
        confidence_v0=confidence,
    )


def apply_sequential_entry_mapping(
    idtracker: pd.DataFrame, mapping: SequentialEntryMapping
) -> pd.DataFrame:
    """Return a canonicalized copy while retaining original idtracker IDs."""

    result = idtracker.copy(deep=True)
    source = pd.to_numeric(result["identity"], errors="raise").astype(int)
    unknown = sorted(set(int(value) for value in source.unique()) - mapping.source_to_canonical.keys())
    if unknown:
        raise _fail(mapping.trial_id, f"mapping lacks source identities {unknown}")
    result.insert(
        result.columns.get_loc("identity") + 1,
        "source_idtracker_identity",
        source,
    )
    result["identity"] = source.map(mapping.source_to_canonical).astype(int)
    result.insert(
        result.columns.get_loc("source_idtracker_identity") + 1,
        "canonical_identity_label",
        source.map(mapping.source_to_label),
    )
    result["identity_mapping_confidence_v0"] = mapping.confidence_v0
    result["identity_mapping_method"] = "sequential_entry_single_identity_v0"
    if result.duplicated(["frame_id", "identity"]).any():
        raise _fail(mapping.trial_id, "canonicalization created duplicate identities")
    return result


def _file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def run_sequential_entry_identity_mapping(
    *,
    idtracker_csv: Path,
    animals: Sequence[IMUAnimalSource],
    output_dir: Path,
    output_stem: str,
    search_start_s: float = 0.0,
    search_end_s: float = 120.0,
    gravity_window_s: float = 0.8,
    threshold_z: float = 6.0,
    after_first_entry_margin_s: float = 0.5,
    before_second_entry_margin_s: float = 0.5,
    minimum_anchor_duration_s: float = 1.0,
    minimum_anchor_frames: int = 10,
    minimum_single_identity_fraction: float = 0.80,
    minimum_selected_coverage: float = 0.80,
    maximum_competing_coverage: float = 0.20,
    minimum_coverage_margin: float = 0.60,
) -> dict[str, Any]:
    """Run mapping, write canonical CSV/mapping CSV/JSON, and return manifest."""

    idtracker_csv = idtracker_csv.resolve()
    if not idtracker_csv.is_file():
        raise FileNotFoundError(idtracker_csv)
    events = resolve_entry_events(
        animals,
        search_start_s=search_start_s,
        search_end_s=search_end_s,
        gravity_window_s=gravity_window_s,
        threshold_z=threshold_z,
    )
    idtracker = pd.read_csv(idtracker_csv)
    mapping = infer_sequential_entry_mapping(
        idtracker,
        events,
        trial_id=output_stem,
        after_first_entry_margin_s=after_first_entry_margin_s,
        before_second_entry_margin_s=before_second_entry_margin_s,
        minimum_anchor_duration_s=minimum_anchor_duration_s,
        minimum_anchor_frames=minimum_anchor_frames,
        minimum_single_identity_fraction=minimum_single_identity_fraction,
        minimum_selected_coverage=minimum_selected_coverage,
        maximum_competing_coverage=maximum_competing_coverage,
        minimum_coverage_margin=minimum_coverage_margin,
    )
    canonical = apply_sequential_entry_mapping(idtracker, mapping)
    canonical_csv = output_dir / f"{output_stem}_idtracker_kinematics_canonical.csv"
    mapping_csv = output_dir / f"{output_stem}_imu_video_identity_mapping.csv"
    summary_json = output_dir / f"{output_stem}_imu_video_identity_mapping.json"
    input_paths = {idtracker_csv, *(animal.csv_path.resolve() for animal in animals)}
    if any(path.resolve() in input_paths for path in (canonical_csv, mapping_csv, summary_json)):
        raise _fail(output_stem, "output paths must not overwrite input files")
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(canonical_csv, index=False, encoding="utf-8-sig")

    rows = []
    for source_identity, canonical_identity in sorted(
        mapping.source_to_canonical.items(), key=lambda item: item[1]
    ):
        rows.append(
            {
                "canonical_identity": canonical_identity,
                "canonical_identity_label": mapping.source_to_label[source_identity],
                "source_idtracker_identity": source_identity,
                "entry_time_s": next(
                    event.time_s
                    for event in mapping.entry_events
                    if event.label == mapping.source_to_label[source_identity]
                ),
                "anchor_coverage": mapping.coverage_by_source_identity[source_identity],
                "mapping_confidence_v0": mapping.confidence_v0,
            }
        )
    pd.DataFrame(rows).to_csv(mapping_csv, index=False, encoding="utf-8-sig")

    parameters = {
        "entry_search_start_s": search_start_s,
        "entry_search_end_s": search_end_s,
        "gravity_window_s": gravity_window_s,
        "threshold_z": threshold_z,
        "after_first_entry_margin_s": after_first_entry_margin_s,
        "before_second_entry_margin_s": before_second_entry_margin_s,
        "minimum_anchor_duration_s": minimum_anchor_duration_s,
        "minimum_anchor_frames": minimum_anchor_frames,
        "minimum_single_identity_fraction": minimum_single_identity_fraction,
        "minimum_selected_coverage": minimum_selected_coverage,
        "maximum_competing_coverage": maximum_competing_coverage,
        "minimum_coverage_margin": minimum_coverage_margin,
    }
    manifest: dict[str, Any] = {
        "contract_version": "sequential_entry_identity_v0",
        "trial_id": output_stem,
        "method": "sequential_entry_single_identity_v0",
        "idtracker_csv": str(idtracker_csv),
        "entry_events": [asdict(event) for event in mapping.entry_events],
        "anchor_interval_s": [mapping.anchor_start_s, mapping.anchor_end_s],
        "anchor_frame_count": mapping.anchor_frame_count,
        "source_to_canonical": {
            str(key): value for key, value in mapping.source_to_canonical.items()
        },
        "source_to_label": {
            str(key): value for key, value in mapping.source_to_label.items()
        },
        "canonical_identity_order": [1, 2],
        "canonical_label_order": [event.label for event in mapping.entry_events],
        "observation_basis": "finite valid centroid tracks; raw blob count is not supplied",
        "coverage_by_source_identity": {
            str(key): value
            for key, value in mapping.coverage_by_source_identity.items()
        },
        "single_valid_identity_fraction": mapping.single_valid_identity_fraction,
        "selected_source_identity": mapping.selected_source_identity,
        "competing_identity_coverage": mapping.competing_identity_coverage,
        "coverage_margin": mapping.coverage_margin,
        "confidence_v0": mapping.confidence_v0,
        "confidence_note": (
            "V0 confidence is a conservative anchor-window quality score, not a "
            "calibrated probability. Ambiguous windows raise instead of being guessed."
        ),
        "parameters": parameters,
        "input_files": [
            _file_signature(idtracker_csv),
            *[_file_signature(animal.csv_path.resolve()) for animal in animals],
        ],
        "outputs": {
            "canonical_idtracker_csv": str(canonical_csv),
            "mapping_csv": str(mapping_csv),
            "summary_json": str(summary_json),
        },
    }
    summary_json.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def _parse_key_value(values: Iterable[str], *, option: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} expects LABEL=VALUE, got {value!r}")
        label, item = value.split("=", 1)
        label = label.strip()
        item = item.strip()
        if not label or not item:
            raise ValueError(f"{option} expects non-empty LABEL=VALUE")
        if label in parsed:
            raise ValueError(f"duplicate {option} label {label!r}")
        parsed[label] = item
    return parsed


def _animals_from_args(
    animal_values: Sequence[str], entry_time_values: Sequence[str]
) -> tuple[IMUAnimalSource, ...]:
    paths = _parse_key_value(animal_values, option="--animal")
    entry_times = _parse_key_value(entry_time_values, option="--entry-time")
    unknown = sorted(set(entry_times) - set(paths))
    if unknown:
        raise ValueError(f"--entry-time labels missing from --animal: {unknown}")
    return tuple(
        IMUAnimalSource(
            label=label,
            csv_path=Path(path),
            entry_time_s=float(entry_times[label]) if label in entry_times else None,
        )
        for label, path in paths.items()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map arbitrary idtracker.ai IDs to canonical IMU identities using the "
            "single-animal interval between two sequential arena entries."
        )
    )
    parser.add_argument("--idtracker-csv", required=True, type=Path)
    parser.add_argument(
        "--animal",
        action="append",
        required=True,
        metavar="LABEL=IMU_CSV",
        help="Repeat exactly twice.",
    )
    parser.add_argument(
        "--entry-time",
        action="append",
        default=[],
        metavar="LABEL=SECONDS",
        help="Optional visually confirmed override; otherwise detect from IMU.",
    )
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--output-stem", required=True)
    parser.add_argument("--entry-search-start-s", type=float, default=0.0)
    parser.add_argument("--entry-search-end-s", type=float, default=120.0)
    parser.add_argument("--gravity-window-s", type=float, default=0.8)
    parser.add_argument("--threshold-z", type=float, default=6.0)
    parser.add_argument("--after-first-entry-margin-s", type=float, default=0.5)
    parser.add_argument("--before-second-entry-margin-s", type=float, default=0.5)
    parser.add_argument("--minimum-anchor-duration-s", type=float, default=1.0)
    parser.add_argument("--minimum-anchor-frames", type=int, default=10)
    parser.add_argument("--minimum-single-identity-fraction", type=float, default=0.80)
    parser.add_argument("--minimum-selected-coverage", type=float, default=0.80)
    parser.add_argument("--maximum-competing-coverage", type=float, default=0.20)
    parser.add_argument("--minimum-coverage-margin", type=float, default=0.60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    animals = _animals_from_args(args.animal, args.entry_time)
    manifest = run_sequential_entry_identity_mapping(
        idtracker_csv=args.idtracker_csv,
        animals=animals,
        output_dir=args.outdir,
        output_stem=args.output_stem,
        search_start_s=args.entry_search_start_s,
        search_end_s=args.entry_search_end_s,
        gravity_window_s=args.gravity_window_s,
        threshold_z=args.threshold_z,
        after_first_entry_margin_s=args.after_first_entry_margin_s,
        before_second_entry_margin_s=args.before_second_entry_margin_s,
        minimum_anchor_duration_s=args.minimum_anchor_duration_s,
        minimum_anchor_frames=args.minimum_anchor_frames,
        minimum_single_identity_fraction=args.minimum_single_identity_fraction,
        minimum_selected_coverage=args.minimum_selected_coverage,
        maximum_competing_coverage=args.maximum_competing_coverage,
        minimum_coverage_margin=args.minimum_coverage_margin,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
