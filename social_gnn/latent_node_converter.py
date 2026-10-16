"""Aggregate aligned per-sample IMU latents onto the fixed social clock.

The mature upstream latent files contain one ``downstream`` vector every
approximately 0.04 seconds.  Social-GNN uses one graph every 0.16 seconds, so
this converter computes one validity-weighted node vector per animal and
social patch without changing the stored upstream files.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from .authoritative_clock import load_node_clock, validate_fixed_patch_grid
except ImportError:  # pragma: no cover - direct script execution
    from authoritative_clock import load_node_clock, validate_fixed_patch_grid


NODE_CONTRACT_VERSION = "social_node_features_v0"
DEFAULT_PATCH_DURATION_S = 0.16
DEFAULT_PATCH_STRIDE_S = 0.16


class LatentNodeConversionError(ValueError):
    """Raised when aligned latent inputs cannot form a canonical node NPZ."""


@dataclass(frozen=True)
class LatentAnimalSource:
    """One named animal and its mature upstream latent NPZ."""

    identity: str
    path: Path


@dataclass(frozen=True)
class _LoadedLatent:
    identity: str
    path: Path
    values: np.ndarray
    time_s: np.ndarray
    interpolated: np.ndarray


def _fail(trial_id: str, message: str) -> LatentNodeConversionError:
    return LatentNodeConversionError(f"Trial {trial_id!r}: {message}")


def _load_source(source: LatentAnimalSource, trial_id: str) -> _LoadedLatent:
    path = Path(source.path)
    try:
        with np.load(path, allow_pickle=False) as payload:
            missing = {"downstream", "time", "interp_mask"} - set(payload.files)
            if missing:
                raise _fail(
                    trial_id,
                    f"latent {path} is missing arrays {sorted(missing)}",
                )
            values = np.array(payload["downstream"], dtype=np.float32, copy=True)
            time_s = np.array(payload["time"], dtype=np.float64, copy=True)
            raw_interp = np.array(payload["interp_mask"], copy=True)
    except LatentNodeConversionError:
        raise
    except (OSError, ValueError) as exc:
        raise _fail(trial_id, f"could not read latent {path}: {exc}") from exc

    if not str(source.identity):
        raise _fail(trial_id, f"latent {path} has an empty identity")
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise _fail(
            trial_id,
            f"latent {path} downstream must have non-empty shape [S,D], "
            f"got {values.shape}",
        )
    if time_s.shape != (values.shape[0],):
        raise _fail(
            trial_id,
            f"latent {path} time shape {time_s.shape} does not match "
            f"downstream length {values.shape[0]}",
        )
    if raw_interp.shape != (values.shape[0],):
        raise _fail(
            trial_id,
            f"latent {path} interp_mask shape {raw_interp.shape} does not match "
            f"downstream length {values.shape[0]}",
        )
    if not np.isfinite(values).all() or not np.isfinite(time_s).all():
        raise _fail(trial_id, f"latent {path} contains non-finite values or times")
    if values.shape[0] < 2 or np.any(np.diff(time_s) <= 0):
        raise _fail(trial_id, f"latent {path} time must be strictly increasing")

    if raw_interp.dtype == np.bool_:
        interpolated = raw_interp.astype(bool, copy=False)
    else:
        try:
            numeric_interp = np.asarray(raw_interp, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise _fail(trial_id, f"latent {path} interp_mask is not binary") from exc
        if not np.isfinite(numeric_interp).all() or not np.isin(
            numeric_interp, [0.0, 1.0]
        ).all():
            raise _fail(trial_id, f"latent {path} interp_mask must contain only 0/1")
        interpolated = numeric_interp.astype(bool)

    return _LoadedLatent(
        identity=str(source.identity),
        path=path,
        values=values,
        time_s=time_s,
        interpolated=interpolated,
    )


def _validate_sampling(time_s: np.ndarray, trial_id: str) -> float:
    differences = np.diff(time_s)
    sample_period_s = float(np.median(differences))
    if not np.isfinite(sample_period_s) or sample_period_s <= 0:
        raise _fail(trial_id, "could not derive a positive latent sample period")
    if not np.allclose(
        differences, sample_period_s, rtol=1e-6, atol=1e-9
    ):
        index = int(
            np.flatnonzero(
                ~np.isclose(
                    differences, sample_period_s, rtol=1e-6, atol=1e-9
                )
            )[0]
            + 1
        )
        raise _fail(
            trial_id,
            f"latent timestamp gap or variable sample period at sample {index}",
        )
    return sample_period_s


def _align_common_timebase(
    loaded: Sequence[_LoadedLatent], trial_id: str
) -> tuple[list[_LoadedLatent], np.ndarray, float, dict[str, Any]]:
    """Trim boundary-only differences and validate one shared sampled timebase."""
    sample_periods = [
        _validate_sampling(item.time_s, trial_id) for item in loaded
    ]
    reference_period = float(np.median(sample_periods))
    timestamp_tolerance_s = max(1e-6, reference_period * 1e-5)
    if not np.allclose(
        sample_periods,
        reference_period,
        rtol=1e-6,
        atol=timestamp_tolerance_s,
    ):
        raise _fail(
            trial_id,
            f"latent sample periods disagree: {sample_periods}",
        )

    overlap_start_s = max(float(item.time_s[0]) for item in loaded)
    overlap_end_s = min(float(item.time_s[-1]) for item in loaded)
    if overlap_end_s < overlap_start_s - timestamp_tolerance_s:
        raise _fail(trial_id, "animal latent streams have no common time range")

    sliced: list[_LoadedLatent] = []
    dropped_prefix: list[int] = []
    dropped_suffix: list[int] = []
    for item in loaded:
        left = int(
            np.searchsorted(
                item.time_s, overlap_start_s - timestamp_tolerance_s, side="left"
            )
        )
        right = int(
            np.searchsorted(
                item.time_s, overlap_end_s + timestamp_tolerance_s, side="right"
            )
        )
        dropped_prefix.append(left)
        dropped_suffix.append(len(item.time_s) - right)
        sliced.append(
            _LoadedLatent(
                identity=item.identity,
                path=item.path,
                values=item.values[left:right],
                time_s=item.time_s[left:right],
                interpolated=item.interpolated[left:right],
            )
        )

    lengths = [len(item.time_s) for item in sliced]
    if len(set(lengths)) != 1:
        raise _fail(
            trial_id,
            "latent streams do not reduce to the same sampled boundary overlap; "
            f"aligned lengths are {lengths}",
        )
    time_matrix = np.stack([item.time_s for item in sliced], axis=0)
    spread = np.ptp(time_matrix, axis=0)
    invalid = np.flatnonzero(spread > timestamp_tolerance_s)
    if invalid.size:
        index = int(invalid[0])
        raise _fail(
            trial_id,
            f"animal timestamps are misaligned at common sample {index}; "
            f"spread {spread[index]} s exceeds tolerance "
            f"{timestamp_tolerance_s} s",
        )

    # Tiny exporter-specific floating offsets have no physical significance.
    # Use their mean as one shared authoritative time for all animals.
    common_time = time_matrix.mean(axis=0, dtype=np.float64)
    aligned = [
        _LoadedLatent(
            identity=item.identity,
            path=item.path,
            values=item.values,
            time_s=common_time,
            interpolated=item.interpolated,
        )
        for item in sliced
    ]
    alignment = {
        "timestamp_tolerance_s": timestamp_tolerance_s,
        "maximum_timestamp_spread_s": float(spread.max(initial=0.0)),
        "boundary_samples_dropped_prefix": dropped_prefix,
        "boundary_samples_dropped_suffix": dropped_suffix,
    }
    return aligned, common_time, reference_period, alignment


def _fixed_clock(
    *,
    first_sample_s: float,
    last_sample_s: float,
    sample_period_s: float,
    patch_duration_s: float,
    patch_stride_s: float,
    clock_start_s: float | None,
    clock_end_s: float | None,
    trial_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    for name, value in (
        ("patch_duration_s", patch_duration_s),
        ("patch_stride_s", patch_stride_s),
    ):
        if not np.isfinite(value) or value <= 0:
            raise _fail(trial_id, f"{name} must be positive and finite")

    source_end_s = float(last_sample_s + sample_period_s)
    start_s = float(first_sample_s if clock_start_s is None else clock_start_s)
    end_s = float(source_end_s if clock_end_s is None else clock_end_s)
    if not np.isfinite(start_s) or not np.isfinite(end_s) or end_s <= start_s:
        raise _fail(trial_id, "clock bounds must be finite with end greater than start")
    tolerance = max(1e-9, sample_period_s * 1e-6)
    if start_s < first_sample_s - tolerance:
        raise _fail(
            trial_id,
            f"clock_start_s {start_s} precedes first latent time {first_sample_s}",
        )
    if end_s > source_end_s + tolerance:
        raise _fail(
            trial_id,
            f"clock_end_s {end_s} exceeds latent support end {source_end_s}",
        )

    available = end_s - start_s
    if available + tolerance < patch_duration_s:
        raise _fail(trial_id, "clock bounds contain no complete social patch")
    patch_count = int(
        np.floor((available - patch_duration_s + tolerance) / patch_stride_s) + 1
    )
    starts = start_s + np.arange(patch_count, dtype=np.float64) * patch_stride_s
    ends = starts + patch_duration_s
    try:
        validate_fixed_patch_grid(
            starts,
            ends,
            social_step=np.arange(patch_count, dtype=np.int64),
            label="generated latent social clock",
        )
    except ValueError as exc:
        raise _fail(trial_id, str(exc)) from exc
    return starts, ends


def _aggregate_one_animal(
    loaded: _LoadedLatent,
    patch_start_s: np.ndarray,
    patch_end_s: np.ndarray,
    *,
    expected_samples: int,
    interpolated_weight: float,
    minimum_weight_coverage: float,
    timestamp_tolerance_s: float,
    sample_period_s: float,
    permit_outside_source_support: bool,
    trial_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    patch_count = len(patch_start_s)
    feature_dim = loaded.values.shape[1]
    features = np.zeros((patch_count, feature_dim), dtype=np.float32)
    node_mask = np.zeros(patch_count, dtype=bool)
    weight_coverage = np.zeros(patch_count, dtype=np.float32)
    interpolation_fraction = np.zeros(patch_count, dtype=np.float32)
    sample_count = np.zeros(patch_count, dtype=np.int32)

    for patch in range(patch_count):
        # Treat samples within the validated sub-sample timestamp tolerance as
        # lying exactly on a patch boundary. This preserves half-open interval
        # semantics despite exporter floating drift at values such as 0.16.
        left = int(
            np.searchsorted(
                loaded.time_s,
                patch_start_s[patch] - timestamp_tolerance_s,
                side="left",
            )
        )
        right = int(
            np.searchsorted(
                loaded.time_s,
                patch_end_s[patch] - timestamp_tolerance_s,
                side="left",
            )
        )
        count = right - left
        sample_count[patch] = count
        if count != expected_samples:
            source_start_s = float(loaded.time_s[0])
            source_end_s = float(loaded.time_s[-1] + sample_period_s)
            fully_inside_source_support = (
                patch_start_s[patch] >= source_start_s - timestamp_tolerance_s
                and patch_end_s[patch] <= source_end_s + timestamp_tolerance_s
            )
            if permit_outside_source_support and not fully_inside_source_support:
                # An authoritative trial clock can legitimately extend beyond a
                # newly delivered/cropped latent stream.  Keep that clock row,
                # but make the missing node target explicit through node_mask.
                continue
            raise _fail(
                trial_id,
                f"identity {loaded.identity!r} patch {patch} "
                f"[{patch_start_s[patch]}, {patch_end_s[patch]}) contains "
                f"{count} latent samples; expected {expected_samples}",
            )

        interp = loaded.interpolated[left:right]
        weights = np.where(interp, interpolated_weight, 1.0).astype(np.float64)
        weight_sum = float(weights.sum())
        coverage = weight_sum / expected_samples
        weight_coverage[patch] = coverage
        interpolation_fraction[patch] = float(interp.mean())
        valid = weight_sum > 0 and coverage >= minimum_weight_coverage
        node_mask[patch] = valid
        if valid:
            features[patch] = np.average(
                loaded.values[left:right].astype(np.float64),
                axis=0,
                weights=weights,
            ).astype(np.float32)

    return (
        features,
        node_mask,
        weight_coverage,
        interpolation_fraction,
        sample_count,
    )


def convert_aligned_latents_to_node_npz(
    animal_sources: Sequence[LatentAnimalSource | tuple[str, str | Path]],
    output_npz: str | Path,
    *,
    trial_id: str,
    output_identity: Sequence[Any] | None = None,
    patch_duration_s: float = DEFAULT_PATCH_DURATION_S,
    patch_stride_s: float = DEFAULT_PATCH_STRIDE_S,
    clock_start_s: float | None = None,
    clock_end_s: float | None = None,
    authoritative_clock_npz: str | Path | None = None,
    interpolated_weight: float = 0.0,
    minimum_weight_coverage: float = 0.5,
    allow_duplicate_nodes: bool = False,
    summary_json: str | Path | None = None,
) -> dict[str, Any]:
    """Create patch-level node features from timestamp-aligned latent streams.

    Samples with ``interp_mask == 0`` have weight 1.  Interpolated samples use
    ``interpolated_weight``.  A patch is valid only when the sum of these
    weights divided by the expected sample count reaches
    ``minimum_weight_coverage``.  Invalid patch values are stored as zero and
    marked false in ``node_mask``.
    """
    resolved_id = str(trial_id).strip()
    if not resolved_id:
        raise LatentNodeConversionError("trial_id must not be empty")
    if len(animal_sources) < 2:
        raise _fail(resolved_id, "at least two animal latent sources are required")
    if not 0.0 <= interpolated_weight <= 1.0:
        raise _fail(resolved_id, "interpolated_weight must be in [0,1]")
    if not 0.0 <= minimum_weight_coverage <= 1.0:
        raise _fail(resolved_id, "minimum_weight_coverage must be in [0,1]")

    normalized_sources = [
        source
        if isinstance(source, LatentAnimalSource)
        else LatentAnimalSource(str(source[0]), Path(source[1]))
        for source in animal_sources
    ]
    source_identities = [source.identity for source in normalized_sources]
    if len(set(source_identities)) != len(source_identities):
        raise _fail(resolved_id, f"animal identities must be unique: {source_identities}")

    loaded = [_load_source(source, resolved_id) for source in normalized_sources]
    feature_dim = loaded[0].values.shape[1]
    for item in loaded[1:]:
        if item.values.shape[1] != feature_dim:
            raise _fail(
                resolved_id,
                f"identity {item.identity!r} latent dimension {item.values.shape[1]} "
                f"does not match {feature_dim}",
            )
    loaded, reference_time, sample_period_s, alignment_qc = _align_common_timebase(
        loaded, resolved_id
    )
    reference_clock_meta: dict[str, Any] | None = None
    reference_identity: np.ndarray | None = None
    social_step: np.ndarray
    if authoritative_clock_npz is not None:
        if clock_start_s is not None or clock_end_s is not None:
            raise _fail(
                resolved_id,
                "clock_start_s/clock_end_s cannot be combined with "
                "authoritative_clock_npz",
            )
        try:
            social_step, patch_start_s, patch_end_s, reference_clock_meta = (
                load_node_clock(authoritative_clock_npz)
            )
        except ValueError as exc:
            raise _fail(resolved_id, str(exc)) from exc
        if reference_clock_meta["trial_id"] != resolved_id:
            raise _fail(
                resolved_id,
                "authoritative clock trial_id "
                f"{reference_clock_meta['trial_id']!r} does not match",
            )
        patch_duration_s = float(reference_clock_meta["patch_duration_s"])
        patch_stride_meta = reference_clock_meta["patch_stride_s"]
        patch_stride_s = (
            float(patch_stride_meta)
            if patch_stride_meta is not None
            else float(patch_duration_s)
        )
        try:
            with np.load(authoritative_clock_npz, allow_pickle=False) as reference:
                if "identity" in reference.files:
                    reference_identity = np.array(reference["identity"], copy=True)
                if "source_identity" in reference.files:
                    expected_source_identity = [
                        str(item) for item in np.asarray(reference["source_identity"])
                    ]
                    if expected_source_identity != source_identities:
                        raise _fail(
                            resolved_id,
                            "latent source order does not match authoritative "
                            f"source_identity: got {source_identities}, expected "
                            f"{expected_source_identity}",
                        )
        except LatentNodeConversionError:
            raise
        except (OSError, ValueError) as exc:
            raise _fail(
                resolved_id,
                f"could not inspect authoritative clock identity: {exc}",
            ) from exc
    else:
        patch_start_s, patch_end_s = _fixed_clock(
            first_sample_s=float(reference_time[0]),
            last_sample_s=float(reference_time[-1]),
            sample_period_s=sample_period_s,
            patch_duration_s=float(patch_duration_s),
            patch_stride_s=float(patch_stride_s),
            clock_start_s=clock_start_s,
            clock_end_s=clock_end_s,
            trial_id=resolved_id,
        )
        social_step = np.arange(len(patch_start_s), dtype=np.int64)

    samples_per_patch_float = patch_duration_s / sample_period_s
    expected_samples = int(round(samples_per_patch_float))
    if expected_samples < 1 or not np.isclose(
        samples_per_patch_float, expected_samples, rtol=1e-6, atol=1e-8
    ):
        raise _fail(
            resolved_id,
            f"patch duration {patch_duration_s} is not an integer multiple of "
            f"latent sample period {sample_period_s}",
        )

    timestamp_tolerance_s = float(alignment_qc["timestamp_tolerance_s"])
    first_used = int(
        np.searchsorted(
            reference_time,
            patch_start_s[0] - timestamp_tolerance_s,
            side="left",
        )
    )
    after_last_used = int(
        np.searchsorted(
            reference_time,
            patch_end_s[-1] - timestamp_tolerance_s,
            side="left",
        )
    )
    alignment_qc["aligned_source_sample_count"] = int(len(reference_time))
    alignment_qc["samples_before_clock"] = first_used
    alignment_qc["samples_after_last_complete_patch"] = int(
        len(reference_time) - after_last_used
    )

    node_features = []
    node_masks = []
    node_coverages = []
    interp_fractions = []
    sample_counts = []
    for item in loaded:
        features, mask, coverage, interp_fraction, count = _aggregate_one_animal(
            item,
            patch_start_s,
            patch_end_s,
            expected_samples=expected_samples,
            interpolated_weight=interpolated_weight,
            minimum_weight_coverage=minimum_weight_coverage,
            timestamp_tolerance_s=timestamp_tolerance_s,
            sample_period_s=sample_period_s,
            permit_outside_source_support=authoritative_clock_npz is not None,
            trial_id=resolved_id,
        )
        node_features.append(features)
        node_masks.append(mask)
        node_coverages.append(coverage)
        interp_fractions.append(interp_fraction)
        sample_counts.append(count)

    node_feature_array = np.stack(node_features, axis=1)
    node_mask_array = np.stack(node_masks, axis=1)
    node_coverage_array = np.stack(node_coverages, axis=1)
    interp_fraction_array = np.stack(interp_fractions, axis=1)
    sample_count_array = np.stack(sample_counts, axis=1)

    duplicate_pairs: list[list[str]] = []
    for left in range(len(loaded)):
        for right in range(left + 1, len(loaded)):
            jointly_valid = node_mask_array[:, left] & node_mask_array[:, right]
            if jointly_valid.any() and np.array_equal(
                node_feature_array[jointly_valid, left],
                node_feature_array[jointly_valid, right],
            ):
                duplicate_pairs.append(
                    [loaded[left].identity, loaded[right].identity]
                )
    if duplicate_pairs and not allow_duplicate_nodes:
        raise _fail(
            resolved_id,
            f"exact duplicate aggregated node streams {duplicate_pairs}; possible "
            "upstream recombination failure. Use allow_duplicate_nodes only for "
            "interface testing.",
        )

    labels = (
        list(output_identity)
        if output_identity is not None
        else (
            reference_identity.tolist()
            if reference_identity is not None
            else source_identities
        )
    )
    if len(labels) != len(loaded) or len({str(item) for item in labels}) != len(labels):
        raise _fail(
            resolved_id,
            f"output_identity must contain {len(loaded)} unique labels",
        )
    identity_array = np.asarray(labels)
    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        contract_version=np.asarray(NODE_CONTRACT_VERSION),
        trial_id=np.asarray(resolved_id),
        node_features=node_feature_array,
        node_mask=node_mask_array,
        patch_start_s=patch_start_s,
        patch_end_s=patch_end_s,
        identity=identity_array,
        source_identity=np.asarray(source_identities),
        social_step=social_step,
        node_weight_coverage=node_coverage_array,
        node_interp_fraction=interp_fraction_array,
        node_sample_count=sample_count_array,
        latent_sample_period_s=np.asarray(sample_period_s),
        source_latent_file=np.asarray([str(item.path.resolve()) for item in loaded]),
        interpolated_weight=np.asarray(interpolated_weight),
        minimum_weight_coverage=np.asarray(minimum_weight_coverage),
        training_eligible=np.asarray(not duplicate_pairs),
        authoritative_clock_npz=np.asarray(
            ""
            if authoritative_clock_npz is None
            else str(Path(authoritative_clock_npz).resolve())
        ),
    )

    summary_path = (
        Path(summary_json)
        if summary_json is not None
        else output_npz.with_name(f"{output_npz.stem}_summary.json")
    )
    summary = {
        "contract_version": NODE_CONTRACT_VERSION,
        "trial_id": resolved_id,
        "output_node_npz": str(output_npz.resolve()),
        "source_identity": source_identities,
        "identity": [
            item.item() if isinstance(item, np.generic) else item
            for item in identity_array
        ],
        "source_latent_file": [str(item.path.resolve()) for item in loaded],
        "shape": list(node_feature_array.shape),
        "clock": {
            "mode": (
                "authoritative_node_intervals"
                if authoritative_clock_npz is not None
                else "generated_from_latent_support"
            ),
            "source_node_npz": (
                None
                if authoritative_clock_npz is None
                else str(Path(authoritative_clock_npz).resolve())
            ),
            "source_clock_sha256": (
                None
                if reference_clock_meta is None
                else reference_clock_meta["clock_sha256"]
            ),
            "patch_count": int(len(patch_start_s)),
            "patch_start_s": float(patch_start_s[0]),
            "patch_end_s": float(patch_end_s[-1]),
            "patch_duration_s": float(patch_duration_s),
            "patch_stride_s": float(patch_stride_s),
        },
        "aggregation": {
            "method": "validity_weighted_mean",
            "latent_sample_period_s": sample_period_s,
            "expected_samples_per_patch": expected_samples,
            "interpolated_weight": float(interpolated_weight),
            "minimum_weight_coverage": float(minimum_weight_coverage),
        },
        "time_alignment": alignment_qc,
        "qc": {
            "training_eligible": not duplicate_pairs,
            "duplicate_node_pairs": duplicate_pairs,
            "valid_patch_count_by_identity": node_mask_array.sum(axis=0)
            .astype(int)
            .tolist(),
            "mean_weight_coverage_by_identity": node_coverage_array.mean(axis=0)
            .astype(float)
            .tolist(),
            "missing_patch_count_by_identity": (
                sample_count_array != expected_samples
            ).sum(axis=0).astype(int).tolist(),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary["summary_json"] = str(summary_path.resolve())
    return summary


def _parse_animal(value: str) -> LatentAnimalSource:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--animal must use ID=PATH")
    identity, path = value.split("=", 1)
    if not identity.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--animal must use non-empty ID=PATH")
    return LatentAnimalSource(identity.strip(), Path(path.strip()))


def _identity_label(value: str) -> Any:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    if isinstance(parsed, (str, int, float, bool)) or parsed is None:
        return parsed
    raise argparse.ArgumentTypeError("identity labels must be JSON scalars")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate aligned per-animal latent NPZ files onto a fixed 0.16 s "
            "Social-GNN node clock."
        )
    )
    parser.add_argument("--animal", action="append", type=_parse_animal, required=True)
    parser.add_argument("--output-node-npz", required=True, type=Path)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--output-identity", nargs="+", type=_identity_label)
    parser.add_argument("--patch-duration-s", type=float, default=DEFAULT_PATCH_DURATION_S)
    parser.add_argument("--patch-stride-s", type=float, default=DEFAULT_PATCH_STRIDE_S)
    parser.add_argument("--clock-start-s", type=float)
    parser.add_argument("--clock-end-s", type=float)
    parser.add_argument(
        "--authoritative-clock-npz",
        type=Path,
        help=(
            "Existing canonical node NPZ whose complete patch clock and identity "
            "order must be preserved. Patches outside latent support remain in "
            "the output with node_mask=false."
        ),
    )
    parser.add_argument("--interpolated-weight", type=float, default=0.0)
    parser.add_argument("--minimum-weight-coverage", type=float, default=0.5)
    parser.add_argument("--allow-duplicate-nodes", action="store_true")
    parser.add_argument("--summary-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = convert_aligned_latents_to_node_npz(
            args.animal,
            args.output_node_npz,
            trial_id=args.trial_id,
            output_identity=args.output_identity,
            patch_duration_s=args.patch_duration_s,
            patch_stride_s=args.patch_stride_s,
            clock_start_s=args.clock_start_s,
            clock_end_s=args.clock_end_s,
            authoritative_clock_npz=args.authoritative_clock_npz,
            interpolated_weight=args.interpolated_weight,
            minimum_weight_coverage=args.minimum_weight_coverage,
            allow_duplicate_nodes=args.allow_duplicate_nodes,
            summary_json=args.summary_json,
        )
    except LatentNodeConversionError as exc:
        raise SystemExit(str(exc)) from None
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
