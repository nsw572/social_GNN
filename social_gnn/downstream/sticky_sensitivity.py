"""Controlled sticky-HMM prior sensitivity analysis on frozen embeddings.

The analysis never refits an encoder or PCA.  It reuses one saved PCA16 per
encoder seed, validates and reuses the formal 50000/2 s HMM checkpoints, and
fits only the remaining transition-prior configurations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations, product
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from threadpoolctl import threadpool_limits

from .embeddings import (
    FrozenEmbeddingTrial,
    load_embedding_directory,
    validate_embedding_timestep,
)
from .evaluation import DecodedTrial, bout_rows, decode_trial, segmentation_metrics
from .gaussian_hmm import GaussianHMM
from .preprocessing import EmbeddingPreprocessor
from .state_correspondence import (
    CorrespondenceThresholds,
    boundary_agreement,
    contingency_matrix,
    hungarian_state_matches,
    overlap_metrics,
    split_merge_diagnostics,
)


TIMESTEP_S = 0.04
N_STATES = 6
PCA_DIM = 16
TRAIN_TRIALS = ("A1_C1", "A3_C3", "A4_B4")
VALIDATION_TRIAL = "A4_C4"
EXCLUDED_TRIAL = "A5_C5"
ENCODER_SEEDS = (20261013, 20261029, 20261107)
DOWNSTREAM_SEEDS = (0, 1, 2)
STRENGTHS = (10.0, 1000.0, 10000.0, 25000.0, 50000.0, 100000.0)
DWELL_GRID_STRENGTHS = (10000.0, 50000.0)
DWELL_TARGETS_S = (1.0, 2.0, 4.0)
BASELINE_STRENGTH = 50000.0
BASELINE_DWELL_S = 2.0


def sensitivity_grid() -> list[tuple[float, float]]:
    """Return the 10 unique strength/dwell settings across both experiments."""

    values = {(strength, 2.0) for strength in STRENGTHS}
    values.update(product(DWELL_GRID_STRENGTHS, DWELL_TARGETS_S))
    return sorted(values, key=lambda item: (item[1], item[0]))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _json_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _setting_slug(strength: float, dwell_s: float) -> str:
    strength_text = str(int(strength)) if float(strength).is_integer() else f"{strength:g}"
    dwell_text = str(int(dwell_s)) if float(dwell_s).is_integer() else f"{dwell_s:g}"
    return f"strength_{strength_text}_dwell_{dwell_text}s"


def _sequences(
    trials: Iterable[FrozenEmbeddingTrial], preprocessor: EmbeddingPreprocessor
) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for trial in trials:
        transformed, _indices = preprocessor.transform_trial_sequences(trial)
        result.extend(transformed)
    return result


def _load_assignment(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "predicted_state",
            "valid_time_mask",
            "patch_start_s",
            "patch_end_s",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"assignment {path} is missing {sorted(missing)}")
        # Partition comparisons never use the potentially large posterior
        # matrix. Loading only these four arrays makes the hundreds of paired
        # comparisons both faster and substantially less memory intensive.
        return {key: np.asarray(data[key]) for key in sorted(required)}


def _validate_assignment_clock(
    assignment: dict[str, np.ndarray], trial: FrozenEmbeddingTrial, source: str | Path
) -> None:
    expected_shape = trial.valid_time_mask.shape
    if np.asarray(assignment["predicted_state"]).shape != expected_shape:
        raise ValueError(f"{source}: assignment shape does not match {trial.trial_id}")
    if not np.array_equal(assignment["valid_time_mask"], trial.valid_time_mask):
        raise ValueError(f"{source}: valid mask does not match frozen embedding")
    for key, expected in (
        ("patch_start_s", trial.patch_start_s),
        ("patch_end_s", trial.patch_end_s),
    ):
        if not np.allclose(assignment[key], expected, rtol=0.0, atol=1e-9):
            raise ValueError(f"{source}: {key} does not match frozen embedding clock")


def _save_assignment(output: Path, decoded: DecodedTrial) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        trial_id=np.asarray(decoded.trial.trial_id),
        patch_index=np.arange(decoded.predicted_state.size),
        patch_start_s=decoded.trial.patch_start_s,
        patch_end_s=decoded.trial.patch_end_s,
        patch_center_s=decoded.trial.patch_center_s,
        valid_time_mask=decoded.trial.valid_time_mask,
        predicted_state=decoded.predicted_state,
        state_posterior=decoded.posterior,
        state_confidence=decoded.state_confidence,
    )
    return output


def _effective_states(labels: np.ndarray, k: int = N_STATES) -> float:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=k).astype(float)
    probability = counts / max(counts.sum(), 1.0)
    positive = probability[probability > 0]
    return float(np.exp(-np.sum(positive * np.log(positive))))


def _partition_comparison(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    *,
    k: int = N_STATES,
    thresholds: CorrespondenceThresholds | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare two clock-aligned partitions without assuming label identity."""

    for key in ("patch_start_s", "patch_end_s"):
        if left[key].shape != right[key].shape or not np.allclose(
            left[key], right[key], rtol=0.0, atol=1e-9
        ):
            raise ValueError(f"partition clocks disagree for {key}")
    left_label = np.asarray(left["predicted_state"], dtype=np.int64)
    right_label = np.asarray(right["predicted_state"], dtype=np.int64)
    common = (
        np.asarray(left["valid_time_mask"], dtype=bool)
        & np.asarray(right["valid_time_mask"], dtype=bool)
        & (left_label >= 0)
        & (right_label >= 0)
    )
    if not common.any():
        raise ValueError("partition comparison has no mutually valid timesteps")
    left_valid = left_label[common]
    right_valid = right_label[common]
    matrix = contingency_matrix(left_valid, right_valid, k)
    matches = hungarian_state_matches(matrix, thresholds=thresholds)
    overlap = overlap_metrics(matrix)
    diagnostic = split_merge_diagnostics(matrix, thresholds=thresholds)
    boundary = boundary_agreement(
        left_label,
        right_label,
        common,
        np.asarray(left["patch_start_s"], dtype=float),
        np.asarray(left["patch_end_s"], dtype=float),
        tolerance_patches=1,
    )
    left_occupancy = np.bincount(left_valid, minlength=k).astype(float)
    right_occupancy = np.bincount(right_valid, minlength=k).astype(float)
    left_occupancy /= left_occupancy.sum()
    right_occupancy /= right_occupancy.sum()
    result: dict[str, Any] = {
        "aligned_valid_timesteps": int(common.sum()),
        "ARI": float(adjusted_rand_score(left_valid, right_valid)),
        "NMI": float(normalized_mutual_info_score(left_valid, right_valid)),
        "mean_hungarian_f1": float(matches["f1"].mean()),
        "mean_hungarian_jaccard": float(matches["jaccard"].mean()),
        "hungarian_matched_patch_fraction": float(
            matches["intersection_patches"].sum() / common.sum()
        ),
        "left_split_candidate_count": int(
            diagnostic["split_candidate"].fillna(False).sum()
        ),
        "right_merge_candidate_count": int(
            diagnostic["merge_candidate"].fillna(False).sum()
        ),
        "occupancy_l1_difference": float(np.abs(left_occupancy - right_occupancy).sum()),
        "effective_state_count_left": _effective_states(left_valid, k),
        "effective_state_count_right": _effective_states(right_valid, k),
        "effective_state_count_difference": float(
            _effective_states(left_valid, k) - _effective_states(right_valid, k)
        ),
        **boundary,
    }
    return result, overlap, matches, diagnostic


def _transition_rows(
    *,
    encoder_seed: int,
    downstream_seed: int,
    strength: float,
    dwell_s: float,
    model: GaussianHMM,
    data_counts: np.ndarray,
) -> list[dict[str, Any]]:
    prior = model.transition_prior_counts()
    rows: list[dict[str, Any]] = []
    for source in range(model.n_states):
        data_row = float(data_counts[source].sum())
        prior_row = float(prior[source].sum())
        for target in range(model.n_states):
            rows.append(
                {
                    "encoder_seed": encoder_seed,
                    "downstream_seed": downstream_seed,
                    "prior_strength": strength,
                    "target_dwell_s": dwell_s,
                    "source_state": source,
                    "target_state": target,
                    "is_diagonal": source == target,
                    "prior_pseudocount": float(prior[source, target]),
                    "expected_data_transition_count": float(data_counts[source, target]),
                    "learned_transition_probability": float(
                        model.transition_[source, target]
                    ),
                    "prior_row_pseudocount": prior_row,
                    "expected_data_row_transitions": data_row,
                    "prior_to_data_ratio_row": (
                        prior_row / data_row if data_row > 0 else float("inf")
                    ),
                }
            )
    return rows


def _per_state_duration_rows(
    *,
    encoder_seed: int,
    downstream_seed: int,
    strength: float,
    dwell_s: float,
    model: GaussianHMM,
    bouts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    diagonal = np.diag(model.transition_)
    for state in range(model.n_states):
        duration = np.asarray(
            [row["bout_duration_s"] for row in bouts if int(row["state"]) == state],
            dtype=float,
        )
        rows.append(
            {
                "encoder_seed": encoder_seed,
                "downstream_seed": downstream_seed,
                "prior_strength": strength,
                "target_dwell_s": dwell_s,
                "state": state,
                "learned_self_transition": float(diagonal[state]),
                "model_implied_mean_dwell_s": float(
                    TIMESTEP_S / max(1.0 - diagonal[state], 1e-12)
                ),
                "observed_bouts": int(duration.size),
                "observed_mean_bout_s": float(duration.mean()) if duration.size else float("nan"),
                "observed_median_bout_s": (
                    float(np.median(duration)) if duration.size else float("nan")
                ),
                "observed_q10_bout_s": (
                    float(np.quantile(duration, 0.10)) if duration.size else float("nan")
                ),
                "observed_q90_bout_s": (
                    float(np.quantile(duration, 0.90)) if duration.size else float("nan")
                ),
            }
        )
    return rows


def _validate_model_contract(
    model: GaussianHMM,
    metadata: dict[str, Any],
    *,
    encoder_seed: int,
    downstream_seed: int,
    strength: float,
    dwell_s: float,
    embedding_dir: Path,
    preprocessor_path: Path,
) -> None:
    checks = {
        "n_states": (model.n_states, N_STATES),
        "seed": (model.seed, downstream_seed),
        "timestep_s": (model.timestep_s, TIMESTEP_S),
        "sticky_prior_strength": (model.sticky_prior_strength, strength),
        "sticky_expected_duration_s": (model.sticky_expected_duration_s, dwell_s),
    }
    failed = [name for name, (actual, expected) in checks.items() if actual != expected]
    if failed:
        raise ValueError(f"checkpoint contract mismatch: {failed}")
    if metadata:
        if int(metadata.get("encoder_seed", encoder_seed)) != encoder_seed:
            raise ValueError("checkpoint encoder seed mismatch")
        if Path(metadata.get("embedding_dir", embedding_dir)).resolve() != embedding_dir.resolve():
            raise ValueError("checkpoint embedding directory mismatch")
        if Path(metadata.get("preprocessor", preprocessor_path)).resolve() != preprocessor_path.resolve():
            raise ValueError("checkpoint PCA path mismatch")


def _analyze_one_encoder(payload: dict[str, Any]) -> dict[str, Any]:
    encoder_seed = int(payload["encoder_seed"])
    embedding_dir = Path(payload["embedding_dir"]).resolve()
    baseline_dir = Path(payload["baseline_dir"]).resolve()
    output_root = Path(payload["output_root"]).resolve()
    max_iter = int(payload["max_iter"])
    tolerance = float(payload["tolerance"])
    covariance_floor = float(payload["covariance_floor"])

    trials = load_embedding_directory(embedding_dir)
    validate_embedding_timestep(trials, TIMESTEP_S)
    by_id = {trial.trial_id: trial for trial in trials}
    required = set(TRAIN_TRIALS) | {VALIDATION_TRIAL}
    if required - set(by_id):
        raise ValueError(f"encoder {encoder_seed}: missing trials {sorted(required - set(by_id))}")
    train_trials = [by_id[name] for name in TRAIN_TRIALS]
    validation_trial = by_id[VALIDATION_TRIAL]
    preprocessor_path = baseline_dir / "preprocessing" / f"pca_{PCA_DIM}.npz"
    preprocessor = EmbeddingPreprocessor.load(preprocessor_path)
    if preprocessor.output_dim != PCA_DIM:
        raise ValueError(f"encoder {encoder_seed}: expected PCA{PCA_DIM}")
    train_sequences = _sequences(train_trials, preprocessor)
    validation_sequences = _sequences([validation_trial], preprocessor)
    train_steps = int(sum(sequence.shape[0] for sequence in train_sequences))
    validation_steps = int(sum(sequence.shape[0] for sequence in validation_sequences))
    expected_transitions = int(sum(max(sequence.shape[0] - 1, 0) for sequence in train_sequences))
    embedding_manifest = embedding_dir / "manifest.json"
    embedding_sha = _sha256(embedding_manifest)
    pca_sha = _sha256(preprocessor_path)

    run_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    state_duration_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    reused = 0

    for strength, dwell_s in sensitivity_grid():
        for downstream_seed in DOWNSTREAM_SEEDS:
            started = time.perf_counter()
            slug = _setting_slug(strength, dwell_s)
            run_dir = output_root / "runs" / f"encoder_seed_{encoder_seed}" / slug / f"hmm_seed_{downstream_seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            baseline = strength == BASELINE_STRENGTH and dwell_s == BASELINE_DWELL_S
            reused_this_run = False
            try:
                if baseline:
                    source_run = baseline_dir / "hmm" / f"hmm_k{N_STATES}_seed{downstream_seed}"
                    model_path = source_run / "model.npz"
                    assignment_path = source_run / "assignments" / f"{VALIDATION_TRIAL}_states.npz"
                    model, metadata = GaussianHMM.load(model_path)
                    _validate_model_contract(
                        model,
                        metadata,
                        encoder_seed=encoder_seed,
                        downstream_seed=downstream_seed,
                        strength=strength,
                        dwell_s=dwell_s,
                        embedding_dir=embedding_dir,
                        preprocessor_path=preprocessor_path,
                    )
                    saved_assignment = _load_assignment(assignment_path)
                    _validate_assignment_clock(saved_assignment, validation_trial, assignment_path)
                    decoded = decode_trial(model, preprocessor, validation_trial)
                    if not np.array_equal(
                        saved_assignment["predicted_state"], decoded.predicted_state
                    ):
                        raise ValueError("formal baseline checkpoint and saved assignment disagree")
                    source_artifact = str(model_path)
                    reused += 1
                    reused_this_run = True
                    _write_json(
                        run_dir / "baseline_reference.json",
                        {
                            "model": str(model_path),
                            "model_sha256": _sha256(model_path),
                            "assignment": str(assignment_path),
                            "assignment_sha256": _sha256(assignment_path),
                            "preprocessor": str(preprocessor_path),
                            "preprocessor_sha256": pca_sha,
                        },
                    )
                else:
                    model_path = run_dir / "model.npz"
                    assignment_path = run_dir / f"{VALIDATION_TRIAL}_states.npz"
                    if model_path.exists() and assignment_path.exists():
                        model, metadata = GaussianHMM.load(model_path)
                        _validate_model_contract(
                            model,
                            metadata,
                            encoder_seed=encoder_seed,
                            downstream_seed=downstream_seed,
                            strength=strength,
                            dwell_s=dwell_s,
                            embedding_dir=embedding_dir,
                            preprocessor_path=preprocessor_path,
                        )
                        decoded = decode_trial(model, preprocessor, validation_trial)
                        saved_assignment = _load_assignment(assignment_path)
                        _validate_assignment_clock(saved_assignment, validation_trial, assignment_path)
                        if not np.array_equal(
                            saved_assignment["predicted_state"], decoded.predicted_state
                        ):
                            raise ValueError("resumed checkpoint and assignment disagree")
                        source_artifact = str(model_path)
                        reused += 1
                        reused_this_run = True
                    else:
                        model = GaussianHMM(
                            n_states=N_STATES,
                            covariance_floor=covariance_floor,
                            timestep_s=TIMESTEP_S,
                            sticky_expected_duration_s=dwell_s,
                            sticky_prior_strength=strength,
                            seed=downstream_seed,
                        )
                        with threadpool_limits(limits=1):
                            model.fit(
                                train_sequences,
                                max_iter=max_iter,
                                tolerance=tolerance,
                            )
                        decoded = decode_trial(model, preprocessor, validation_trial)
                        metadata = {
                            "analysis": "sticky_prior_sensitivity_v1",
                            "encoder": "graph_full",
                            "encoder_seed": encoder_seed,
                            "downstream_seed": downstream_seed,
                            "embedding_dir": str(embedding_dir),
                            "embedding_manifest_sha256": embedding_sha,
                            "preprocessor": str(preprocessor_path),
                            "preprocessor_sha256": pca_sha,
                            "train_trials": list(TRAIN_TRIALS),
                            "validation_trials": [VALIDATION_TRIAL],
                            "excluded_trials": [EXCLUDED_TRIAL],
                        }
                        model.save(model_path, metadata=metadata)
                        _save_assignment(assignment_path, decoded)
                        source_artifact = str(model_path)

                train_likelihood = float(model.score(train_sequences))
                validation_likelihood = float(model.score(validation_sequences))
                data_counts = model.expected_transition_counts(train_sequences)
                if not np.isclose(data_counts.sum(), expected_transitions, rtol=1e-6, atol=1e-3):
                    raise ValueError(
                        "posterior transition counts do not equal available within-sequence transitions"
                    )
                prior_counts = model.transition_prior_counts()
                diagonal = np.diag(model.transition_)
                transition_entropy = -np.sum(
                    model.transition_ * np.log(np.maximum(model.transition_, 1e-300)), axis=1
                )
                decoded_metrics = segmentation_metrics([decoded], N_STATES)
                bouts = bout_rows(decoded)
                durations = np.asarray([row["bout_duration_s"] for row in bouts], dtype=float)
                states = decoded.predicted_state[decoded.predicted_state >= 0]
                occupancy = np.bincount(states, minlength=N_STATES).astype(float)
                occupancy /= occupancy.sum()
                pd.DataFrame(bouts).to_csv(run_dir / "bouts.csv", index=False)
                np.savez_compressed(
                    run_dir / "transition_diagnostics.npz",
                    learned_transition=model.transition_,
                    prior_pseudocounts=prior_counts,
                    expected_data_transition_counts=data_counts,
                )
                row: dict[str, Any] = {
                    "status": "completed",
                    "encoder": "graph_full",
                    "encoder_seed": encoder_seed,
                    "downstream_seed": downstream_seed,
                    "prior_strength": strength,
                    "target_dwell_s": dwell_s,
                    "target_self_transition": model.sticky_self_transition_probability,
                    "K": N_STATES,
                    "pca_dim": preprocessor.output_dim,
                    "timestep_s": TIMESTEP_S,
                    "train_valid_points": train_steps,
                    "validation_valid_points": validation_steps,
                    "available_training_transitions": expected_transitions,
                    "soft_expected_training_transitions": float(data_counts.sum()),
                    "prior_pseudocount_mass_per_row": float(prior_counts[0].sum()),
                    "train_log_likelihood_total": train_likelihood,
                    "train_log_likelihood_per_point": train_likelihood / train_steps,
                    "validation_log_likelihood_total": validation_likelihood,
                    "validation_log_likelihood_per_point": (
                        validation_likelihood / validation_steps
                    ),
                    "iterations": len(model.history_),
                    "converged": bool(model.converged_),
                    "learned_self_transition_mean": float(diagonal.mean()),
                    "learned_self_transition_min": float(diagonal.min()),
                    "learned_self_transition_max": float(diagonal.max()),
                    "transition_entropy_mean": float(transition_entropy.mean()),
                    "transition_entropy_min": float(transition_entropy.min()),
                    "transition_entropy_max": float(transition_entropy.max()),
                    "fraction_bouts_shorter_0p20_s": float(np.mean(durations < 0.20)),
                    "fraction_bouts_shorter_0p50_s": float(np.mean(durations < 0.50)),
                    "occupancy_distribution_json": json.dumps(occupancy.tolist()),
                    "embedding_dir": str(embedding_dir),
                    "embedding_manifest_sha256": embedding_sha,
                    "preprocessor": str(preprocessor_path),
                    "preprocessor_sha256": pca_sha,
                    "checkpoint": str(model_path),
                    "assignment": str(assignment_path),
                    "source_artifact": source_artifact,
                    "reused_existing_artifact": reused_this_run,
                    "elapsed_s": time.perf_counter() - started,
                    **decoded_metrics,
                }
                for state, value in enumerate(diagonal):
                    row[f"learned_self_transition_state_{state}"] = float(value)
                    row[f"model_implied_mean_dwell_state_{state}_s"] = float(
                        TIMESTEP_S / max(1.0 - value, 1e-12)
                    )
                    row[f"occupancy_state_{state}"] = float(occupancy[state])
                run_rows.append(row)
                transition_rows.extend(
                    _transition_rows(
                        encoder_seed=encoder_seed,
                        downstream_seed=downstream_seed,
                        strength=strength,
                        dwell_s=dwell_s,
                        model=model,
                        data_counts=data_counts,
                    )
                )
                state_duration_rows.extend(
                    _per_state_duration_rows(
                        encoder_seed=encoder_seed,
                        downstream_seed=downstream_seed,
                        strength=strength,
                        dwell_s=dwell_s,
                        model=model,
                        bouts=bouts,
                    )
                )
            except Exception as error:
                failure = {
                    "status": "failed",
                    "encoder_seed": encoder_seed,
                    "downstream_seed": downstream_seed,
                    "prior_strength": strength,
                    "target_dwell_s": dwell_s,
                    "failure": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "elapsed_s": time.perf_counter() - started,
                }
                failures.append(failure)
                run_rows.append(failure)
            pd.DataFrame(run_rows).to_csv(
                output_root / f"_partial_runs_encoder_{encoder_seed}.csv", index=False
            )

    return {
        "encoder_seed": encoder_seed,
        "runs": run_rows,
        "transition_rows": transition_rows,
        "state_duration_rows": state_duration_rows,
        "failures": failures,
        "reused_runs": reused,
        "embedding_manifest_sha256": embedding_sha,
        "preprocessor_sha256": pca_sha,
    }


def _assignment_for_row(row: pd.Series) -> dict[str, np.ndarray]:
    return _load_assignment(str(row["assignment"]))


def _comparison_row(
    left: pd.Series,
    right: pd.Series,
    *,
    scope: str,
    thresholds: CorrespondenceThresholds,
    assignment_cache: dict[str, dict[str, np.ndarray]] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def load(row: pd.Series) -> dict[str, np.ndarray]:
        path = str(row["assignment"])
        if assignment_cache is None:
            return _load_assignment(path)
        if path not in assignment_cache:
            assignment_cache[path] = _load_assignment(path)
        return assignment_cache[path]

    metrics, overlap, matches, diagnostic = _partition_comparison(
        load(left),
        load(right),
        thresholds=thresholds,
    )
    metadata = {
        "comparison_scope": scope,
        "left_encoder_seed": int(left.encoder_seed),
        "right_encoder_seed": int(right.encoder_seed),
        "left_downstream_seed": int(left.downstream_seed),
        "right_downstream_seed": int(right.downstream_seed),
        "left_prior_strength": float(left.prior_strength),
        "right_prior_strength": float(right.prior_strength),
        "left_target_dwell_s": float(left.target_dwell_s),
        "right_target_dwell_s": float(right.target_dwell_s),
    }
    return {**metadata, **metrics}, overlap.assign(**metadata), matches.assign(**metadata), diagnostic.assign(**metadata)


def _compute_comparisons(
    runs: pd.DataFrame, output: Path, thresholds: CorrespondenceThresholds
) -> dict[str, pd.DataFrame]:
    completed = runs[runs.status.eq("completed")].copy()
    expected = len(ENCODER_SEEDS) * len(DOWNSTREAM_SEEDS) * len(sensitivity_grid())
    if completed.shape[0] != expected:
        raise RuntimeError(
            f"cannot compare incomplete sensitivity grid: {completed.shape[0]}/{expected} runs"
        )
    assignment_cache: dict[str, dict[str, np.ndarray]] = {}

    baseline_rows: list[dict[str, Any]] = []
    overlap_frames: list[pd.DataFrame] = []
    match_frames: list[pd.DataFrame] = []
    diagnostic_frames: list[pd.DataFrame] = []
    for (encoder_seed, downstream_seed), group in completed.groupby(
        ["encoder_seed", "downstream_seed"]
    ):
        baseline = group[
            group.prior_strength.eq(BASELINE_STRENGTH)
            & group.target_dwell_s.eq(BASELINE_DWELL_S)
        ]
        if baseline.shape[0] != 1:
            raise ValueError(
                f"baseline match for encoder={encoder_seed}, seed={downstream_seed} has {baseline.shape[0]} rows"
            )
        reference = baseline.iloc[0]
        for _, candidate in group.iterrows():
            row, overlap, matches, diagnostic = _comparison_row(
                candidate,
                reference,
                scope="same_encoder_seed_and_hmm_seed_vs_50000_2s",
                thresholds=thresholds,
                assignment_cache=assignment_cache,
            )
            baseline_rows.append(row)
            overlap_frames.append(overlap)
            match_frames.append(matches)
            diagnostic_frames.append(diagnostic)
    baseline_frame = pd.DataFrame(baseline_rows)
    baseline_frame.to_csv(output / "baseline_partition_correspondence.csv", index=False)
    pd.concat(overlap_frames, ignore_index=True).to_csv(
        output / "state_overlap_diagnostics.csv", index=False
    )
    pd.concat(match_frames, ignore_index=True).to_csv(
        output / "hungarian_state_matches.csv", index=False
    )
    pd.concat(diagnostic_frames, ignore_index=True).to_csv(
        output / "split_merge_diagnostics.csv", index=False
    )

    pairwise_strength: list[dict[str, Any]] = []
    strength_only = completed[completed.target_dwell_s.eq(2.0)]
    for (_encoder, _seed), group in strength_only.groupby(
        ["encoder_seed", "downstream_seed"]
    ):
        ordered = group.sort_values("prior_strength")
        for left_index in range(ordered.shape[0]):
            for right_index in range(left_index, ordered.shape[0]):
                left = ordered.iloc[left_index]
                right = ordered.iloc[right_index]
                row, _overlap, _matches, _diagnostic = _comparison_row(
                    left,
                    right,
                    scope="fixed_encoder_and_hmm_seed_strength_pair",
                    thresholds=thresholds,
                    assignment_cache=assignment_cache,
                )
                pairwise_strength.append(row)
    pairwise_frame = pd.DataFrame(pairwise_strength)
    pairwise_frame.to_csv(output / "pairwise_strength_ari.csv", index=False)

    within_rows: list[dict[str, Any]] = []
    for (_strength, _dwell, _encoder), group in completed.groupby(
        ["prior_strength", "target_dwell_s", "encoder_seed"]
    ):
        ordered = group.sort_values("downstream_seed")
        for left_index, right_index in combinations(range(ordered.shape[0]), 2):
            row, _overlap, _matches, _diagnostic = _comparison_row(
                ordered.iloc[left_index],
                ordered.iloc[right_index],
                scope="fixed_encoder_different_hmm_seed",
                thresholds=thresholds,
                assignment_cache=assignment_cache,
            )
            within_rows.append(row)
    within_frame = pd.DataFrame(within_rows)
    within_frame.to_csv(output / "within_encoder_stability.csv", index=False)

    cross_rows: list[dict[str, Any]] = []
    for (_strength, _dwell), group in completed.groupby(
        ["prior_strength", "target_dwell_s"]
    ):
        by_encoder = {
            int(seed): frame.sort_values("downstream_seed")
            for seed, frame in group.groupby("encoder_seed")
        }
        for left_seed, right_seed in combinations(sorted(by_encoder), 2):
            for _, left in by_encoder[left_seed].iterrows():
                for _, right in by_encoder[right_seed].iterrows():
                    row, _overlap, _matches, _diagnostic = _comparison_row(
                        left,
                        right,
                        scope="different_encoder_seed",
                        thresholds=thresholds,
                        assignment_cache=assignment_cache,
                    )
                    row["downstream_seed_relation"] = (
                        "matched_identifier"
                        if int(left.downstream_seed) == int(right.downstream_seed)
                        else "different_identifier"
                    )
                    cross_rows.append(row)
    cross_frame = pd.DataFrame(cross_rows)
    cross_frame.to_csv(output / "cross_encoder_stability.csv", index=False)
    return {
        "baseline": baseline_frame,
        "pairwise_strength": pairwise_frame,
        "within": within_frame,
        "cross": cross_frame,
    }


def _add_likelihood_deltas(runs: pd.DataFrame) -> pd.DataFrame:
    frame = runs.copy()
    baseline = frame[
        frame.status.eq("completed")
        & frame.prior_strength.eq(BASELINE_STRENGTH)
        & frame.target_dwell_s.eq(BASELINE_DWELL_S)
    ][
        [
            "encoder_seed",
            "downstream_seed",
            "train_log_likelihood_per_point",
            "validation_log_likelihood_per_point",
        ]
    ].rename(
        columns={
            "train_log_likelihood_per_point": "baseline_train_log_likelihood_per_point",
            "validation_log_likelihood_per_point": "baseline_validation_log_likelihood_per_point",
        }
    )
    if baseline.duplicated(["encoder_seed", "downstream_seed"]).any():
        raise ValueError("baseline likelihood table has duplicate seed matches")
    frame = frame.merge(
        baseline,
        on=["encoder_seed", "downstream_seed"],
        how="left",
        validate="many_to_one",
    )
    frame["train_log_likelihood_delta_vs_50000_2s"] = (
        frame.train_log_likelihood_per_point
        - frame.baseline_train_log_likelihood_per_point
    )
    frame["validation_log_likelihood_delta_vs_50000_2s"] = (
        frame.validation_log_likelihood_per_point
        - frame.baseline_validation_log_likelihood_per_point
    )
    return frame


def _mean_sd(values: pd.Series) -> str:
    numeric = values.astype(float)
    return f"{numeric.mean():.4f} +/- {numeric.std(ddof=1):.4f}"


def _strength_summary(
    runs: pd.DataFrame, comparisons: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    strength_runs = runs[runs.target_dwell_s.eq(2.0) & runs.status.eq("completed")]
    summary = strength_runs.groupby("prior_strength", as_index=False).agg(
        learned_self_transition_mean=("learned_self_transition_mean", "mean"),
        learned_self_transition_sd=("learned_self_transition_mean", "std"),
        bouts_per_minute_mean=("bouts_per_minute", "mean"),
        bouts_per_minute_sd=("bouts_per_minute", "std"),
        median_bout_s_mean=("median_bout_duration_s", "mean"),
        one_timestep_fraction_mean=("fraction_one_timestep_bouts", "mean"),
        validation_ll_delta_mean=(
            "validation_log_likelihood_delta_vs_50000_2s",
            "mean",
        ),
        effective_states_mean=("effective_entropy_states", "mean"),
        largest_state_mean=("largest_state_occupancy", "mean"),
    )
    baseline = comparisons["baseline"]
    baseline = baseline[
        baseline.left_target_dwell_s.eq(2.0)
        & baseline.right_target_dwell_s.eq(2.0)
    ].groupby("left_prior_strength", as_index=False).agg(
        baseline_ARI_mean=("ARI", "mean"),
        baseline_ARI_sd=("ARI", "std"),
        baseline_matched_f1_mean=("mean_hungarian_f1", "mean"),
        baseline_boundary_f1_mean=("boundary_f1_exact", "mean"),
    ).rename(columns={"left_prior_strength": "prior_strength"})
    within = comparisons["within"]
    within = within[
        within.left_target_dwell_s.eq(2.0)
    ].groupby("left_prior_strength", as_index=False).agg(
        within_encoder_ARI_mean=("ARI", "mean"),
        within_encoder_ARI_sd=("ARI", "std"),
        within_encoder_ARI_min=("ARI", "min"),
    ).rename(columns={"left_prior_strength": "prior_strength"})
    cross = comparisons["cross"]
    cross = cross[
        cross.left_target_dwell_s.eq(2.0)
    ].groupby("left_prior_strength", as_index=False).agg(
        cross_encoder_ARI_mean=("ARI", "mean"),
        cross_encoder_ARI_sd=("ARI", "std"),
    ).rename(columns={"left_prior_strength": "prior_strength"})
    return summary.merge(baseline, on="prior_strength").merge(
        within, on="prior_strength"
    ).merge(cross, on="prior_strength")


def _dwell_summary(
    runs: pd.DataFrame, comparisons: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    selected = runs[
        runs.status.eq("completed") & runs.prior_strength.isin(DWELL_GRID_STRENGTHS)
    ]
    summary = selected.groupby(
        ["prior_strength", "target_dwell_s"], as_index=False
    ).agg(
        learned_self_transition_mean=("learned_self_transition_mean", "mean"),
        validation_ll_delta_mean=(
            "validation_log_likelihood_delta_vs_50000_2s",
            "mean",
        ),
        bouts_per_minute_mean=("bouts_per_minute", "mean"),
        median_bout_s_mean=("median_bout_duration_s", "mean"),
        one_timestep_fraction_mean=("fraction_one_timestep_bouts", "mean"),
        effective_states_mean=("effective_entropy_states", "mean"),
    )
    baseline = comparisons["baseline"].groupby(
        ["left_prior_strength", "left_target_dwell_s"], as_index=False
    ).agg(baseline_ARI_mean=("ARI", "mean")).rename(
        columns={
            "left_prior_strength": "prior_strength",
            "left_target_dwell_s": "target_dwell_s",
        }
    )
    within = comparisons["within"].groupby(
        ["left_prior_strength", "left_target_dwell_s"], as_index=False
    ).agg(within_encoder_ARI_mean=("ARI", "mean")).rename(
        columns={
            "left_prior_strength": "prior_strength",
            "left_target_dwell_s": "target_dwell_s",
        }
    )
    cross = comparisons["cross"].groupby(
        ["left_prior_strength", "left_target_dwell_s"], as_index=False
    ).agg(cross_encoder_ARI_mean=("ARI", "mean")).rename(
        columns={
            "left_prior_strength": "prior_strength",
            "left_target_dwell_s": "target_dwell_s",
        }
    )
    return summary.merge(baseline, on=["prior_strength", "target_dwell_s"]).merge(
        within, on=["prior_strength", "target_dwell_s"]
    ).merge(cross, on=["prior_strength", "target_dwell_s"])


def _plot_strength_metrics(runs: pd.DataFrame, figures: Path) -> None:
    frame = runs[runs.status.eq("completed") & runs.target_dwell_s.eq(2.0)].copy()
    specs = (
        ("learned_self_transition_mean", "Learned mean self-transition"),
        ("bouts_per_minute", "Bouts / minute"),
        ("median_bout_duration_s", "Median bout (s)"),
        ("fraction_one_timestep_bouts", "One-timestep bout fraction"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    colors = {20261013: "#1f77b4", 20261029: "#ff7f0e", 20261107: "#2ca02c"}
    for axis, (column, label) in zip(axes.ravel(), specs):
        for encoder_seed, group in frame.groupby("encoder_seed"):
            for downstream_seed, run in group.groupby("downstream_seed"):
                ordered = run.sort_values("prior_strength")
                axis.plot(
                    ordered.prior_strength,
                    ordered[column],
                    color=colors[int(encoder_seed)],
                    alpha=0.25,
                    linewidth=1,
                )
            average = group.groupby("prior_strength", as_index=False)[column].mean()
            axis.plot(
                average.prior_strength,
                average[column],
                color=colors[int(encoder_seed)],
                marker="o",
                linewidth=2,
                label=f"encoder {int(encoder_seed)}",
            )
        axis.set_xscale("log")
        axis.set_xlabel("Sticky prior strength")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    figure.savefig(figures / "prior_strength_vs_temporal_metrics.png", dpi=220)
    plt.close(figure)


def _plot_likelihood(runs: pd.DataFrame, figures: Path) -> None:
    frame = runs[runs.status.eq("completed") & runs.target_dwell_s.eq(2.0)]
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for encoder_seed, group in frame.groupby("encoder_seed"):
        for downstream_seed, run in group.groupby("downstream_seed"):
            ordered = run.sort_values("prior_strength")
            axis.plot(
                ordered.prior_strength,
                ordered.validation_log_likelihood_delta_vs_50000_2s,
                marker="o",
                alpha=0.55,
                label=f"{int(encoder_seed)} / HMM {int(downstream_seed)}",
            )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xscale("log")
    axis.set_xlabel("Sticky prior strength")
    axis.set_ylabel("A4_C4 marginal log-likelihood / point delta vs 50000/2 s")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=6, ncol=3)
    figure.savefig(figures / "prior_strength_vs_likelihood.png", dpi=220)
    plt.close(figure)


def _plot_ari(
    summary: pd.DataFrame, baseline: pd.DataFrame, figures: Path
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.plot(
        summary.prior_strength,
        summary.within_encoder_ARI_mean,
        marker="o",
        label="within encoder",
    )
    axis.plot(
        summary.prior_strength,
        summary.cross_encoder_ARI_mean,
        marker="o",
        label="cross encoder",
    )
    axis.set_xscale("log")
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("Sticky prior strength")
    axis.set_ylabel("Pairwise ARI")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(figures / "prior_strength_vs_ari.png", dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for encoder_seed, group in baseline.groupby("left_encoder_seed"):
        average = group.groupby("left_prior_strength", as_index=False).ARI.mean()
        axis.plot(
            average.left_prior_strength,
            average.ARI,
            marker="o",
            label=f"encoder {int(encoder_seed)}",
        )
    axis.set_xscale("log")
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("Sticky prior strength")
    axis.set_ylabel("ARI versus matched 50000/2 s run")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(figures / "baseline_partition_ari.png", dpi=220)
    plt.close(figure)


def _plot_pairwise_heatmaps(pairwise: pd.DataFrame, figures: Path) -> None:
    output = figures / "pairwise_strength_heatmaps"
    output.mkdir(parents=True, exist_ok=True)
    strengths = np.asarray(STRENGTHS, dtype=float)
    matrices: list[np.ndarray] = []
    for (encoder_seed, downstream_seed), group in pairwise.groupby(
        ["left_encoder_seed", "left_downstream_seed"]
    ):
        matrix = np.eye(len(strengths))
        lookup = {float(value): index for index, value in enumerate(strengths)}
        for row in group.itertuples():
            left = lookup[float(row.left_prior_strength)]
            right = lookup[float(row.right_prior_strength)]
            matrix[left, right] = matrix[right, left] = float(row.ARI)
        matrices.append(matrix)
        figure, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        axis.set_xticks(range(len(strengths)), [f"{value:g}" for value in strengths], rotation=45)
        axis.set_yticks(range(len(strengths)), [f"{value:g}" for value in strengths])
        axis.set_title(f"Encoder {int(encoder_seed)}, HMM seed {int(downstream_seed)}")
        axis.set_xlabel("Prior strength")
        axis.set_ylabel("Prior strength")
        figure.colorbar(image, ax=axis, label="ARI")
        figure.savefig(
            output / f"encoder_{int(encoder_seed)}_hmm_{int(downstream_seed)}.png",
            dpi=200,
        )
        plt.close(figure)
    mean_matrix = np.mean(np.stack(matrices), axis=0)
    figure, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    image = axis.imshow(mean_matrix, vmin=0, vmax=1, cmap="viridis")
    axis.set_xticks(range(len(strengths)), [f"{value:g}" for value in strengths], rotation=45)
    axis.set_yticks(range(len(strengths)), [f"{value:g}" for value in strengths])
    axis.set_title("Mean across 9 matched encoder/HMM seed blocks")
    axis.set_xlabel("Prior strength")
    axis.set_ylabel("Prior strength")
    figure.colorbar(image, ax=axis, label="ARI")
    figure.savefig(output / "mean_across_seed_blocks.png", dpi=220)
    plt.close(figure)


def _plot_occupancy(runs: pd.DataFrame, figures: Path) -> None:
    frame = runs[runs.status.eq("completed") & runs.target_dwell_s.eq(2.0)]
    grouped = frame.groupby("prior_strength", as_index=False).agg(
        effective=("effective_entropy_states", "mean"),
        largest=("largest_state_occupancy", "mean"),
    )
    figure, left_axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    right_axis = left_axis.twinx()
    left_axis.plot(grouped.prior_strength, grouped.effective, marker="o", color="#1f77b4")
    right_axis.plot(grouped.prior_strength, grouped.largest, marker="s", color="#d62728")
    left_axis.set_xscale("log")
    left_axis.set_xlabel("Sticky prior strength")
    left_axis.set_ylabel("Entropy-effective states", color="#1f77b4")
    right_axis.set_ylabel("Largest-state occupancy", color="#d62728")
    left_axis.grid(alpha=0.25)
    figure.savefig(figures / "occupancy_vs_prior.png", dpi=220)
    plt.close(figure)


def _plot_dwell_grid(summary: pd.DataFrame, figures: Path) -> None:
    metrics = (
        ("validation_ll_delta_mean", "Development LL delta"),
        ("within_encoder_ARI_mean", "Within-encoder ARI"),
        ("cross_encoder_ARI_mean", "Cross-encoder ARI"),
        ("bouts_per_minute_mean", "Bouts / min"),
        ("baseline_ARI_mean", "ARI vs 50000/2 s"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for axis, (column, title) in zip(axes.ravel(), metrics):
        pivot = summary.pivot(
            index="prior_strength", columns="target_dwell_s", values=column
        ).reindex(index=DWELL_GRID_STRENGTHS, columns=DWELL_TARGETS_S)
        image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
        axis.set_xticks(range(len(DWELL_TARGETS_S)), [f"{value:g}" for value in DWELL_TARGETS_S])
        axis.set_yticks(range(len(DWELL_GRID_STRENGTHS)), [f"{value:g}" for value in DWELL_GRID_STRENGTHS])
        axis.set_xlabel("Target dwell (s)")
        axis.set_ylabel("Prior strength")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=0.8)
    axes.ravel()[-1].axis("off")
    figure.savefig(figures / "dwell_strength_grid.png", dpi=220)
    plt.close(figure)


def _write_audit(output: Path) -> None:
    text = f"""# Sticky HMM transition-prior implementation audit

The implementation uses a row-wise Dirichlet pseudocount matrix inside every
EM transition update. For timestep `dt` and target mean geometric dwell `tau`,
the prior target is `p_stay = 1 - dt / tau`. At `dt={TIMESTEP_S}` s and
`tau={BASELINE_DWELL_S}` s this is `0.98`.

For K={N_STATES}, a strength S contributes total mass S per source-state row:
`S * 0.98` on the diagonal and `S * (0.02 / 5)` to each off-diagonal cell.
The ordinary isotropic transition pseudocount `0.01` is then added to every
cell, so the exact total prior row mass is `S + 0.06`. Strength is therefore
not diagonal-only mass.

At every EM iteration, the implementation initializes transition counts with
this prior matrix, adds posterior expected soft transition counts `xi` from
forward-backward separately for every valid sequence, and row-normalizes the
sum. It never inserts a transition at a trial boundary or invalid gap. The
same target matrix also initializes the transition probabilities before the
first E-step. No transition clipping or probability floor is applied to the
M-step result; a numerical floor of `1e-300` is used only before logarithms.

The exported `transition_prior_vs_counts.csv` contains exact post-fit soft
expected counts recomputed with each fitted model, not Viterbi hard counts or
patch occupancy. `transition_diagnostics.npz` preserves the full learned
matrix, prior matrix, and soft expected-count matrix for every newly fitted
run. Existing formal 50000/2 s checkpoints remain unchanged and are referenced
by hash.
"""
    (output / "prior_implementation_audit.md").write_text(text, encoding="utf-8")


def _markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str]]) -> list[str]:
    header = "| " + " | ".join(label for _column, label in columns) + " |"
    separator = "| " + " | ".join("---:" for _column, _label in columns) + " |"
    rows = [header, separator]
    for value in frame.itertuples(index=False):
        pieces: list[str] = []
        for column, _label in columns:
            item = getattr(value, column)
            pieces.append(f"{item:.4f}" if isinstance(item, (float, np.floating)) else str(item))
        rows.append("| " + " | ".join(pieces) + " |")
    return rows


def _write_report(
    output: Path,
    runs: pd.DataFrame,
    strength: pd.DataFrame,
    dwell: pd.DataFrame,
    comparisons: dict[str, pd.DataFrame],
) -> None:
    transition = pd.read_csv(output / "transition_prior_vs_counts.csv")
    transition_by_row = transition.drop_duplicates(
        [
            "encoder_seed",
            "downstream_seed",
            "prior_strength",
            "target_dwell_s",
            "source_state",
        ]
    )
    baseline_leverage = transition_by_row[
        transition_by_row.prior_strength.eq(BASELINE_STRENGTH)
        & transition_by_row.target_dwell_s.eq(BASELINE_DWELL_S)
    ]
    best_likelihood = strength.loc[strength.validation_ll_delta_mean.idxmax()]
    most_stable = strength.loc[strength.within_encoder_ARI_mean.idxmax()]
    baseline = strength.loc[strength.prior_strength.eq(BASELINE_STRENGTH)].iloc[0]
    lower_plateau = strength.loc[strength.prior_strength.eq(25000.0)].iloc[0]
    weak = strength.loc[strength.prior_strength.eq(STRENGTHS[0])].iloc[0]
    strong = strength.loc[strength.prior_strength.eq(STRENGTHS[-1])].iloc[0]
    dwell_50k = dwell[dwell.prior_strength.eq(50000.0)].sort_values("target_dwell_s")
    dwell_1s = dwell_50k[dwell_50k.target_dwell_s.eq(1.0)].iloc[0]
    dwell_4s = dwell_50k[dwell_50k.target_dwell_s.eq(4.0)].iloc[0]
    matched_cross = comparisons["cross"][
        comparisons["cross"].downstream_seed_relation.eq("matched_identifier")
        & comparisons["cross"].left_target_dwell_s.eq(2.0)
    ]
    cross_by_strength = matched_cross.groupby("left_prior_strength").ARI.mean()

    lines = [
        "# 0.04 s Social-GNN: Sticky HMM Prior Sensitivity, State Stability, and Temporal Segmentation",
        "",
        "## 1. Executive summary",
        "",
        f"All {runs.status.eq('completed').sum()} unique configurations completed; "
        f"{int(runs.reused_existing_artifact.sum())} runs were validated existing artifacts or resumptions. "
        "The analysis reused frozen Graph Full embeddings and each encoder seed's saved PCA16; no encoder or PCA was retrained.",
        "",
        f"The prior materially controls temporal segmentation: increasing strength from {STRENGTHS[0]:g} to {STRENGTHS[-1]:g} "
        f"changed mean learned self-transition from {weak.learned_self_transition_mean:.4f} to {strong.learned_self_transition_mean:.4f}, "
        f"bouts/min from {weak.bouts_per_minute_mean:.2f} to {strong.bouts_per_minute_mean:.2f}, and median bout duration from "
        f"{weak.median_bout_s_mean:.3f} s to {strong.median_bout_s_mean:.3f} s.",
        "",
        f"The six-state partition is substantially, but not perfectly, conserved. Matched-seed ARI against the current baseline is {weak.baseline_ARI_mean:.3f} at strength 10, {lower_plateau.baseline_ARI_mean:.3f} at 25000, and {strong.baseline_ARI_mean:.3f} at 100000. Thus 25000-100000 is a broad high-strength region rather than evidence for one sharply tuned optimum.",
        "",
        f"At the current 50000/2 s setting, mean baseline-relative ARI is exactly {baseline.baseline_ARI_mean:.3f} by definition, "
        f"within-encoder ARI is {baseline.within_encoder_ARI_mean:.3f}, and cross-encoder ARI is {baseline.cross_encoder_ARI_mean:.3f}. "
        f"The highest within-encoder stability occurred at strength {most_stable.prior_strength:g} "
        f"(ARI {most_stable.within_encoder_ARI_mean:.3f}); the best mean development likelihood delta occurred at "
        f"strength {best_likelihood.prior_strength:g} ({best_likelihood.validation_ll_delta_mean:+.4f} per point versus 50000/2 s).",
        "",
        f"Changing the dwell target at strength 50000 has a smaller effect than weakening the prior: 1 s and 4 s targets retain baseline ARI {dwell_1s.baseline_ARI_mean:.3f} and {dwell_4s.baseline_ARI_mean:.3f}, respectively. The 1 s target improves mean development log-likelihood by {dwell_1s.validation_ll_delta_mean:+.4f} per point but lowers within-encoder ARI to {dwell_1s.within_encoder_ARI_mean:.3f}; the 4 s target changes likelihood by {dwell_4s.validation_ll_delta_mean:+.4f} and leaves within-encoder ARI at {dwell_4s.within_encoder_ARI_mean:.3f}.",
        "",
        f"Stronger regularization improves conditional HMM reproducibility modestly but does not resolve representation variation: within-encoder ARI rises from {weak.within_encoder_ARI_mean:.3f} at strength 10 to {baseline.within_encoder_ARI_mean:.3f} at 50000, whereas cross-encoder ARI remains only {baseline.cross_encoder_ARI_mean:.3f}.",
        "",
        "The six-state solutions are not evaluated as biological ground truth. ARI, likelihood, occupancy and fragmentation are complementary diagnostics, and A4_C4 remains reused development data.",
        "",
        "## 2. Transition-prior implementation audit",
        "",
        "The HMM uses a row-wise Dirichlet pseudocount prior in every EM M-step. Strength is total sticky mass per row, distributed according to the target transition matrix; it is not diagonal-only mass. Forward-backward soft expected transition counts are added independently for every valid run, then rows are normalized. See `prior_implementation_audit.md` and `transition_prior_vs_counts.csv` for the exact audit.",
        "",
        f"At 50000/2 s, each row receives 50000.06 total prior pseudocounts (50000 sticky plus six 0.01 isotropic counts). The fitted models contain a mean {baseline_leverage.expected_data_row_transitions.mean():.1f} posterior-expected data transitions per source-state row. Prior/data row ratios average {baseline_leverage.prior_to_data_ratio_row.mean():.3f} and range from {baseline_leverage.prior_to_data_ratio_row.min():.3f} to {baseline_leverage.prior_to_data_ratio_row.max():.3f}; the prior therefore has substantial, state-dependent leverage rather than acting as a negligible initializer.",
        "",
        "## 3. Experiment A: strength sensitivity",
        "",
        *_markdown_table(
            strength,
            [
                ("prior_strength", "strength"),
                ("learned_self_transition_mean", "learned pself"),
                ("bouts_per_minute_mean", "bouts/min"),
                ("median_bout_s_mean", "median bout s"),
                ("validation_ll_delta_mean", "dev LL delta"),
                ("baseline_ARI_mean", "ARI vs 50000"),
                ("within_encoder_ARI_mean", "within ARI"),
                ("cross_encoder_ARI_mean", "cross ARI"),
            ],
        ),
        "",
        "## 4. State partition robustness",
        "",
        "`baseline_partition_correspondence.csv` reports matched-seed comparisons against 50000/2 s, while `pairwise_strength_ari.csv` and the seed-specific heatmaps expose the full strength-by-strength geometry. Hungarian matches are used only to summarize overlap quality; they are not treated as proof that two states are identical. Split/merge flags use a documented 20% substantial-component threshold.",
        "",
        "## 5. Probability fit and occupancy",
        "",
        f"Across the strength sweep, entropy-effective state count ranged from {strength.effective_states_mean.min():.3f} to {strength.effective_states_mean.max():.3f}, and mean largest-state occupancy ranged from {strength.largest_state_mean.min():.3f} to {strength.largest_state_mean.max():.3f}. Development likelihood changes are always calculated within the same encoder seed, HMM seed, and frozen PCA space before aggregation.",
        "",
        "## 6. Random-seed decomposition",
        "",
        f"Across the six 2 s strength settings, matched-downstream-seed cross-encoder ARI ranged from {cross_by_strength.min():.3f} to {cross_by_strength.max():.3f}. "
        "Within-encoder and cross-encoder comparisons are kept separate because overlapping ARI pairs are descriptive and not independent biological replicates.",
        "",
        "## 7. Experiment B: dwell-target sensitivity",
        "",
        *_markdown_table(
            dwell,
            [
                ("prior_strength", "strength"),
                ("target_dwell_s", "target dwell s"),
                ("learned_self_transition_mean", "learned pself"),
                ("validation_ll_delta_mean", "dev LL delta"),
                ("bouts_per_minute_mean", "bouts/min"),
                ("baseline_ARI_mean", "ARI vs baseline"),
                ("within_encoder_ARI_mean", "within ARI"),
                ("cross_encoder_ARI_mean", "cross ARI"),
            ],
        ),
        "",
        "## 8. Limitations",
        "",
        "- A4_C4 has been reused for encoder development and prior calibration; it is not an untouched test.",
        "- Temporal patches and pairwise ARI comparisons are correlated, so no naive patch-level or pair-level inferential statistics are reported.",
        "- Absolute Gaussian log densities are not compared across encoder/PCA spaces; only within-seed deltas are aggregated.",
        "- Stability, fewer bouts, and uniform occupancy do not establish behavioral correctness.",
        "- No independent behavioral labels or new animal pair are available for biological validation.",
        "",
        "## 9. Recommendation",
        "",
        f"Retain the current **50000 / 2 s** setting for the next development stage. It lies inside the 25000-100000 partition-stability plateau, preserves all six occupied states, gives substantially less fragmentation than weak priors, and has within-encoder ARI {baseline.within_encoder_ARI_mean:.3f}. Strength 100000 provides no meaningful stability gain and reduces development likelihood by {strong.validation_ll_delta_mean:+.4f} per point. A 1 s target has slightly better development likelihood but lower within-encoder reproducibility, while 4 s is nearly partition-equivalent without a compelling advantage. This is a development-setting recommendation, not a biological optimum; encoder-seed instability remains the larger unresolved limitation.",
        "",
        "Reproducible run-level evidence is in `strength_sweep_runs.csv`, `dwell_grid_runs.csv`, `transition_prior_vs_counts.csv`, `baseline_partition_correspondence.csv`, `within_encoder_stability.csv`, and `cross_encoder_stability.csv`.",
    ]
    (output / "sticky_prior_sensitivity_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_sensitivity_analysis(
    *,
    embedding_root: str | Path,
    baseline_root: str | Path,
    output_dir: str | Path,
    workers: int = 3,
    max_iter: int = 40,
    tolerance: float = 1e-4,
    covariance_floor: float = 1e-3,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    embedding_root = Path(embedding_root).resolve()
    baseline_root = Path(baseline_root).resolve()
    thresholds = CorrespondenceThresholds(component_fraction=0.20)
    payloads = [
        {
            "encoder_seed": seed,
            "embedding_dir": str(embedding_root / f"seed_{seed}"),
            "baseline_dir": str(baseline_root / f"seed_{seed}"),
            "output_root": str(output),
            "max_iter": max_iter,
            "tolerance": tolerance,
            "covariance_floor": covariance_floor,
        }
        for seed in ENCODER_SEEDS
    ]
    results: list[dict[str, Any]] = []
    if workers == 1:
        results = [_analyze_one_encoder(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(payloads))) as executor:
            futures = {executor.submit(_analyze_one_encoder, value): value for value in payloads}
            for future in as_completed(futures):
                results.append(future.result())

    run_rows = [row for result in results for row in result["runs"]]
    transition_rows = [
        row for result in results for row in result["transition_rows"]
    ]
    state_duration_rows = [
        row for result in results for row in result["state_duration_rows"]
    ]
    runs = pd.DataFrame(run_rows).sort_values(
        ["target_dwell_s", "prior_strength", "encoder_seed", "downstream_seed"]
    )
    runs = _add_likelihood_deltas(runs)
    runs.to_csv(output / "all_runs.csv", index=False)
    runs[runs.target_dwell_s.eq(2.0)].to_csv(
        output / "strength_sweep_runs.csv", index=False
    )
    runs[runs.prior_strength.isin(DWELL_GRID_STRENGTHS)].to_csv(
        output / "dwell_grid_runs.csv", index=False
    )
    pd.DataFrame(transition_rows).to_csv(
        output / "transition_prior_vs_counts.csv", index=False
    )
    pd.DataFrame(state_duration_rows).to_csv(
        output / "temporal_metrics.csv", index=False
    )
    occupancy_columns = [
        "encoder_seed",
        "downstream_seed",
        "prior_strength",
        "target_dwell_s",
        "occupied_states",
        "effective_entropy_states",
        "largest_state_occupancy",
        "smallest_nonempty_state_occupancy",
        "mean_posterior_entropy",
        "occupancy_distribution_json",
    ]
    runs[occupancy_columns].to_csv(output / "occupancy_metrics.csv", index=False)
    likelihood_columns = [
        "encoder_seed",
        "downstream_seed",
        "prior_strength",
        "target_dwell_s",
        "train_log_likelihood_per_point",
        "validation_log_likelihood_per_point",
        "train_log_likelihood_delta_vs_50000_2s",
        "validation_log_likelihood_delta_vs_50000_2s",
        "converged",
        "iterations",
    ]
    runs[likelihood_columns].to_csv(output / "likelihood_comparison.csv", index=False)

    failures = runs[~runs.status.eq("completed")]
    if not failures.empty:
        failures.to_csv(output / "failed_runs.csv", index=False)
        raise RuntimeError(f"sticky sensitivity has {failures.shape[0]} failed runs")
    comparisons = _compute_comparisons(runs, output, thresholds)
    strength_summary = _strength_summary(runs, comparisons)
    dwell_summary = _dwell_summary(runs, comparisons)
    strength_summary.to_csv(output / "strength_summary.csv", index=False)
    dwell_summary.to_csv(output / "dwell_grid_summary.csv", index=False)

    _plot_strength_metrics(runs, figures)
    _plot_likelihood(runs, figures)
    _plot_ari(strength_summary, comparisons["baseline"], figures)
    _plot_pairwise_heatmaps(comparisons["pairwise_strength"], figures)
    _plot_occupancy(runs, figures)
    _plot_dwell_grid(dwell_summary, figures)
    _write_audit(output)
    _write_report(output, runs, strength_summary, dwell_summary, comparisons)

    manifest = {
        "analysis": "sticky_prior_sensitivity_v1",
        "status": "completed",
        "output_dir": str(output),
        "encoder": "graph_full",
        "encoder_seeds": list(ENCODER_SEEDS),
        "downstream_seeds": list(DOWNSTREAM_SEEDS),
        "train_trials": list(TRAIN_TRIALS),
        "development_trial": VALIDATION_TRIAL,
        "excluded_trial": EXCLUDED_TRIAL,
        "timestep_s": TIMESTEP_S,
        "K": N_STATES,
        "pca_dim": PCA_DIM,
        "strengths": list(STRENGTHS),
        "dwell_grid_strengths": list(DWELL_GRID_STRENGTHS),
        "dwell_targets_s": list(DWELL_TARGETS_S),
        "unique_configurations": len(sensitivity_grid()) * len(ENCODER_SEEDS) * len(DOWNSTREAM_SEEDS),
        "completed_configurations": int(runs.status.eq("completed").sum()),
        "reused_formal_baseline_configurations": 9,
        "reused_or_resumed_configurations": int(runs.reused_existing_artifact.sum()),
        "max_iter": max_iter,
        "tolerance": tolerance,
        "covariance_floor": covariance_floor,
        "comparison_thresholds": {
            "substantial_split_merge_component_fraction": thresholds.component_fraction,
            "ambiguity_margin": thresholds.ambiguity_margin,
            "weak_match_f1": thresholds.weak_f1,
            "stable_match_f1": thresholds.stable_f1,
            "defensible_jaccard": thresholds.defensible_jaccard,
        },
        "embedding_manifest_sha256_by_encoder_seed": {
            str(result["encoder_seed"]): result["embedding_manifest_sha256"]
            for result in results
        },
        "preprocessor_sha256_by_encoder_seed": {
            str(result["encoder_seed"]): result["preprocessor_sha256"]
            for result in results
        },
        "notes": [
            "No encoder or PCA was retrained.",
            "The nine formal 50000/2 s checkpoints were validated and referenced, not overwritten.",
            "Every invalid-gap-delimited run and every trial remains a separate HMM sequence.",
            "Likelihood deltas are matched within encoder seed and downstream seed before aggregation.",
            "Pairwise partition comparisons are dependent descriptive diagnostics.",
        ],
    }
    _write_json(output / "experiment_manifest.json", manifest)
    return manifest


def aggregate_existing_sensitivity(output_dir: str | Path) -> dict[str, Any]:
    """Resume the deterministic comparison/report stage from complete run CSVs."""

    output = Path(output_dir).resolve()
    runs_path = output / "all_runs.csv"
    if not runs_path.is_file():
        raise FileNotFoundError(runs_path)
    runs = pd.read_csv(runs_path)
    expected = len(sensitivity_grid()) * len(ENCODER_SEEDS) * len(DOWNSTREAM_SEEDS)
    if runs.shape[0] != expected or not runs.status.eq("completed").all():
        raise ValueError(
            f"aggregate-only requires {expected} completed runs, found "
            f"{int(runs.status.eq('completed').sum())}/{runs.shape[0]}"
        )
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    thresholds = CorrespondenceThresholds(component_fraction=0.20)
    comparisons = _compute_comparisons(runs, output, thresholds)
    strength_summary = _strength_summary(runs, comparisons)
    dwell_summary = _dwell_summary(runs, comparisons)
    strength_summary.to_csv(output / "strength_summary.csv", index=False)
    dwell_summary.to_csv(output / "dwell_grid_summary.csv", index=False)
    _plot_strength_metrics(runs, figures)
    _plot_likelihood(runs, figures)
    _plot_ari(strength_summary, comparisons["baseline"], figures)
    _plot_pairwise_heatmaps(comparisons["pairwise_strength"], figures)
    _plot_occupancy(runs, figures)
    _plot_dwell_grid(dwell_summary, figures)
    _write_audit(output)
    _write_report(output, runs, strength_summary, dwell_summary, comparisons)
    manifest = {
        "analysis": "sticky_prior_sensitivity_v1",
        "status": "completed",
        "resumed_aggregate_only": True,
        "output_dir": str(output),
        "encoder": "graph_full",
        "encoder_seeds": list(ENCODER_SEEDS),
        "downstream_seeds": list(DOWNSTREAM_SEEDS),
        "train_trials": list(TRAIN_TRIALS),
        "development_trial": VALIDATION_TRIAL,
        "excluded_trial": EXCLUDED_TRIAL,
        "timestep_s": TIMESTEP_S,
        "K": N_STATES,
        "pca_dim": PCA_DIM,
        "strengths": list(STRENGTHS),
        "dwell_grid_strengths": list(DWELL_GRID_STRENGTHS),
        "dwell_targets_s": list(DWELL_TARGETS_S),
        "unique_configurations": expected,
        "completed_configurations": int(runs.status.eq("completed").sum()),
        "reused_formal_baseline_configurations": 9,
        "reused_or_resumed_configurations": int(
            runs.reused_existing_artifact.astype(bool).sum()
        ),
        "max_iter": 40,
        "tolerance": 1e-4,
        "covariance_floor": 1e-3,
        "comparison_thresholds": {
            "substantial_split_merge_component_fraction": thresholds.component_fraction,
            "ambiguity_margin": thresholds.ambiguity_margin,
            "weak_match_f1": thresholds.weak_f1,
            "stable_match_f1": thresholds.stable_f1,
            "defensible_jaccard": thresholds.defensible_jaccard,
        },
        "embedding_manifest_sha256_by_encoder_seed": {
            str(int(seed)): str(group.embedding_manifest_sha256.iloc[0])
            for seed, group in runs.groupby("encoder_seed")
        },
        "preprocessor_sha256_by_encoder_seed": {
            str(int(seed)): str(group.preprocessor_sha256.iloc[0])
            for seed, group in runs.groupby("encoder_seed")
        },
        "notes": [
            "No encoder or PCA was retrained.",
            "The nine formal 50000/2 s checkpoints were validated and referenced, not overwritten.",
            "Aggregate-only resumed after all 90 run-level artifacts had completed.",
            "Every invalid-gap-delimited run and every trial remains a separate HMM sequence.",
            "Likelihood deltas are matched within encoder seed and downstream seed before aggregation.",
        ],
    }
    _write_json(output / "experiment_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-root")
    parser.add_argument("--baseline-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--covariance-floor", type=float, default=1e-3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.aggregate_only:
        manifest = aggregate_existing_sensitivity(args.output_dir)
    else:
        if not args.embedding_root or not args.baseline_root:
            raise ValueError("--embedding-root and --baseline-root are required unless --aggregate-only")
        manifest = run_sensitivity_analysis(
            embedding_root=args.embedding_root,
            baseline_root=args.baseline_root,
            output_dir=args.output_dir,
            workers=args.workers,
            max_iter=args.max_iter,
            tolerance=args.tolerance,
            covariance_floor=args.covariance_floor,
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
