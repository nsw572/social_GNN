"""Formal HMM state-number selection on frozen 0.04 s Social-GNN embeddings.

The analysis deliberately freezes encoder embeddings and each encoder seed's
train-fitted PCA16 transform.  HMMs are fit independently for every K and
downstream seed; no warm starts or state-count continuation are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from itertools import combinations, product
from pathlib import Path
from threading import Event, Thread
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from threadpoolctl import threadpool_limits

from .embeddings import (
    FrozenEmbeddingTrial,
    load_embedding_directory,
    validate_embedding_timestep,
)
from .evaluation import (
    DecodedTrial,
    bout_rows,
    decode_trial,
    segmentation_metrics,
    transition_counts,
)
from .gaussian_hmm import GaussianHMM
from .preprocessing import EmbeddingPreprocessor
from .state_correspondence import (
    CorrespondenceThresholds,
    boundary_agreement,
    hungarian_state_matches,
    overlap_metrics,
    split_merge_diagnostics,
)


TIMESTEP_S = 0.04
PCA_DIM = 16
TARGET_DWELL_S = 2.0
FIXED_PRIOR_STRENGTH = 50000.0
REFERENCE_K = 6
K_GRID = (4, 5, 6, 7, 8, 10, 12)
ENCODER_SEEDS = (20261013, 20261029, 20261107)
HMM_SEEDS = (0, 1, 2)
ARCHITECTURES = ("graph_full", "invariant_mlp_full")
TRAIN_TRIALS = ("A1_C1", "A3_C3", "A4_B4")
DEV_TRIAL = "A4_C4"
EXCLUDED_TRIAL = "A5_C5"
ADJACENT_K_PAIRS = tuple(zip(K_GRID[:-1], K_GRID[1:]))


class _PeakRssMonitor:
    """Best-effort native RSS sampler for one model fit/resume operation."""

    def __init__(self, interval_s: float = 0.10) -> None:
        self.interval_s = float(interval_s)
        self.peak_bytes = 0
        self._stop = Event()
        self._thread: Thread | None = None
        self._process = None

    def __enter__(self) -> "_PeakRssMonitor":
        try:
            import psutil

            self._process = psutil.Process()
            self.peak_bytes = int(self._process.memory_info().rss)
            self._thread = Thread(target=self._sample, daemon=True)
            self._thread.start()
        except Exception:
            self._process = None
        return self

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.peak_bytes = max(
                    self.peak_bytes, int(self._process.memory_info().rss)
                )
            except Exception:
                return

    def __exit__(self, *_error: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 5.0 * self.interval_s))
        if self._process is not None:
            try:
                self.peak_bytes = max(
                    self.peak_bytes, int(self._process.memory_info().rss)
                )
            except Exception:
                pass

    @property
    def peak_rss_mb(self) -> float:
        return float(self.peak_bytes / (1024.0 * 1024.0)) if self.peak_bytes else float("nan")


@dataclass(frozen=True)
class OccupancyThresholds:
    """Transparent warning thresholds; they do not select K by themselves."""

    near_empty_fraction: float = 0.005
    small_state_fraction: float = 0.01
    minimum_effective_ratio: float = 0.50
    maximum_largest_fraction: float = 0.80

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")


def hmm_parameter_count(k: int, dimension: int = PCA_DIM) -> int:
    """Free parameters for a diagonal-Gaussian K-state HMM."""

    if k < 2 or dimension < 1:
        raise ValueError("k must be >=2 and dimension must be positive")
    return int((k - 1) + k * (k - 1) + 2 * k * dimension)


def likelihood_per_point(total_log_likelihood: float, observations: int) -> float:
    if observations <= 0:
        raise ValueError("observations must be positive")
    return float(total_log_likelihood) / int(observations)


def information_criteria(
    total_train_log_likelihood: float,
    *,
    k: int,
    dimension: int,
    observations: int,
) -> dict[str, float | int]:
    """Conventional AIC/BIC using actual valid training observations."""

    parameters = hmm_parameter_count(k, dimension)
    if observations <= 0:
        raise ValueError("observations must be positive")
    return {
        "parameter_count": parameters,
        "aic_train": float(-2.0 * total_train_log_likelihood + 2.0 * parameters),
        "bic_train": float(
            -2.0 * total_train_log_likelihood
            + math.log(float(observations)) * parameters
        ),
    }


def prior_strength_for_k(policy: str, k: int) -> float:
    if k < 2:
        raise ValueError("k must be >=2")
    if policy == "fixed":
        return FIXED_PRIOR_STRENGTH
    if policy == "normalized_6_over_k":
        return FIXED_PRIOR_STRENGTH * REFERENCE_K / k
    raise ValueError(f"unknown prior policy {policy!r}")


def rectangular_contingency(
    left_labels: np.ndarray,
    right_labels: np.ndarray,
    left_k: int,
    right_k: int,
) -> np.ndarray:
    """Rectangular co-assignment counts for partitions with different K."""

    left = np.asarray(left_labels, dtype=np.int64)
    right = np.asarray(right_labels, dtype=np.int64)
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError("state arrays must be equal-length one-dimensional arrays")
    if left.size == 0:
        raise ValueError("state arrays must not be empty")
    if np.any((left < 0) | (left >= left_k)):
        raise ValueError("left labels outside declared K")
    if np.any((right < 0) | (right >= right_k)):
        raise ValueError("right labels outside declared K")
    matrix = np.zeros((left_k, right_k), dtype=np.int64)
    np.add.at(matrix, (left, right), 1)
    return matrix


def _entropy(probability: np.ndarray) -> float:
    positive = np.asarray(probability, dtype=float)
    positive = positive[positive > 0]
    return float(-np.sum(positive * np.log(positive)))


def rectangular_overlap_summary(matrix: np.ndarray) -> dict[str, float]:
    """Label-invariant overlap, conditional entropy, and VI diagnostics."""

    counts = np.asarray(matrix, dtype=np.float64)
    if counts.ndim != 2 or counts.size == 0 or counts.sum() <= 0:
        raise ValueError("contingency matrix must contain observations")
    total = float(counts.sum())
    joint = counts / total
    left_p = joint.sum(axis=1)
    right_p = joint.sum(axis=0)
    h_left = _entropy(left_p)
    h_right = _entropy(right_p)
    nonzero = joint > 0
    independent = left_p[:, None] * right_p[None, :]
    mutual = float(np.sum(joint[nonzero] * np.log(joint[nonzero] / independent[nonzero])))
    h_right_given_left = h_right - mutual
    h_left_given_right = h_left - mutual
    left_purity = float(counts.max(axis=1).sum() / total)
    right_coverage = float(counts.max(axis=0).sum() / total)
    row, col = linear_sum_assignment(-counts)
    return {
        "left_to_right_purity": left_purity,
        "right_to_left_coverage": right_coverage,
        "conditional_entropy_right_given_left": h_right_given_left,
        "conditional_entropy_left_given_right": h_left_given_right,
        "variation_of_information": h_left + h_right - 2.0 * mutual,
        "mutual_information_nats": mutual,
        "hungarian_matched_patch_fraction": float(counts[row, col].sum() / total),
    }


def occupancy_metrics(
    labels: np.ndarray,
    posterior: np.ndarray,
    k: int,
    *,
    thresholds: OccupancyThresholds | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    thresholds = thresholds or OccupancyThresholds()
    state = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(posterior, dtype=np.float64)
    if state.ndim != 1 or probability.shape != (state.size, k):
        raise ValueError("labels/posterior shapes do not match K")
    if state.size == 0 or np.any((state < 0) | (state >= k)):
        raise ValueError("occupancy requires nonempty valid labels")
    counts = np.bincount(state, minlength=k).astype(float)
    hard = counts / counts.sum()
    posterior_occupancy = probability.sum(axis=0)
    posterior_occupancy /= max(float(posterior_occupancy.sum()), 1.0)
    entropy = _entropy(hard)
    effective = float(np.exp(entropy))
    positive = hard[hard > 0]
    flags: list[str] = []
    if np.any(hard == 0):
        flags.append("unoccupied_state")
    if np.any((hard > 0) & (hard < thresholds.near_empty_fraction)):
        flags.append("near_empty_state")
    if effective / k < thresholds.minimum_effective_ratio:
        flags.append("low_effective_state_ratio")
    if hard.max() > thresholds.maximum_largest_fraction:
        flags.append("dominant_state")
    summary = {
        "occupied_states": int(np.sum(hard > 0)),
        "effective_states": effective,
        "effective_state_ratio": effective / k,
        "occupancy_entropy_nats": entropy,
        "largest_state_occupancy": float(hard.max()),
        "smallest_nonempty_state_occupancy": float(positive.min()) if positive.size else 0.0,
        "full_occupancy": bool(np.all(hard > 0)),
        "states_below_one_percent": int(np.sum((hard > 0) & (hard < thresholds.small_state_fraction))),
        "states_below_near_empty_threshold": int(
            np.sum((hard > 0) & (hard < thresholds.near_empty_fraction))
        ),
        "occupancy_warning": "|".join(flags),
    }
    rows = [
        {
            "state": state_id,
            "hard_count": int(counts[state_id]),
            "hard_occupancy": float(hard[state_id]),
            "posterior_occupancy": float(posterior_occupancy[state_id]),
            "near_empty": bool(
                0 < hard[state_id] < thresholds.near_empty_fraction
            ),
        }
        for state_id in range(k)
    ]
    return summary, rows


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _sequences(
    trials: Iterable[FrozenEmbeddingTrial], preprocessor: EmbeddingPreprocessor
) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for trial in trials:
        transformed, _indices = preprocessor.transform_trial_sequences(trial)
        result.extend(transformed)
    return result


def _save_assignment(path: Path, decoded: DecodedTrial) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        trial_id=np.asarray(decoded.trial.trial_id),
        patch_start_s=decoded.trial.patch_start_s,
        patch_end_s=decoded.trial.patch_end_s,
        valid_time_mask=decoded.trial.valid_time_mask,
        predicted_state=decoded.predicted_state,
        state_posterior=decoded.posterior.astype(np.float32),
        state_confidence=decoded.state_confidence.astype(np.float32),
    )
    return path


def _load_assignment(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = ("patch_start_s", "patch_end_s", "valid_time_mask", "predicted_state")
        missing = set(required) - set(data.files)
        if missing:
            raise ValueError(f"{path}: missing assignment arrays {sorted(missing)}")
        return {key: np.asarray(data[key]) for key in required}


def _validate_assignment(
    assignment: dict[str, np.ndarray], trial: FrozenEmbeddingTrial, source: str | Path
) -> None:
    if assignment["predicted_state"].shape != trial.valid_time_mask.shape:
        raise ValueError(f"{source}: assignment length mismatch")
    if not np.array_equal(assignment["valid_time_mask"], trial.valid_time_mask):
        raise ValueError(f"{source}: valid mask mismatch")
    for key, expected in (
        ("patch_start_s", trial.patch_start_s),
        ("patch_end_s", trial.patch_end_s),
    ):
        if not np.allclose(assignment[key], expected, rtol=0.0, atol=1e-9):
            raise ValueError(f"{source}: authoritative clock mismatch for {key}")


def _validate_model_contract(
    model: GaussianHMM,
    metadata: dict[str, Any],
    *,
    architecture: str,
    encoder_seed: int,
    hmm_seed: int,
    k: int,
    strength: float,
    embedding_dir: Path,
    preprocessor_path: Path,
) -> None:
    checks = {
        "n_states": (model.n_states, k),
        "seed": (model.seed, hmm_seed),
        "timestep_s": (model.timestep_s, TIMESTEP_S),
        "sticky_expected_duration_s": (model.sticky_expected_duration_s, TARGET_DWELL_S),
        "sticky_prior_strength": (model.sticky_prior_strength, strength),
        "n_features": (model.n_features, PCA_DIM),
    }
    failed = [name for name, (actual, expected) in checks.items() if actual != expected]
    if failed:
        raise ValueError(f"checkpoint contract mismatch: {failed}")
    if metadata:
        if str(metadata.get("encoder", architecture)) != architecture:
            raise ValueError("checkpoint architecture mismatch")
        if int(metadata.get("encoder_seed", encoder_seed)) != encoder_seed:
            raise ValueError("checkpoint encoder seed mismatch")
        if Path(metadata.get("embedding_dir", embedding_dir)).resolve() != embedding_dir:
            raise ValueError("checkpoint embedding directory mismatch")
        if Path(metadata.get("preprocessor", preprocessor_path)).resolve() != preprocessor_path:
            raise ValueError("checkpoint PCA path mismatch")


def _baseline_dir(root: Path, architecture: str, encoder_seed: int) -> Path:
    if architecture == "graph_full":
        return (
            root
            / "evaluation_runs"
            / "gnn_9dim_0p04_downstream_v2"
            / "sticky_hmm_strength50000"
            / architecture
            / f"seed_{encoder_seed}"
        )
    return (
        root
        / "evaluation_runs"
        / "gnn_9dim_0p04_downstream_v2"
        / architecture
        / f"seed_{encoder_seed}"
    )


def _embedding_dir(root: Path, architecture: str, encoder_seed: int) -> Path:
    return (
        root
        / "evaluation_runs"
        / "gnn_9dim_0p04_v2"
        / "embeddings"
        / architecture
        / f"seed_{encoder_seed}"
    )


def _decode_many(
    model: GaussianHMM,
    preprocessor: EmbeddingPreprocessor,
    trials: Sequence[FrozenEmbeddingTrial],
) -> list[DecodedTrial]:
    return [decode_trial(model, preprocessor, trial) for trial in trials]


def _temporal_rows(
    decoded: Sequence[DecodedTrial], k: int
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    metrics = segmentation_metrics(decoded, k)
    bouts = [row for trial in decoded for row in bout_rows(trial)]
    duration = np.asarray([row["bout_duration_s"] for row in bouts], dtype=float)
    timestep = np.asarray([row["timesteps"] for row in bouts], dtype=int)
    metrics.update(
        {
            "fraction_bouts_shorter_0p20_s": float(np.mean(duration < 0.20)),
            "fraction_bouts_shorter_0p50_s": float(np.mean(duration < 0.50)),
            "fraction_one_step_bouts": float(np.mean(timestep == 1)),
        }
    )
    state_rows: list[dict[str, Any]] = []
    valid_labels = np.concatenate(
        [value.predicted_state[value.predicted_state >= 0] for value in decoded]
    )
    valid_posterior = np.concatenate(
        [value.posterior[value.trial.valid_time_mask] for value in decoded], axis=0
    )
    occupancy_summary, occupancy_rows = occupancy_metrics(valid_labels, valid_posterior, k)
    metrics.update(occupancy_summary)
    for state in range(k):
        selected = duration[
            np.asarray([int(row["state"]) == state for row in bouts], dtype=bool)
        ]
        state_rows.append(
            {
                "state": state,
                "hard_occupancy": float(occupancy_rows[state]["hard_occupancy"]),
                "posterior_occupancy": float(
                    occupancy_rows[state]["posterior_occupancy"]
                ),
                "bout_count": int(selected.size),
                "mean_bout_duration_s": float(selected.mean()) if selected.size else float("nan"),
                "median_bout_duration_s": float(np.median(selected)) if selected.size else float("nan"),
                "p10_bout_duration_s": float(np.quantile(selected, 0.10)) if selected.size else float("nan"),
                "p90_bout_duration_s": float(np.quantile(selected, 0.90)) if selected.size else float("nan"),
            }
        )
    return metrics, occupancy_rows, state_rows, transition_counts(decoded, k)


def _annotate_rows(rows: list[dict[str, Any]], identifiers: dict[str, Any]) -> None:
    for row in rows:
        row.update(identifiers)


def _baseline_artifacts(
    baseline: Path, hmm_seed: int
) -> tuple[Path, Path]:
    run = baseline / "hmm" / f"hmm_k{REFERENCE_K}_seed{hmm_seed}"
    return run / "model.npz", run / "assignments" / f"{DEV_TRIAL}_states.npz"


def _analyze_worker(payload: dict[str, Any]) -> dict[str, Any]:
    root = Path(payload["root"]).resolve()
    output = Path(payload["output"]).resolve()
    architecture = str(payload["architecture"])
    encoder_seed = int(payload["encoder_seed"])
    k_values = tuple(int(value) for value in payload["k_values"])
    policy = str(payload.get("policy", "fixed"))
    max_iter = int(payload.get("max_iter", 40))
    tolerance = float(payload.get("tolerance", 1e-4))
    covariance_floor = float(payload.get("covariance_floor", 1e-3))
    hmm_seeds = tuple(int(value) for value in payload.get("hmm_seeds", HMM_SEEDS))
    embedding_dir = _embedding_dir(root, architecture, encoder_seed)
    baseline = _baseline_dir(root, architecture, encoder_seed)
    preprocessor_path = baseline / "preprocessing" / f"pca_{PCA_DIM}.npz"
    trials = load_embedding_directory(embedding_dir)
    validate_embedding_timestep(trials, TIMESTEP_S)
    by_id = {trial.trial_id: trial for trial in trials}
    missing = (set(TRAIN_TRIALS) | {DEV_TRIAL}) - set(by_id)
    if missing:
        raise ValueError(f"missing required trials {sorted(missing)}")
    train_trials = [by_id[name] for name in TRAIN_TRIALS]
    dev_trial = by_id[DEV_TRIAL]
    preprocessor = EmbeddingPreprocessor.load(preprocessor_path)
    if preprocessor.output_dim != PCA_DIM:
        raise ValueError(f"expected frozen PCA{PCA_DIM}")
    train_sequences = _sequences(train_trials, preprocessor)
    dev_sequences = _sequences([dev_trial], preprocessor)
    train_n = int(sum(value.shape[0] for value in train_sequences))
    dev_n = int(sum(value.shape[0] for value in dev_sequences))
    expected_transitions = int(sum(max(value.shape[0] - 1, 0) for value in train_sequences))
    embedding_manifest = embedding_dir / "manifest.json"
    embedding_sha = _sha256(embedding_manifest)
    pca_sha = _sha256(preprocessor_path)

    collections: dict[str, list[dict[str, Any]]] = {
        "runs": [],
        "occupancy": [],
        "temporal": [],
        "per_state": [],
        "transitions": [],
        "prior": [],
        "failures": [],
    }
    reused = 0
    for k in k_values:
        strength = prior_strength_for_k(policy, k)
        for hmm_seed in hmm_seeds:
            started = time.perf_counter()
            run_dir = (
                output
                / "runs"
                / policy
                / architecture
                / f"encoder_seed_{encoder_seed}"
                / f"k_{k}"
                / f"hmm_seed_{hmm_seed}"
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            identifiers = {
                "architecture": architecture,
                "encoder_seed": encoder_seed,
                "hmm_seed": hmm_seed,
                "K": k,
                "prior_policy": policy,
                "prior_strength": float(strength),
            }
            memory = _PeakRssMonitor()
            try:
                memory.__enter__()
                exact_baseline = (
                    k == REFERENCE_K
                    and math.isclose(strength, FIXED_PRIOR_STRENGTH)
                    and math.isclose(TARGET_DWELL_S, 2.0)
                )
                reused_this_run = False
                if exact_baseline:
                    model_path, original_assignment = _baseline_artifacts(baseline, hmm_seed)
                    model, metadata = GaussianHMM.load(model_path)
                    _validate_model_contract(
                        model,
                        metadata,
                        architecture=architecture,
                        encoder_seed=encoder_seed,
                        hmm_seed=hmm_seed,
                        k=k,
                        strength=strength,
                        embedding_dir=embedding_dir,
                        preprocessor_path=preprocessor_path,
                    )
                    saved = _load_assignment(original_assignment)
                    _validate_assignment(saved, dev_trial, original_assignment)
                    dev_decoded = decode_trial(model, preprocessor, dev_trial)
                    if not np.array_equal(saved["predicted_state"], dev_decoded.predicted_state):
                        raise ValueError("reused K=6 checkpoint disagrees with saved assignment")
                    assignment_path = original_assignment
                    reused += 1
                    reused_this_run = True
                    _write_json(
                        run_dir / "baseline_reference.json",
                        {
                            "model": str(model_path),
                            "model_sha256": _sha256(model_path),
                            "assignment": str(original_assignment),
                            "assignment_sha256": _sha256(original_assignment),
                            "preprocessor": str(preprocessor_path),
                            "preprocessor_sha256": pca_sha,
                        },
                    )
                else:
                    model_path = run_dir / "model.npz"
                    assignment_path = run_dir / f"{DEV_TRIAL}_states.npz"
                    if model_path.exists() and assignment_path.exists():
                        model, metadata = GaussianHMM.load(model_path)
                        _validate_model_contract(
                            model,
                            metadata,
                            architecture=architecture,
                            encoder_seed=encoder_seed,
                            hmm_seed=hmm_seed,
                            k=k,
                            strength=strength,
                            embedding_dir=embedding_dir,
                            preprocessor_path=preprocessor_path,
                        )
                        saved = _load_assignment(assignment_path)
                        _validate_assignment(saved, dev_trial, assignment_path)
                        dev_decoded = decode_trial(model, preprocessor, dev_trial)
                        if not np.array_equal(saved["predicted_state"], dev_decoded.predicted_state):
                            raise ValueError("resumed model and assignment disagree")
                        reused += 1
                        reused_this_run = True
                    else:
                        model = GaussianHMM(
                            n_states=k,
                            covariance_floor=covariance_floor,
                            timestep_s=TIMESTEP_S,
                            sticky_expected_duration_s=TARGET_DWELL_S,
                            sticky_prior_strength=strength,
                            seed=hmm_seed,
                        )
                        with threadpool_limits(limits=1):
                            model.fit(train_sequences, max_iter=max_iter, tolerance=tolerance)
                        metadata = {
                            "analysis": "formal_hmm_k_selection_0p04_v1",
                            "architecture": architecture,
                            "encoder": architecture,
                            "encoder_seed": encoder_seed,
                            "hmm_seed": hmm_seed,
                            "K": k,
                            "prior_policy": policy,
                            "embedding_dir": str(embedding_dir),
                            "embedding_manifest_sha256": embedding_sha,
                            "preprocessor": str(preprocessor_path),
                            "preprocessor_sha256": pca_sha,
                            "train_trials": list(TRAIN_TRIALS),
                            "dev_trial": DEV_TRIAL,
                            "excluded_trial": EXCLUDED_TRIAL,
                            "warm_start": False,
                        }
                        model.save(model_path, metadata=metadata)
                        dev_decoded = decode_trial(model, preprocessor, dev_trial)
                        _save_assignment(assignment_path, dev_decoded)

                train_decoded = _decode_many(model, preprocessor, train_trials)
                train_ll = float(model.score(train_sequences))
                dev_ll = float(model.score(dev_sequences))
                criteria = information_criteria(
                    train_ll, k=k, dimension=preprocessor.output_dim, observations=train_n
                )
                expected_counts = model.expected_transition_counts(train_sequences)
                if not np.isclose(
                    expected_counts.sum(), expected_transitions, rtol=1e-6, atol=1e-3
                ):
                    raise ValueError("soft xi counts include a sequence-boundary transition")
                prior = model.transition_prior_counts()
                train_metrics, train_occupancy, train_states, empirical_train = _temporal_rows(
                    train_decoded, k
                )
                dev_metrics, dev_occupancy, dev_states, empirical_dev = _temporal_rows(
                    [dev_decoded], k
                )
                train_bouts = [row for value in train_decoded for row in bout_rows(value)]
                dev_bouts = bout_rows(dev_decoded)
                pd.DataFrame(train_bouts).to_csv(run_dir / "train_bouts.csv", index=False)
                pd.DataFrame(dev_bouts).to_csv(run_dir / "dev_bouts.csv", index=False)
                np.savez_compressed(
                    run_dir / "transition_diagnostics.npz",
                    learned_transition=model.transition_,
                    prior_pseudocounts=prior,
                    expected_training_xi=expected_counts,
                    empirical_train_counts=empirical_train,
                    empirical_dev_counts=empirical_dev,
                )
                diagonal = np.diag(model.transition_)
                row = {
                    **identifiers,
                    "status": "completed",
                    "reused_existing_artifact": reused_this_run,
                    "train_valid_points": train_n,
                    "dev_valid_points": dev_n,
                    "train_log_likelihood_total": train_ll,
                    "train_log_likelihood_per_point": likelihood_per_point(train_ll, train_n),
                    "dev_log_likelihood_total": dev_ll,
                    "dev_log_likelihood_per_point": likelihood_per_point(dev_ll, dev_n),
                    **criteria,
                    "iterations": len(model.history_),
                    "converged": bool(model.converged_),
                    "final_em_history_log_likelihood": (
                        float(model.history_[-1]) if model.history_ else float("nan")
                    ),
                    "em_history_length": len(model.history_),
                    "target_self_transition": model.sticky_self_transition_probability,
                    "learned_self_transition_mean": float(diagonal.mean()),
                    "learned_self_transition_min": float(diagonal.min()),
                    "learned_self_transition_max": float(diagonal.max()),
                    "embedding_dir": str(embedding_dir),
                    "embedding_manifest_sha256": embedding_sha,
                    "preprocessor": str(preprocessor_path),
                    "preprocessor_sha256": pca_sha,
                    "checkpoint": str(model_path),
                    "assignment": str(assignment_path),
                    "elapsed_s": time.perf_counter() - started,
                    "peak_rss_mb": float("nan"),
                    "failure": "",
                }
                for prefix, values in (("train", train_metrics), ("dev", dev_metrics)):
                    for key, value in values.items():
                        row[f"{prefix}_{key}"] = value
                    collections["temporal"].append({**identifiers, "split": prefix, **values})
                for split, rows in (("train", train_occupancy), ("dev", dev_occupancy)):
                    _annotate_rows(rows, {**identifiers, "split": split})
                    collections["occupancy"].extend(rows)
                for split, rows in (("train", train_states), ("dev", dev_states)):
                    _annotate_rows(rows, {**identifiers, "split": split})
                    collections["per_state"].extend(rows)

                empirical_probability = np.divide(
                    empirical_train,
                    empirical_train.sum(axis=1, keepdims=True),
                    out=np.zeros_like(empirical_train),
                    where=empirical_train.sum(axis=1, keepdims=True) > 0,
                )
                prior_probability = prior / prior.sum(axis=1, keepdims=True)
                for source in range(k):
                    prior_row_mass = float(prior[source].sum())
                    xi_row_mass = float(expected_counts[source].sum())
                    source_bouts = [
                        value["bout_duration_s"]
                        for value in train_bouts
                        if int(value["state"]) == source
                    ]
                    collections["prior"].append(
                        {
                            **identifiers,
                            "source_state": source,
                            "prior_row_mass": prior_row_mass,
                            "prior_mean_self_transition": float(prior_probability[source, source]),
                            "posterior_expected_xi_row_mass": xi_row_mass,
                            "prior_to_xi_ratio": prior_row_mass / xi_row_mass if xi_row_mass else float("inf"),
                            "learned_self_transition": float(model.transition_[source, source]),
                            "empirical_self_transition": float(empirical_probability[source, source]),
                            "model_implied_mean_dwell_s": float(
                                TIMESTEP_S / max(1.0 - model.transition_[source, source], 1e-12)
                            ),
                            "observed_mean_bout_s": float(np.mean(source_bouts)) if source_bouts else float("nan"),
                            "observed_median_bout_s": float(np.median(source_bouts)) if source_bouts else float("nan"),
                        }
                    )
                    for target in range(k):
                        collections["transitions"].append(
                            {
                                **identifiers,
                                "source_state": source,
                                "target_state": target,
                                "is_diagonal": source == target,
                                "prior_pseudocount": float(prior[source, target]),
                                "prior_mean_probability": float(prior_probability[source, target]),
                                "posterior_expected_xi": float(expected_counts[source, target]),
                                "learned_transition_probability": float(model.transition_[source, target]),
                                "empirical_train_count": float(empirical_train[source, target]),
                                "empirical_dev_count": float(empirical_dev[source, target]),
                            }
                        )
                collections["runs"].append(row)
            except Exception as error:
                failure = {
                    **identifiers,
                    "status": "failed",
                    "failure": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "elapsed_s": time.perf_counter() - started,
                    "peak_rss_mb": float("nan"),
                }
                collections["failures"].append(failure)
                collections["runs"].append(failure)
            finally:
                memory.__exit__(None, None, None)
                collections["runs"][-1]["peak_rss_mb"] = memory.peak_rss_mb
            pd.DataFrame(collections["runs"]).to_csv(
                output / f"_partial_{policy}_{architecture}_{encoder_seed}.csv", index=False
            )
    return {
        **collections,
        "architecture": architecture,
        "encoder_seed": encoder_seed,
        "reused_runs": reused,
        "embedding_manifest_sha256": embedding_sha,
        "preprocessor_sha256": pca_sha,
    }


def _run_workers(
    *,
    root: Path,
    output: Path,
    policy: str,
    k_by_architecture: dict[str, Sequence[int]],
    workers: int,
    max_iter: int,
    tolerance: float,
    covariance_floor: float,
    architectures: Sequence[str] = ARCHITECTURES,
    encoder_seeds: Sequence[int] = ENCODER_SEEDS,
    hmm_seeds: Sequence[int] = HMM_SEEDS,
) -> dict[str, Any]:
    payloads = [
        {
            "root": str(root),
            "output": str(output),
            "policy": policy,
            "architecture": architecture,
            "encoder_seed": encoder_seed,
            "k_values": list(k_by_architecture[architecture]),
            "max_iter": max_iter,
            "tolerance": tolerance,
            "covariance_floor": covariance_floor,
            "hmm_seeds": list(hmm_seeds),
        }
        for architecture, encoder_seed in product(architectures, encoder_seeds)
    ]
    results: list[dict[str, Any]] = []
    if workers == 1:
        for payload in payloads:
            results.append(_analyze_worker(payload))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_analyze_worker, payload): payload for payload in payloads}
            for future in as_completed(futures):
                results.append(future.result())
    merged: dict[str, Any] = {
        key: [row for result in results for row in result[key]]
        for key in ("runs", "occupancy", "temporal", "per_state", "transitions", "prior", "failures")
    }
    merged["reused_runs"] = int(sum(result["reused_runs"] for result in results))
    merged["input_hashes"] = [
        {
            "architecture": result["architecture"],
            "encoder_seed": result["encoder_seed"],
            "embedding_manifest_sha256": result["embedding_manifest_sha256"],
            "preprocessor_sha256": result["preprocessor_sha256"],
        }
        for result in results
    ]
    return merged


def add_reference_deltas(runs: pd.DataFrame) -> pd.DataFrame:
    result = runs.copy()
    completed = result[result.status.eq("completed")]
    baseline = completed[completed.K.eq(REFERENCE_K)][
        [
            "architecture",
            "encoder_seed",
            "hmm_seed",
            "dev_log_likelihood_total",
            "dev_log_likelihood_per_point",
            "aic_train",
            "bic_train",
        ]
    ].rename(
        columns={
            "dev_log_likelihood_total": "reference_dev_log_likelihood_total",
            "dev_log_likelihood_per_point": "reference_dev_log_likelihood_per_point",
            "aic_train": "reference_aic_train",
            "bic_train": "reference_bic_train",
        }
    )
    if baseline.duplicated(["architecture", "encoder_seed", "hmm_seed"]).any():
        raise ValueError("multiple K=6 references in primary results")
    result = result.merge(
        baseline, on=["architecture", "encoder_seed", "hmm_seed"], how="left"
    )
    result["dev_delta_log_likelihood_vs_k6"] = (
        result.dev_log_likelihood_total - result.reference_dev_log_likelihood_total
    )
    result["dev_delta_log_likelihood_per_point_vs_k6"] = (
        result.dev_log_likelihood_per_point
        - result.reference_dev_log_likelihood_per_point
    )
    result["delta_aic_vs_k6"] = result.aic_train - result.reference_aic_train
    result["delta_bic_vs_k6"] = result.bic_train - result.reference_bic_train
    return result


def audit_run_grid(
    runs: pd.DataFrame,
    *,
    architectures: Sequence[str] = ARCHITECTURES,
    k_values: Sequence[int] = K_GRID,
    encoder_seeds: Sequence[int] = ENCODER_SEEDS,
    hmm_seeds: Sequence[int] = HMM_SEEDS,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Expose missing, duplicate, failed, and nonconverged grid cells."""

    keys = ["architecture", "encoder_seed", "hmm_seed", "K"]
    if runs.duplicated(keys).any():
        duplicate = runs.loc[runs.duplicated(keys, keep=False), keys]
        raise ValueError(f"duplicate primary run cells: {duplicate.to_dict('records')}")
    expected = pd.DataFrame(
        list(product(architectures, encoder_seeds, hmm_seeds, k_values)),
        columns=keys,
    )
    available = runs.copy()
    keep = keys + [
        value
        for value in ("status", "converged", "failure", "checkpoint", "assignment")
        if value in available
    ]
    audit = expected.merge(available[keep], on=keys, how="left")
    audit["status"] = audit.status.fillna("missing")
    converged = audit.get("converged", pd.Series(False, index=audit.index)).fillna(False)
    counts = {
        "expected": int(len(audit)),
        "completed": int(audit.status.eq("completed").sum()),
        "failed": int(audit.status.eq("failed").sum()),
        "missing": int(audit.status.eq("missing").sum()),
        "nonconverged_completed": int(
            (audit.status.eq("completed") & ~converged.astype(bool)).sum()
        ),
    }
    return audit, counts


def _partition_pair(
    left_path: str | Path,
    right_path: str | Path,
    left_k: int,
    right_k: int,
    *,
    thresholds: CorrespondenceThresholds | None = None,
) -> tuple[dict[str, Any], np.ndarray, pd.DataFrame, pd.DataFrame]:
    left = _load_assignment(left_path)
    right = _load_assignment(right_path)
    for key in ("patch_start_s", "patch_end_s"):
        if left[key].shape != right[key].shape or not np.allclose(
            left[key], right[key], rtol=0.0, atol=1e-9
        ):
            raise ValueError(f"assignment clocks disagree for {key}")
    left_label = np.asarray(left["predicted_state"], dtype=np.int64)
    right_label = np.asarray(right["predicted_state"], dtype=np.int64)
    common = (
        np.asarray(left["valid_time_mask"], dtype=bool)
        & np.asarray(right["valid_time_mask"], dtype=bool)
        & (left_label >= 0)
        & (right_label >= 0)
    )
    if not common.any():
        raise ValueError("partition pair has no common valid timesteps")
    matrix = rectangular_contingency(
        left_label[common], right_label[common], left_k, right_k
    )
    matches = hungarian_state_matches(matrix, thresholds=thresholds)
    diagnostics = split_merge_diagnostics(matrix, thresholds=thresholds)
    result = {
        "aligned_valid_timesteps": int(common.sum()),
        "ARI": float(adjusted_rand_score(left_label[common], right_label[common])),
        "NMI": float(normalized_mutual_info_score(left_label[common], right_label[common])),
        "mean_hungarian_f1": float(matches.f1.mean()),
        "minimum_hungarian_f1": float(matches.f1.min()),
        **rectangular_overlap_summary(matrix),
        **boundary_agreement(
            left_label,
            right_label,
            common,
            np.asarray(left["patch_start_s"], dtype=float),
            np.asarray(left["patch_end_s"], dtype=float),
            tolerance_patches=0,
        ),
        "left_split_candidate_count": int(
            diagnostics.query("direction == 'left_to_right'").split_candidate.sum()
        ),
        "right_merge_candidate_count": int(
            diagnostics.query("direction == 'right_to_left'").merge_candidate.sum()
        ),
    }
    return result, matrix, matches, diagnostics


def _summarize_pairs(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    metrics = ("ARI", "NMI", "mean_hungarian_f1", "boundary_f1_exact")
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return pd.DataFrame()
    for key, subset in frame.groupby(groups, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(groups, key))
        row["pair_count"] = int(len(subset))
        for metric in metrics:
            values = subset[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_median"] = float(values.median())
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def stability_tables(runs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    completed = runs[runs.status.eq("completed")].copy()
    within_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []
    for (architecture, k, encoder_seed), subset in completed.groupby(
        ["architecture", "K", "encoder_seed"]
    ):
        records = list(subset.sort_values("hmm_seed").to_dict("records"))
        for left, right in combinations(records, 2):
            metrics, _matrix, _matches, _diagnostics = _partition_pair(
                left["assignment"], right["assignment"], int(k), int(k)
            )
            within_rows.append(
                {
                    "architecture": architecture,
                    "K": int(k),
                    "encoder_seed": int(encoder_seed),
                    "left_hmm_seed": int(left["hmm_seed"]),
                    "right_hmm_seed": int(right["hmm_seed"]),
                    **metrics,
                }
            )
    for (architecture, k), subset in completed.groupby(["architecture", "K"]):
        records = list(subset.to_dict("records"))
        for left, right in combinations(records, 2):
            if int(left["encoder_seed"]) == int(right["encoder_seed"]):
                continue
            if int(left["encoder_seed"]) > int(right["encoder_seed"]):
                left, right = right, left
            metrics, _matrix, _matches, diagnostics = _partition_pair(
                left["assignment"], right["assignment"], int(k), int(k)
            )
            relation = (
                "matched_hmm_seed"
                if int(left["hmm_seed"]) == int(right["hmm_seed"])
                else "different_hmm_seed"
            )
            cross_rows.append(
                {
                    "architecture": architecture,
                    "K": int(k),
                    "left_encoder_seed": int(left["encoder_seed"]),
                    "right_encoder_seed": int(right["encoder_seed"]),
                    "left_hmm_seed": int(left["hmm_seed"]),
                    "right_hmm_seed": int(right["hmm_seed"]),
                    "hmm_seed_relation": relation,
                    "split_merge_row_count": int(
                        diagnostics.split_candidate.fillna(False).sum()
                        + diagnostics.merge_candidate.fillna(False).sum()
                    ),
                    **metrics,
                }
            )
    within = pd.DataFrame(within_rows)
    cross = pd.DataFrame(cross_rows)
    return {
        "within_pairwise": within,
        "within_summary": _summarize_pairs(within, ["architecture", "K"]),
        "cross_pairwise": cross,
        "cross_summary": _summarize_pairs(
            cross, ["architecture", "K", "hmm_seed_relation"]
        ),
    }


def prior_policy_stability_tables(
    policy_runs: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    within_parts: list[pd.DataFrame] = []
    cross_parts: list[pd.DataFrame] = []
    for policy, subset in policy_runs.groupby("prior_policy"):
        tables = stability_tables(subset)
        within = tables["within_summary"].copy()
        cross = tables["cross_summary"].copy()
        within["prior_policy"] = policy
        cross["prior_policy"] = policy
        within_parts.append(within)
        cross_parts.append(cross)
    fixed = policy_runs[policy_runs.prior_policy.eq("fixed")]
    normalized = policy_runs[
        policy_runs.prior_policy.eq("normalized_6_over_k")
    ]
    normalized_by_key = {
        (
            row.architecture,
            int(row.encoder_seed),
            int(row.hmm_seed),
            int(row.K),
        ): row
        for _, row in normalized.iterrows()
        if row.status == "completed"
    }
    alignment_rows: list[dict[str, Any]] = []
    for _, left in fixed.iterrows():
        if left.status != "completed":
            continue
        key = (
            left.architecture,
            int(left.encoder_seed),
            int(left.hmm_seed),
            int(left.K),
        )
        right = normalized_by_key.get(key)
        if right is None:
            continue
        metrics, _matrix, _matches, _diagnostics = _partition_pair(
            left.assignment, right.assignment, int(left.K), int(right.K)
        )
        alignment_rows.append(
            {
                "architecture": left.architecture,
                "encoder_seed": int(left.encoder_seed),
                "hmm_seed": int(left.hmm_seed),
                "K": int(left.K),
                **metrics,
            }
        )
    return {
        "prior_policy_within_summary": pd.concat(
            within_parts, ignore_index=True
        ) if within_parts else pd.DataFrame(),
        "prior_policy_cross_summary": pd.concat(
            cross_parts, ignore_index=True
        ) if cross_parts else pd.DataFrame(),
        "fixed_vs_normalized_alignment": pd.DataFrame(alignment_rows),
    }


def cross_k_tables(
    runs: pd.DataFrame,
    per_state: pd.DataFrame,
    output: Path,
    *,
    k_grid: Sequence[int] = K_GRID,
    reference_k: int = REFERENCE_K,
    adjacent_pairs: Sequence[tuple[int, int]] | None = None,
) -> dict[str, pd.DataFrame]:
    completed = runs[runs.status.eq("completed")].copy()
    pair_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    refinement_rows: list[dict[str, Any]] = []
    matrix_dir = output / "cross_k_overlap_matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    grid = tuple(int(value) for value in k_grid)
    requested = set(
        tuple(map(int, pair))
        for pair in (
            adjacent_pairs
            if adjacent_pairs is not None
            else tuple(zip(grid[:-1], grid[1:]))
        )
    )
    requested.update((int(reference_k), k) for k in grid if k != reference_k)
    for (architecture, encoder_seed, hmm_seed), subset in completed.groupby(
        ["architecture", "encoder_seed", "hmm_seed"]
    ):
        by_k = {int(row.K): row for _, row in subset.iterrows()}
        for left_k, right_k in sorted(requested):
            if left_k not in by_k or right_k not in by_k:
                continue
            left = by_k[left_k]
            right = by_k[right_k]
            metrics, matrix, matches, diagnostics = _partition_pair(
                left.assignment, right.assignment, left_k, right_k
            )
            pair_id = (
                f"{architecture}_enc{encoder_seed}_hmm{hmm_seed}_k{left_k}_k{right_k}"
            )
            matrix_path = matrix_dir / f"{pair_id}.npz"
            np.savez_compressed(
                matrix_path,
                contingency=matrix,
                row_normalized=np.divide(
                    matrix,
                    matrix.sum(axis=1, keepdims=True),
                    out=np.zeros_like(matrix, dtype=float),
                    where=matrix.sum(axis=1, keepdims=True) > 0,
                ),
                column_normalized=np.divide(
                    matrix,
                    matrix.sum(axis=0, keepdims=True),
                    out=np.zeros_like(matrix, dtype=float),
                    where=matrix.sum(axis=0, keepdims=True) > 0,
                ),
            )
            identifiers = {
                "pair_id": pair_id,
                "architecture": architecture,
                "encoder_seed": int(encoder_seed),
                "hmm_seed": int(hmm_seed),
                "left_K": left_k,
                "right_K": right_k,
                "comparison_scope": (
                    f"vs_k{reference_k}" if left_k == reference_k else "adjacent_k"
                ),
            }
            pair_rows.append({**identifiers, "matrix_path": str(matrix_path), **metrics})
            overlap = overlap_metrics(matrix)
            for row in overlap.to_dict("records"):
                cell_rows.append({**identifiers, **row})
            for row in diagnostics.to_dict("records"):
                source_k = left_k if row["direction"] == "left_to_right" else right_k
                split = per_state[
                    per_state.architecture.eq(architecture)
                    & per_state.encoder_seed.eq(encoder_seed)
                    & per_state.hmm_seed.eq(hmm_seed)
                    & per_state.K.eq(source_k)
                    & per_state.split.eq("dev")
                    & per_state.state.eq(int(row["source_state"]))
                ]
                extra = {}
                if not split.empty:
                    extra = {
                        "source_state_bout_count": int(split.iloc[0].bout_count),
                        "source_state_median_bout_duration_s": float(
                            split.iloc[0].median_bout_duration_s
                        ),
                    }
                split_rows.append({**identifiers, **row, **extra})
            if left_k == reference_k and right_k > reference_k:
                # Canonicalize baseline labels by descending occupancy.  This is
                # label-invariant and avoids claiming semantic correspondence
                # when cross-encoder K=6 agreement itself is weak.
                left_counts = matrix.sum(axis=1)
                rank = {
                    int(state): int(position + 1)
                    for position, state in enumerate(np.argsort(left_counts)[::-1])
                }
                for row in diagnostics.query(
                    "direction == 'left_to_right' and split_candidate == True"
                ).to_dict("records"):
                    refinement_rows.append(
                        {
                            **identifiers,
                            "baseline_state": int(row["source_state"]),
                            "baseline_occupancy_rank": rank[int(row["source_state"])],
                            "substantial_child_count": int(row["substantial_target_count"]),
                            "substantial_children": row["substantial_targets"],
                            "best_child_fraction": float(row["best_overlap_fraction"]),
                            "second_child_fraction": float(row["second_overlap_fraction"]),
                        }
                    )
    pair_frame = pd.DataFrame(pair_rows)
    refinement = pd.DataFrame(refinement_rows)
    reproducibility_rows: list[dict[str, Any]] = []
    if not refinement.empty:
        for key, subset in refinement.groupby(
            ["architecture", "right_K", "baseline_occupancy_rank"]
        ):
            architecture, k, rank = key
            reproducibility_rows.append(
                {
                    "architecture": architecture,
                    "K": int(k),
                    "baseline_occupancy_rank": int(rank),
                    "runs_with_equivalent_refinement": int(len(subset)),
                    "fraction_of_nine_runs": float(len(subset) / 9.0),
                    "median_substantial_child_count": float(
                        subset.substantial_child_count.median()
                    ),
                    "mean_second_child_fraction": float(
                        subset.second_child_fraction.mean()
                    ),
                }
            )
    return {
        "cross_k_pairs": pair_frame,
        "cross_k_cells": pd.DataFrame(cell_rows),
        "cross_k_split_merge": pd.DataFrame(split_rows),
        "refinement_events": refinement,
        "refinement_reproducibility": pd.DataFrame(reproducibility_rows),
    }


def seedwise_rankings(runs: pd.DataFrame) -> pd.DataFrame:
    completed = runs[runs.status.eq("completed")].copy()
    rows: list[dict[str, Any]] = []
    for (architecture, encoder_seed, hmm_seed), subset in completed.groupby(
        ["architecture", "encoder_seed", "hmm_seed"]
    ):
        for criterion in ("aic_train", "bic_train"):
            ranked = subset.sort_values([criterion, "K"])
            best = float(ranked.iloc[0][criterion])
            for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
                rows.append(
                    {
                        "architecture": architecture,
                        "encoder_seed": int(encoder_seed),
                        "hmm_seed": int(hmm_seed),
                        "criterion": criterion,
                        "K": int(row.K),
                        "rank": rank,
                        "value": float(row[criterion]),
                        "delta_from_seedwise_best": float(row[criterion] - best),
                        "seedwise_winner": rank == 1,
                    }
                )
    return pd.DataFrame(rows)


def _mode_lower(values: Iterable[int]) -> int:
    series = pd.Series(list(values), dtype=int)
    counts = series.value_counts()
    return int(min(counts[counts.eq(counts.max())].index))


def choose_shortlist(
    rankings: pd.DataFrame,
    within_summary: pd.DataFrame,
    cross_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Union of five declared, non-weighted selectors (at most five K values)."""

    rows: list[dict[str, Any]] = []
    for architecture in ARCHITECTURES:
        selected: dict[int, list[str]] = {REFERENCE_K: ["mandatory_reference"]}
        subset = rankings[
            rankings.architecture.eq(architecture) & rankings.seedwise_winner
        ]
        for criterion in ("aic_train", "bic_train"):
            winner = _mode_lower(subset[subset.criterion.eq(criterion)].K.astype(int))
            selected.setdefault(winner, []).append(f"modal_seedwise_{criterion}_winner")
        within = within_summary[within_summary.architecture.eq(architecture)]
        if not within.empty:
            best = within.sort_values(
                ["ARI_mean", "K"], ascending=[False, True]
            ).iloc[0]
            selected.setdefault(int(best.K), []).append("highest_mean_within_encoder_ARI")
        cross = cross_summary[
            cross_summary.architecture.eq(architecture)
            & cross_summary.hmm_seed_relation.eq("matched_hmm_seed")
        ]
        if not cross.empty:
            best = cross.sort_values(
                ["ARI_mean", "K"], ascending=[False, True]
            ).iloc[0]
            selected.setdefault(int(best.K), []).append("highest_mean_cross_encoder_ARI")
        for k, reasons in sorted(selected.items()):
            rows.append(
                {
                    "architecture": architecture,
                    "K": k,
                    "selection_reasons": "|".join(reasons),
                }
            )
    return pd.DataFrame(rows)


def _primary_summary(
    runs: pd.DataFrame,
    within: pd.DataFrame,
    cross: pd.DataFrame,
) -> pd.DataFrame:
    completed = runs[runs.status.eq("completed")].copy()
    aggregate_columns = {
        "dev_delta_log_likelihood_per_point_vs_k6": "mean_dev_delta_ll_per_point_vs_k6",
        "delta_aic_vs_k6": "mean_delta_aic_vs_k6",
        "delta_bic_vs_k6": "mean_delta_bic_vs_k6",
        "dev_effective_state_ratio": "mean_dev_effective_state_ratio",
        "dev_largest_state_occupancy": "mean_dev_largest_state_occupancy",
        "dev_bouts_per_minute": "mean_dev_bouts_per_minute",
        "dev_median_bout_duration_s": "mean_dev_median_bout_duration_s",
        "converged": "convergence_fraction",
    }
    summary = (
        completed.groupby(["architecture", "K"])[list(aggregate_columns)]
        .mean()
        .rename(columns=aggregate_columns)
        .reset_index()
    )
    within_cols = within[["architecture", "K", "ARI_mean", "ARI_min"]].rename(
        columns={"ARI_mean": "within_encoder_ari_mean", "ARI_min": "within_encoder_ari_min"}
    )
    matched = cross[cross.hmm_seed_relation.eq("matched_hmm_seed")][
        ["architecture", "K", "ARI_mean", "ARI_min"]
    ].rename(
        columns={"ARI_mean": "cross_encoder_ari_mean", "ARI_min": "cross_encoder_ari_min"}
    )
    summary = summary.merge(within_cols, on=["architecture", "K"], how="left")
    summary = summary.merge(matched, on=["architecture", "K"], how="left")
    warnings = (
        completed.groupby(["architecture", "K"])
        .agg(
            nonconverged_runs=("converged", lambda value: int((~value.astype(bool)).sum())),
            occupancy_warning_runs=(
                "dev_occupancy_warning",
                lambda value: int(value.fillna("").astype(str).ne("").sum()),
            ),
        )
        .reset_index()
    )
    summary = summary.merge(warnings, on=["architecture", "K"], how="left")
    summary["warning"] = summary.apply(
        lambda row: "|".join(
            value
            for value, active in (
                ("nonconvergence", row.nonconverged_runs > 0),
                ("occupancy", row.occupancy_warning_runs > 0),
            )
            if active
        ),
        axis=1,
    )
    return summary


def recommendation_table(
    summary: pd.DataFrame,
    rankings: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for architecture in ARCHITECTURES:
        rank = rankings[
            rankings.architecture.eq(architecture)
            & rankings.criterion.eq("bic_train")
            & rankings.seedwise_winner
        ]
        statistical = _mode_lower(rank.K.astype(int))
        subset = summary[summary.architecture.eq(architecture)].copy()
        reproducibility = int(
            subset.sort_values(
                ["cross_encoder_ari_mean", "within_encoder_ari_mean", "K"],
                ascending=[False, False, True],
            ).iloc[0].K
        )
        near_bic = subset[subset.mean_delta_bic_vs_k6 <= subset.mean_delta_bic_vs_k6.min() + 2.0]
        stable = near_bic[
            (near_bic.within_encoder_ari_mean >= subset.within_encoder_ari_mean.median())
            & (near_bic.cross_encoder_ari_mean >= subset.cross_encoder_ari_mean.median())
            & near_bic.warning.eq("")
        ]
        practical = (
            str(int(stable.sort_values("K").iloc[0].K))
            if not stable.empty
            else f"{min(statistical, reproducibility)}-{max(statistical, reproducibility)}"
        )
        if statistical == max(K_GRID) and practical == str(max(K_GRID)):
            practical = f"{max(K_GRID)} provisional; extend K grid"
        rows.append(
            {
                "architecture": architecture,
                "statistical_preferred_K_modal_BIC": statistical,
                "statistical_preference_at_grid_boundary": bool(
                    statistical == max(K_GRID)
                ),
                "reproducibility_preferred_K_lexicographic": reproducibility,
                "practical_K_or_range": practical,
                "k6_statistically_preferred": statistical == REFERENCE_K,
                "k6_reproducibility_preferred": reproducibility == REFERENCE_K,
            }
        )
    return pd.DataFrame(rows)


def _save_tables(output: Path, tables: dict[str, pd.DataFrame]) -> None:
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)


def _plot_line(
    frame: pd.DataFrame,
    y_columns: Sequence[str],
    labels: Sequence[str],
    ylabel: str,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for axis, architecture in zip(axes, ARCHITECTURES):
        subset = frame[frame.architecture.eq(architecture)].sort_values("K")
        for column, label in zip(y_columns, labels):
            axis.plot(subset.K, subset[column], marker="o", label=label)
        axis.set_title(architecture)
        axis.set_xlabel("K")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_figures(
    output: Path,
    summary: pd.DataFrame,
    rankings: pd.DataFrame,
    cross_k: pd.DataFrame,
    prior_summary: pd.DataFrame,
) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _plot_line(
        summary,
        ["mean_delta_aic_vs_k6", "mean_delta_bic_vs_k6"],
        ["delta AIC", "delta BIC"],
        "mean delta vs K=6",
        figures / "aic_bic_curves.png",
    )
    _plot_line(
        summary,
        ["mean_dev_delta_ll_per_point_vs_k6"],
        ["dev delta LL / point"],
        "dev marginal log-likelihood gain",
        figures / "dev_delta_log_likelihood.png",
    )
    _plot_line(
        summary,
        ["mean_dev_effective_state_ratio", "mean_dev_largest_state_occupancy"],
        ["K_eff / K", "largest occupancy"],
        "occupancy diagnostic",
        figures / "occupancy_and_effective_states.png",
    )
    _plot_line(
        summary,
        ["mean_dev_bouts_per_minute", "mean_dev_median_bout_duration_s"],
        ["bouts/min", "median duration (s)"],
        "temporal diagnostic",
        figures / "bouts_and_duration.png",
    )
    _plot_line(
        summary,
        ["within_encoder_ari_mean", "within_encoder_ari_min"],
        ["mean ARI", "minimum ARI"],
        "within-encoder stability",
        figures / "within_encoder_ari.png",
    )
    _plot_line(
        summary,
        ["cross_encoder_ari_mean", "cross_encoder_ari_min"],
        ["mean ARI", "minimum ARI"],
        "cross-encoder matched-seed stability",
        figures / "cross_encoder_ari.png",
    )
    if not cross_k.empty:
        aggregated = (
            cross_k.groupby(["architecture", "left_K", "right_K"])
            .variation_of_information.mean()
            .reset_index()
        )
        aggregated["pair"] = aggregated.left_K.astype(str) + "-" + aggregated.right_K.astype(str)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for axis, architecture in zip(axes, ARCHITECTURES):
            subset = aggregated[aggregated.architecture.eq(architecture)]
            axis.bar(np.arange(len(subset)), subset.variation_of_information)
            axis.set_xticks(np.arange(len(subset)), subset.pair, rotation=45, ha="right")
            axis.set_title(architecture)
            axis.set_ylabel("variation of information (nats)")
        fig.tight_layout()
        fig.savefig(figures / "cross_k_overlap.png", dpi=180)
        plt.close(fig)
    winners = rankings[rankings.seedwise_winner]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for axis, architecture in zip(axes, ARCHITECTURES):
        subset = winners[winners.architecture.eq(architecture)]
        for criterion, marker in (("aic_train", "o"), ("bic_train", "x")):
            values = subset[subset.criterion.eq(criterion)]
            x = np.arange(len(values))
            axis.scatter(x, values.K, marker=marker, label=criterion)
        axis.set_title(architecture)
        axis.set_ylabel("seedwise preferred K")
        axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "seedwise_preferred_k.png", dpi=180)
    plt.close(fig)
    if not prior_summary.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for axis, architecture in zip(axes, ARCHITECTURES):
            subset = prior_summary[prior_summary.architecture.eq(architecture)]
            for policy, group in subset.groupby("prior_policy"):
                axis.plot(group.K, group.dev_log_likelihood_per_point, marker="o", label=policy)
            axis.set_title(architecture)
            axis.set_xlabel("K")
            axis.set_ylabel("dev LL / point")
            axis.legend()
            axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures / "fixed_vs_normalized_prior.png", dpi=180)
        plt.close(fig)


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "NA" if not np.isfinite(number) else f"{number:.{digits}f}"


def write_report(
    output: Path,
    summary: pd.DataFrame,
    recommendations: pd.DataFrame,
    shortlist: pd.DataFrame,
    prior_comparison: pd.DataFrame,
    prior_diagnostics: pd.DataFrame,
    policy_alignment: pd.DataFrame,
    refinement: pd.DataFrame,
    runs: pd.DataFrame,
) -> Path:
    lines = [
        "# 0.04 s Social-GNN: HMM State-Number Selection and Cross-Seed Reproducibility",
        "",
        "## 1. Executive summary",
        "",
    ]
    for _, row in recommendations.iterrows():
        boundary = (
            " This is the tested upper boundary, so the statistical optimum is unresolved."
            if row.statistical_preference_at_grid_boundary
            else ""
        )
        lines.append(
            f"- **{row.architecture}:** statistical preference K={row.statistical_preferred_K_modal_BIC}; "
            f"reproducibility preference K={row.reproducibility_preferred_K_lexicographic}; "
            f"practical recommendation `{row.practical_K_or_range}`. K=6 is "
            f"{'supported' if row.k6_statistically_preferred or row.k6_reproducibility_preferred else 'not the leading choice'} "
            f"by at least one of the two declared criteria.{boundary}"
        )
    lines.extend(
        [
            "",
            "No encoder, SSL model, embedding, or PCA transform was retrained. Conclusions are development-set diagnostics, not behavioral validity.",
            "",
            "## 2. Frozen inputs and protocol",
            "",
            f"- Architectures: {', '.join(ARCHITECTURES)}.",
            f"- K grid: {list(K_GRID)}; encoder seeds: {list(ENCODER_SEEDS)}; HMM seeds: {list(HMM_SEEDS)}.",
            f"- Training: {list(TRAIN_TRIALS)}; reused development trial: {DEV_TRIAL}; excluded: {EXCLUDED_TRIAL}.",
            "- Existing train-fitted PCA16 transforms were reused per encoder seed. Invalid patches split sequences; no transition crosses a trial or clock gap.",
            f"- Primary prior: dt={TIMESTEP_S} s, target dwell={TARGET_DWELL_S} s (target p_self=0.98), strength={FIXED_PRIOR_STRENGTH:g}.",
            f"- Completed primary runs: {int(runs.status.eq('completed').sum())}/{len(runs)}; exact K=6 artifacts reused: {int(runs.reused_existing_artifact.fillna(False).sum())}.",
            "",
            "## 3. Likelihood and information criteria",
            "",
            "AIC/BIC use p=(K-1)+K(K-1)+2KD=K^2+32K-1 for D=16 and n equal to the actual valid training observations. Raw likelihoods are interpreted only within one architecture and encoder-seed PCA space. Development values below are deltas from matched K=6, never cross-space raw comparisons.",
            "Both architectures improve monotonically through K=12 on AIC, BIC, and development likelihood. K=12 is therefore an upper-bound winner, not an interior optimum; the grid must be extended before claiming that the statistical state number is identified.",
            "",
            "| architecture | K | dev delta LL/point | delta AIC | delta BIC |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {row.architecture} | {int(row.K)} | {_fmt(row.mean_dev_delta_ll_per_point_vs_k6, 5)} | {_fmt(row.mean_delta_aic_vs_k6, 1)} | {_fmt(row.mean_delta_bic_vs_k6, 1)} |"
        )
    lines.extend(
        [
            "",
            "## 4. Occupancy and effective-state use",
            "",
            "Warnings use fixed thresholds: near-empty <0.5%, small <1%, K_eff/K <0.50, or largest state >80%. They are flags, not automatic exclusion rules.",
            "",
            "| architecture | K | K_eff/K | largest occupancy | warning runs |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {row.architecture} | {int(row.K)} | {_fmt(row.mean_dev_effective_state_ratio)} | {_fmt(row.mean_dev_largest_state_occupancy)} | {int(row.occupancy_warning_runs)} |"
        )
    lines.extend(
        [
            "",
            "## 5. Temporal behavior",
            "",
            "| architecture | K | bouts/min | median bout (s) | convergence |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {row.architecture} | {int(row.K)} | {_fmt(row.mean_dev_bouts_per_minute)} | {_fmt(row.mean_dev_median_bout_duration_s)} | {_fmt(row.convergence_fraction)} |"
        )
    lines.extend(
        [
            "",
            "## 6. Within-encoder HMM-seed reproducibility",
            "",
            "Each encoder checkpoint contributes three pairwise comparisons among HMM seeds. Reported summaries are label-permutation invariant.",
            "",
            "| architecture | K | mean ARI | minimum ARI |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {row.architecture} | {int(row.K)} | {_fmt(row.within_encoder_ari_mean)} | {_fmt(row.within_encoder_ari_min)} |"
        )
    lines.extend(
        [
            "",
            "## 7. Cross-encoder reproducibility",
            "",
            "Matched downstream seed identifiers are summarized separately in the CSV outputs; equal integers do not imply equal initialization in different embedding/PCA spaces.",
            "",
            "| architecture | K | matched-seed mean ARI | minimum ARI |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {row.architecture} | {int(row.K)} | {_fmt(row.cross_encoder_ari_mean)} | {_fmt(row.cross_encoder_ari_min)} |"
        )
    lines.extend(
        [
            "",
            "## 8. Cross-K refinement and split/merge structure",
            "",
        ]
    )
    if refinement.empty:
        lines.append("No substantial K=6-to-higher-K split recurred under the 20% component rule.")
    else:
        strongest = refinement.sort_values("fraction_of_nine_runs", ascending=False).head(8)
        lines.extend(
            [
                "Equivalent refinement is conservatively defined as splitting the same K=6 state after canonicalizing K=6 labels by descending occupancy rank. This is a label-invariant structural diagnostic, not a biological state match.",
                "",
                "| architecture | K | parent occupancy rank | recurrence / 9 |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for _, row in strongest.iterrows():
            lines.append(
                f"| {row.architecture} | {int(row.K)} | {int(row.baseline_occupancy_rank)} | {int(row.runs_with_equivalent_refinement)}/9 |"
            )
    lines.extend(
        [
            "",
            "## 9. Prior-policy sensitivity on the shortlist",
            "",
            "The shortlist is the union of K=6, modal seedwise AIC and BIC winners, the best mean within-encoder ARI K, and the best matched-seed cross-encoder ARI K. This is reproducible and uses no weighted score.",
            "",
        ]
    )
    for architecture in ARCHITECTURES:
        values = shortlist[shortlist.architecture.eq(architecture)]
        lines.append(
            f"- {architecture}: "
            + ", ".join(
                f"K={int(row.K)} ({row.selection_reasons})" for _, row in values.iterrows()
            )
        )
    if not prior_comparison.empty:
        lines.extend(
            [
                "",
                "Fixed 50,000 and K-normalized 50,000×6/K results are in `prior_policy_comparison.csv`; this check tests whether apparent K effects are partly induced by changing prior/data leverage.",
                "The companion `prior_policy_within_summary.csv`, `prior_policy_cross_summary.csv`, and `fixed_vs_normalized_alignment.csv` show whether the policy changes the recovered partitions, not only their likelihood and dwell statistics.",
            ]
        )
    if not prior_diagnostics.empty:
        leverage = (
            prior_diagnostics.groupby(["architecture", "K"])
            .prior_to_xi_ratio.agg(["mean", "max"])
            .reset_index()
        )
        lines.extend(
            [
                "",
                "Fixed-prior leverage (per-state-row prior mass divided by posterior expected training xi count):",
                "",
                "| architecture | K | mean prior/xi | maximum prior/xi |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for _, row in leverage.iterrows():
            lines.append(
                f"| {row.architecture} | {int(row.K)} | {_fmt(row['mean'])} | {_fmt(row['max'])} |"
            )
    if not policy_alignment.empty:
        alignment = (
            policy_alignment.groupby(["architecture", "K"]).ARI.agg(["mean", "min"]).reset_index()
        )
        lines.extend(
            [
                "",
                "Direct fixed-vs-normalized partition agreement for matched encoder/HMM seeds:",
                "",
                "| architecture | K | mean ARI | minimum ARI |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for _, row in alignment.iterrows():
            lines.append(
                f"| {row.architecture} | {int(row.K)} | {_fmt(row['mean'])} | {_fmt(row['min'])} |"
            )
    lines.extend(
        [
            "",
            "## 10. Recommended K and status of K=6",
            "",
            "Statistical preference is the modal seedwise BIC winner. Reproducibility preference is lexicographic: maximize matched-seed cross-encoder ARI, then within-encoder ARI, then prefer lower K. A practical single K is emitted only when a no-warning candidate lies within mean delta-BIC 2 of the best and is at or above median stability on both axes; otherwise a range is reported.",
            "",
        ]
    )
    for _, row in recommendations.iterrows():
        caveat = (
            " The statistical winner is at Kmax and is not a resolved optimum."
            if row.statistical_preference_at_grid_boundary
            else ""
        )
        lines.append(
            f"- **{row.architecture}:** statistical K={row.statistical_preferred_K_modal_BIC}; reproducibility K={row.reproducibility_preferred_K_lexicographic}; practical `{row.practical_K_or_range}`.{caveat}"
        )
    lines.extend(
        [
            "",
            "There is no single architecture-independent reproducible state number in this experiment: Graph Full favors K=12 on both declared axes, whereas Invariant MLP Full favors K=12 statistically but K=6 for cross-encoder reproducibility. K=6 remains a defensible conservative anchor for the MLP, but it is not the likelihood/BIC optimum and is not justified as a universal final K.",
        ]
    )
    lines.extend(
        [
            "",
            "## 11. Limitations and next validation",
            "",
            "- A4_C4 is reused development data and was involved upstream in encoder checkpoint selection; it is not an untouched biological test.",
            "- Encoder seeds define different representation/PCA spaces, so absolute likelihood, AIC, and BIC are not compared across those spaces.",
            "- ARI/NMI and boundary agreement measure reproducibility, not behavioral correctness. Stable partitions can still be biologically uninformative.",
            "- The sticky prior is strong relative to some state-specific expected transition counts. The per-row prior/data ratios must be consulted before interpreting fragmentation as data-driven.",
            "- K-normalized prior strength is a sensitivity analysis, not a replacement selected after looking at outcomes.",
            "- A genuinely independent animal pair and blinded video review remain required before assigning behavior names or fixing K for production.",
        ]
    )
    path = output / "hmm_k_selection_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_k_selection(
    *,
    root: str | Path,
    output_dir: str | Path,
    workers: int = 3,
    max_iter: int = 40,
    tolerance: float = 1e-4,
    covariance_floor: float = 1e-3,
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    primary = _run_workers(
        root=root_path,
        output=output,
        policy="fixed",
        k_by_architecture={name: K_GRID for name in ARCHITECTURES},
        workers=workers,
        max_iter=max_iter,
        tolerance=tolerance,
        covariance_floor=covariance_floor,
    )
    runs = add_reference_deltas(pd.DataFrame(primary["runs"]))
    run_grid_audit, run_grid_counts = audit_run_grid(runs)
    base_tables = {
        "runs": runs,
        "likelihood_information_criteria": runs[
            [
                column
                for column in (
                    "architecture", "encoder_seed", "hmm_seed", "K", "status",
                    "train_valid_points", "dev_valid_points",
                    "train_log_likelihood_total", "train_log_likelihood_per_point",
                    "dev_log_likelihood_total", "dev_log_likelihood_per_point",
                    "dev_delta_log_likelihood_vs_k6",
                    "dev_delta_log_likelihood_per_point_vs_k6",
                    "parameter_count", "aic_train", "bic_train",
                    "delta_aic_vs_k6", "delta_bic_vs_k6", "iterations", "converged", "failure",
                )
                if column in runs
            ]
        ],
        "occupancy_by_state": pd.DataFrame(primary["occupancy"]),
        "temporal_metrics": pd.DataFrame(primary["temporal"]),
        "per_state_temporal_metrics": pd.DataFrame(primary["per_state"]),
        "transition_matrices": pd.DataFrame(primary["transitions"]),
        "prior_diagnostics": pd.DataFrame(primary["prior"]),
        "failures": pd.DataFrame(primary["failures"]),
        "run_grid_audit": run_grid_audit,
    }
    stability = stability_tables(runs)
    cross_k = cross_k_tables(
        runs, base_tables["per_state_temporal_metrics"], output
    )
    rankings = seedwise_rankings(runs)
    shortlist = choose_shortlist(
        rankings, stability["within_summary"], stability["cross_summary"]
    )
    summary = _primary_summary(
        runs, stability["within_summary"], stability["cross_summary"]
    )
    recommendations = recommendation_table(summary, rankings)

    normalized_k = {
        architecture: tuple(
            int(value)
            for value in shortlist[
                shortlist.architecture.eq(architecture) & shortlist.K.ne(REFERENCE_K)
            ].K
        )
        for architecture in ARCHITECTURES
    }
    normalized = _run_workers(
        root=root_path,
        output=output / "prior_policy",
        policy="normalized_6_over_k",
        k_by_architecture=normalized_k,
        workers=workers,
        max_iter=max_iter,
        tolerance=tolerance,
        covariance_floor=covariance_floor,
    ) if any(normalized_k.values()) else {
        key: [] for key in ("runs", "occupancy", "temporal", "per_state", "transitions", "prior", "failures")
    }
    fixed_shortlist = runs.merge(
        shortlist[["architecture", "K"]], on=["architecture", "K"], how="inner"
    ).copy()
    normalized_runs = pd.DataFrame(normalized["runs"])
    # K=6 has identical strength under both policies and is represented by the
    # exact same artifact rather than being refit under a duplicate label.
    normalized_k6 = fixed_shortlist[fixed_shortlist.K.eq(REFERENCE_K)].copy()
    normalized_k6["prior_policy"] = "normalized_6_over_k"
    policy_runs = pd.concat(
        [fixed_shortlist, normalized_runs, normalized_k6], ignore_index=True, sort=False
    )
    policy_summary = (
        policy_runs[policy_runs.status.eq("completed")]
        .groupby(["architecture", "K", "prior_policy"])
        .agg(
            completed_runs=("status", "size"),
            dev_log_likelihood_per_point=("dev_log_likelihood_per_point", "mean"),
            effective_state_ratio=("dev_effective_state_ratio", "mean"),
            bouts_per_minute=("dev_bouts_per_minute", "mean"),
            learned_self_transition=("learned_self_transition_mean", "mean"),
        )
        .reset_index()
    )
    policy_stability = prior_policy_stability_tables(policy_runs)
    within_policy = policy_stability["prior_policy_within_summary"][
        ["architecture", "K", "prior_policy", "ARI_mean"]
    ].rename(columns={"ARI_mean": "within_encoder_ari_mean"})
    cross_policy = policy_stability["prior_policy_cross_summary"]
    cross_policy = cross_policy[
        cross_policy.hmm_seed_relation.eq("matched_hmm_seed")
    ][["architecture", "K", "prior_policy", "ARI_mean"]].rename(
        columns={"ARI_mean": "cross_encoder_ari_mean"}
    )
    policy_summary = policy_summary.merge(
        within_policy, on=["architecture", "K", "prior_policy"], how="left"
    ).merge(
        cross_policy, on=["architecture", "K", "prior_policy"], how="left"
    )
    all_tables = {
        **base_tables,
        **stability,
        **cross_k,
        "seedwise_information_criterion_rankings": rankings,
        "shortlist": shortlist,
        "multi_criterion_summary": summary,
        "recommendations": recommendations,
        "prior_policy_runs": policy_runs,
        "prior_policy_comparison": policy_summary,
        "prior_policy_failures": pd.DataFrame(normalized.get("failures", [])),
        "prior_policy_prior_diagnostics": pd.DataFrame(normalized.get("prior", [])),
        **policy_stability,
    }
    _save_tables(output, all_tables)
    make_figures(
        output,
        summary,
        rankings,
        cross_k["cross_k_pairs"],
        policy_summary,
    )
    report = write_report(
        output,
        summary,
        recommendations,
        shortlist,
        policy_summary,
        base_tables["prior_diagnostics"],
        policy_stability["fixed_vs_normalized_alignment"],
        cross_k["refinement_reproducibility"],
        runs,
    )
    manifest = {
        "contract": "formal_hmm_k_selection_0p04_v1",
        "title": "0.04 s Social-GNN: HMM State-Number Selection and Cross-Seed Reproducibility",
        "root": str(root_path),
        "output_dir": str(output),
        "architectures": list(ARCHITECTURES),
        "encoder_seeds": list(ENCODER_SEEDS),
        "hmm_seeds": list(HMM_SEEDS),
        "k_grid": list(K_GRID),
        "train_trials": list(TRAIN_TRIALS),
        "development_trial": DEV_TRIAL,
        "excluded_trial": EXCLUDED_TRIAL,
        "pca_dim": PCA_DIM,
        "timestep_s": TIMESTEP_S,
        "target_dwell_s": TARGET_DWELL_S,
        "target_self_transition": 1.0 - TIMESTEP_S / TARGET_DWELL_S,
        "fixed_prior_strength": FIXED_PRIOR_STRENGTH,
        "normalized_prior_formula": "50000 * 6 / K",
        "parameter_count_formula": "(K-1)+K(K-1)+2*K*D = K^2+32K-1 for D=16",
        "warm_start": False,
        "primary_expected_runs": len(ARCHITECTURES) * len(ENCODER_SEEDS) * len(HMM_SEEDS) * len(K_GRID),
        "primary_completed_runs": int(runs.status.eq("completed").sum()),
        "primary_failed_runs": run_grid_counts["failed"],
        "primary_missing_runs": run_grid_counts["missing"],
        "primary_nonconverged_completed_runs": run_grid_counts[
            "nonconverged_completed"
        ],
        "primary_reused_k6_runs": int(runs.reused_existing_artifact.fillna(False).sum()),
        "secondary_completed_runs_excluding_k6_aliases": int(
            normalized_runs.status.eq("completed").sum()
        ) if not normalized_runs.empty else 0,
        "occupancy_thresholds": asdict(OccupancyThresholds()),
        "correspondence_thresholds": asdict(CorrespondenceThresholds()),
        "shortlist_rule": [
            "mandatory K=6 reference",
            "modal seedwise AIC winner (ties lower K)",
            "modal seedwise BIC winner (ties lower K)",
            "highest mean within-encoder ARI (ties lower K)",
            "highest mean matched-seed cross-encoder ARI (ties lower K)",
        ],
        "input_hashes": primary["input_hashes"],
        "report": str(report),
        "elapsed_s": time.perf_counter() - started,
        "limitations": [
            "A4_C4 is reused development data, not an untouched test.",
            "Raw likelihood and information criteria are not compared across encoder/PCA spaces.",
            "Reproducibility does not establish behavioral validity.",
        ],
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def aggregate_existing_k_selection(output_dir: str | Path) -> dict[str, Any]:
    """Refresh deterministic summaries without loading/fitting HMM workers."""

    output = Path(output_dir).resolve()
    required = (
        "runs.csv",
        "within_summary.csv",
        "cross_summary.csv",
        "cross_k_pairs.csv",
        "refinement_reproducibility.csv",
        "prior_policy_runs.csv",
        "occupancy_by_state.csv",
        "per_state_temporal_metrics.csv",
        "manifest.json",
    )
    missing = [name for name in required if not (output / name).exists()]
    if missing:
        raise FileNotFoundError(f"cannot aggregate; missing {missing}")
    runs = pd.read_csv(output / "runs.csv")
    run_grid_audit, run_grid_counts = audit_run_grid(runs)
    within = pd.read_csv(output / "within_summary.csv")
    cross = pd.read_csv(output / "cross_summary.csv")
    cross_k = pd.read_csv(output / "cross_k_pairs.csv")
    refinement = pd.read_csv(output / "refinement_reproducibility.csv")
    rankings = seedwise_rankings(runs)
    shortlist = choose_shortlist(rankings, within, cross)
    summary = _primary_summary(runs, within, cross)
    recommendations = recommendation_table(summary, rankings)

    per_state = pd.read_csv(output / "per_state_temporal_metrics.csv")
    occupancy = pd.read_csv(output / "occupancy_by_state.csv")
    occupancy_keys = [
        "architecture",
        "encoder_seed",
        "hmm_seed",
        "K",
        "prior_policy",
        "split",
        "state",
    ]
    for column in ("hard_occupancy", "posterior_occupancy"):
        if column in per_state:
            per_state = per_state.drop(columns=column)
    per_state = per_state.merge(
        occupancy[occupancy_keys + ["hard_occupancy", "posterior_occupancy"]],
        on=occupancy_keys,
        how="left",
        validate="one_to_one",
    )

    policy_runs = pd.read_csv(output / "prior_policy_runs.csv")
    policy_stability = prior_policy_stability_tables(policy_runs)
    policy_summary = (
        policy_runs[policy_runs.status.eq("completed")]
        .groupby(["architecture", "K", "prior_policy"])
        .agg(
            completed_runs=("status", "size"),
            dev_log_likelihood_per_point=("dev_log_likelihood_per_point", "mean"),
            effective_state_ratio=("dev_effective_state_ratio", "mean"),
            bouts_per_minute=("dev_bouts_per_minute", "mean"),
            learned_self_transition=("learned_self_transition_mean", "mean"),
        )
        .reset_index()
    )
    within_policy = policy_stability["prior_policy_within_summary"][
        ["architecture", "K", "prior_policy", "ARI_mean"]
    ].rename(columns={"ARI_mean": "within_encoder_ari_mean"})
    cross_policy = policy_stability["prior_policy_cross_summary"]
    cross_policy = cross_policy[
        cross_policy.hmm_seed_relation.eq("matched_hmm_seed")
    ][["architecture", "K", "prior_policy", "ARI_mean"]].rename(
        columns={"ARI_mean": "cross_encoder_ari_mean"}
    )
    policy_summary = policy_summary.merge(
        within_policy, on=["architecture", "K", "prior_policy"], how="left"
    ).merge(
        cross_policy, on=["architecture", "K", "prior_policy"], how="left"
    )
    refreshed = {
        "run_grid_audit": run_grid_audit,
        "per_state_temporal_metrics": per_state,
        "seedwise_information_criterion_rankings": rankings,
        "shortlist": shortlist,
        "multi_criterion_summary": summary,
        "recommendations": recommendations,
        "prior_policy_comparison": policy_summary,
        **policy_stability,
    }
    _save_tables(output, refreshed)
    make_figures(output, summary, rankings, cross_k, policy_summary)
    report = write_report(
        output,
        summary,
        recommendations,
        shortlist,
        policy_summary,
        pd.read_csv(output / "prior_diagnostics.csv"),
        policy_stability["fixed_vs_normalized_alignment"],
        refinement,
        runs,
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(
        {
            "primary_completed_runs": run_grid_counts["completed"],
            "primary_failed_runs": run_grid_counts["failed"],
            "primary_missing_runs": run_grid_counts["missing"],
            "primary_nonconverged_completed_runs": run_grid_counts[
                "nonconverged_completed"
            ],
            "aggregation_refreshed_without_refitting": True,
            "report": str(report),
        }
    )
    _write_json(output / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path.cwd()))
    parser.add_argument(
        "--output-dir",
        default="evaluation_runs/gnn_9dim_0p04_downstream_v2/k_selection_v1",
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--covariance-floor", type=float, default=1e-3)
    parser.add_argument(
        "--aggregate-existing",
        action="store_true",
        help="Refresh tables/report from completed artifacts without model workers",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.aggregate_existing:
        manifest = aggregate_existing_k_selection(args.output_dir)
    else:
        manifest = run_k_selection(
            root=args.root,
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
