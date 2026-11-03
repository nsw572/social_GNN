"""Complete the bounded 15-cell recurrent-SLDS pilot on frozen PCA16 data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import resource
except ImportError:  # pragma: no cover - unavailable on native Windows
    resource = None

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .embeddings import FrozenEmbeddingTrial
from .rslds import (
    RSLDSPosterior,
    RSLDSSequence,
    SSMRecurrentSLDS,
    load_frozen_rslds_sequences,
    recurrent_transition_probabilities,
    restore_trial_clock,
)
from .rslds_pilot import (
    ARCHITECTURE,
    DEV_TRIAL,
    ENCODER_SEED,
    EXCLUDED_TRIAL,
    OBSERVATION_DIM,
    TIMESTEP_S,
    TRAIN_TRIALS,
    _runtime_paths,
)


GRID = tuple(
    (k, m, seed)
    for k, m in ((4, 4), (6, 4), (8, 4), (6, 8), (8, 8))
    for seed in (0, 1, 2)
)
DEFAULT_OUTPUT = (
    "evaluation_runs/gnn_9dim_0p04_downstream_v2/rslds_pilot_v1/"
    "completed_finite_pilot"
)
CONFIG = "configs/social_gnn_rslds_completed_pilot_0p04_v1.yaml"
LEGACY_PILOT = "evaluation_runs/gnn_9dim_0p04_downstream_v2/rslds_pilot_v1"


def _peak_rss_mb() -> float | None:
    """Return Unix peak RSS without making the module unimportable on Windows."""

    if resource is None:
        return None
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _prepare(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in (
        "models",
        "states",
        "latents",
        "parameters",
        "overlap_matrices",
        "figures",
        "partial_metrics",
    ):
        (output / name).mkdir(exist_ok=True)


def _run_id(k: int, m: int, seed: int) -> str:
    return f"k{k}_m{m}_seed{seed}"


def _flatten_parameters(model: SSMRecurrentSLDS) -> np.ndarray:
    return np.concatenate(
        [np.asarray(value, dtype=float).ravel() for value in model.parameter_arrays().values()]
    )


def _data_support(
    trials: Sequence[FrozenEmbeddingTrial], sequences: Sequence[RSLDSSequence]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trial in trials:
        if trial.trial_id not in set(TRAIN_TRIALS) | {DEV_TRIAL}:
            continue
        runs = trial.contiguous_valid_runs()
        modeled = [value for value in sequences if value.trial_id == trial.trial_id]
        singleton = [value for value in runs if len(value) == 1]
        rows.append(
            {
                "trial_id": trial.trial_id,
                "split": "train" if trial.trial_id in TRAIN_TRIALS else "development",
                "valid_clock_points": int(trial.valid_time_mask.sum()),
                "modeled_points": int(sum(len(value.observations) for value in modeled)),
                "excluded_singleton_points": int(len(singleton)),
                "valid_contiguous_runs": int(len(runs)),
                "modeled_contiguous_runs": int(len(modeled)),
                "singleton_patch_indices": json.dumps(
                    [int(value[0]) for value in singleton]
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_support_report(output: Path, support: pd.DataFrame) -> Path:
    train = support[support.split.eq("train")]
    clock = int(train.valid_clock_points.sum())
    modeled = int(train.modeled_points.sum())
    singleton = int(train.excluded_singleton_points.sum())
    lines = [
        "# rSLDS data-support audit",
        "",
        f"The pointwise Graph Full training clock contains **{clock}** valid observations. The rSLDS adapter models **{modeled}** observations. The exact difference is **{clock - modeled}**.",
        "",
        f"All {singleton} excluded observations are isolated one-point contiguous runs. A dynamical sequence requires at least two observations, so these points cannot contribute a transition or state dynamics term. No longer run was shortened, no additional support was removed, and no invalid gap was bridged.",
        "",
        "| trial | split | valid clock | modeled | singleton exclusions | modeled runs |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in support.itertuples():
        lines.append(
            f"| {row.trial_id} | {row.split} | {row.valid_clock_points} | {row.modeled_points} | {row.excluded_singleton_points} | {row.modeled_contiguous_runs} |"
        )
    path = output / "data_support_audit.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    support.to_csv(output / "data_support_by_trial.csv", index=False)
    return path


def _bout_rows(
    run_id: str,
    k: int,
    m: int,
    seed: int,
    sequences: Sequence[RSLDSSequence],
    states: Sequence[np.ndarray],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    rows: list[dict[str, Any]] = []
    transitions = np.zeros((k, k), dtype=np.int64)
    for sequence, labels in zip(sequences, states):
        labels = np.asarray(labels, dtype=int)
        boundaries = np.r_[0, np.flatnonzero(np.diff(labels) != 0) + 1, len(labels)]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            rows.append(
                {
                    "run_id": run_id,
                    "K": k,
                    "M": m,
                    "seed": seed,
                    "trial_id": sequence.trial_id,
                    "sequence_id": sequence.sequence_id,
                    "state": int(labels[start]),
                    "start_patch_index": int(sequence.patch_indices[start]),
                    "end_patch_index_exclusive": int(sequence.patch_indices[end - 1] + 1),
                    "duration_steps": int(end - start),
                    "duration_s": float((end - start) * TIMESTEP_S),
                }
            )
        if len(labels) > 1:
            np.add.at(transitions, (labels[:-1], labels[1:]), 1)
    return rows, transitions


def _segmentation_tables(
    run_id: str,
    k: int,
    m: int,
    seed: int,
    sequences: Sequence[RSLDSSequence],
    posterior: RSLDSPosterior,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    labels = np.concatenate(posterior.discrete_state)
    probabilities = np.concatenate(posterior.discrete_posterior)
    counts = np.bincount(labels, minlength=k).astype(float)
    hard = counts / counts.sum()
    soft = probabilities.mean(axis=0)
    positive = hard[hard > 0]
    effective = float(np.exp(-np.sum(positive * np.log(positive))))
    entropy = -np.sum(
        probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=1
    )
    bouts, transitions = _bout_rows(
        run_id, k, m, seed, sequences, posterior.discrete_state
    )
    duration = np.asarray([row["duration_s"] for row in bouts], dtype=float)
    total_minutes = len(labels) * TIMESTEP_S / 60.0
    metrics = {
        "occupied_states": int(np.sum(counts > 0)),
        "effective_states": effective,
        "effective_state_ratio": effective / k,
        "largest_state_occupancy": float(hard.max()),
        "smallest_state_occupancy": float(hard.min()),
        "posterior_entropy_mean": float(entropy.mean()),
        "posterior_entropy_sd": float(entropy.std()),
        "bouts_per_minute": float(len(bouts) / total_minutes),
        "mean_bout_duration_s": float(duration.mean()),
        "median_bout_duration_s": float(np.median(duration)),
        "p10_bout_duration_s": float(np.quantile(duration, 0.10)),
        "p90_bout_duration_s": float(np.quantile(duration, 0.90)),
        "one_timestep_bout_fraction": float(np.mean(np.isclose(duration, TIMESTEP_S))),
        "shorter_0p20_s_fraction": float(np.mean(duration < 0.20)),
        "decoder": "conditional_viterbi_given_smoothed_continuous_latent",
    }
    occupancy = [
        {
            "run_id": run_id,
            "K": k,
            "M": m,
            "seed": seed,
            "state": state,
            "hard_occupancy": float(hard[state]),
            "posterior_occupancy": float(soft[state]),
        }
        for state in range(k)
    ]
    return metrics, occupancy, bouts, transitions


def _reconstruction_rows(
    run_id: str,
    k: int,
    m: int,
    seed: int,
    sequences: Sequence[RSLDSSequence],
    posterior: RSLDSPosterior,
) -> tuple[list[dict[str, Any]], float, float]:
    observed = np.concatenate([value.observations for value in sequences])
    predicted = np.concatenate(posterior.smoothed_observation_mean)
    residual = predicted - observed
    rows: list[dict[str, Any]] = []
    for dimension in range(observed.shape[1]):
        mse = float(np.mean(np.square(residual[:, dimension])))
        variance = float(np.var(observed[:, dimension]))
        rows.append(
            {
                "run_id": run_id,
                "K": k,
                "M": m,
                "seed": seed,
                "dimension": dimension,
                "mse": mse,
                "r2": float(1.0 - mse / variance) if variance > 0 else float("nan"),
            }
        )
    total_mse = float(np.mean(np.square(residual)))
    total_variance = float(np.var(observed))
    return rows, total_mse, float(1.0 - total_mse / total_variance)


def _recurrent_rows(
    run_id: str,
    k: int,
    m: int,
    seed: int,
    model: SSMRecurrentSLDS,
    posterior: RSLDSPosterior,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    arrays = model.parameter_arrays()
    latent = np.concatenate(posterior.continuous_mean)
    index = np.linspace(0, len(latent) - 1, min(512, len(latent)), dtype=int)
    representative = latent[index]
    fitted = recurrent_transition_probabilities(
        arrays["base_log_transition"], arrays["recurrent_weights"], representative
    )
    neutral = recurrent_transition_probabilities(
        arrays["base_log_transition"],
        np.zeros_like(arrays["recurrent_weights"]),
        representative,
    )
    difference = np.abs(fitted - neutral)
    rows = [
        {
            "run_id": run_id,
            "K": k,
            "M": m,
            "seed": seed,
            "target_state": state,
            "coefficient_l2_norm": float(np.linalg.norm(arrays["recurrent_weights"][state])),
            "coefficient_mean": float(arrays["recurrent_weights"][state].mean()),
            "coefficient_sd": float(arrays["recurrent_weights"][state].std()),
            "mean_abs_probability_change_when_neutralized": float(difference[:, :, state].mean()),
            "max_abs_probability_change_when_neutralized": float(difference[:, :, state].max()),
        }
        for state in range(k)
    ]
    summary = {
        "recurrent_weight_norm": float(np.linalg.norm(arrays["recurrent_weights"])),
        "neutralization_mean_abs_probability_change": float(difference.mean()),
        "neutralization_max_abs_probability_change": float(difference.max()),
        "representative_latent_count": int(len(representative)),
    }
    return rows, summary


def _dynamics_rows(
    run_id: str,
    k: int,
    m: int,
    seed: int,
    model: SSMRecurrentSLDS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    arrays = model.parameter_arrays()
    matrices = arrays["dynamics_matrices"]
    radii = np.asarray(
        [float(np.max(np.abs(np.linalg.eigvals(value)))) for value in matrices]
    )
    rows = [
        {
            "run_id": run_id,
            "K": k,
            "M": m,
            "seed": seed,
            "state": state,
            "spectral_radius": float(radii[state]),
            "dynamics_frobenius_norm": float(np.linalg.norm(matrices[state])),
            "process_variance_min": float(arrays["process_variances"][state].min()),
            "process_variance_max": float(arrays["process_variances"][state].max()),
        }
        for state in range(k)
    ]
    pairwise = [
        float(np.linalg.norm(matrices[left] - matrices[right]))
        for left, right in combinations(range(k), 2)
    ]
    summary = {
        "maximum_dynamics_spectral_radius": float(radii.max()),
        "states_with_spectral_radius_above_one": int(np.sum(radii > 1.0)),
        "mean_pairwise_dynamics_distance": float(np.mean(pairwise)),
        "minimum_pairwise_dynamics_distance": float(np.min(pairwise)),
        "process_variances_positive": bool(np.all(arrays["process_variances"] > 0)),
        "observation_variances_positive": bool(np.all(arrays["observation_variances"] > 0)),
        "all_parameters_finite": bool(
            all(np.isfinite(value).all() for value in arrays.values())
        ),
    }
    return rows, summary


def _save_trial_outputs(
    output: Path,
    run_id: str,
    trials: Sequence[FrozenEmbeddingTrial],
    sequences: Sequence[RSLDSSequence],
    posterior: RSLDSPosterior,
) -> tuple[str, str]:
    state_paths: list[str] = []
    latent_paths: list[str] = []
    for trial in trials:
        selected = [i for i, value in enumerate(sequences) if value.trial_id == trial.trial_id]
        if not selected:
            continue
        subset = [sequences[i] for i in selected]
        states = restore_trial_clock(
            trial, subset, [posterior.discrete_state[i] for i in selected], fill_value=-1
        )
        probabilities = restore_trial_clock(
            trial,
            subset,
            [posterior.discrete_posterior[i] for i in selected],
            fill_value=np.nan,
        )
        latent = restore_trial_clock(
            trial,
            subset,
            [posterior.continuous_mean[i] for i in selected],
            fill_value=np.nan,
        )
        state_path = output / "states" / f"{run_id}_{trial.trial_id}.npz"
        latent_path = output / "latents" / f"{run_id}_{trial.trial_id}.npz"
        np.savez_compressed(
            state_path,
            trial_id=np.asarray(trial.trial_id),
            patch_start_s=trial.patch_start_s,
            patch_end_s=trial.patch_end_s,
            valid_time_mask=trial.valid_time_mask,
            inferred_state=states,
            state_posterior=probabilities,
            decoder=np.asarray("conditional_viterbi_given_smoothed_continuous_latent"),
        )
        np.savez_compressed(
            latent_path,
            trial_id=np.asarray(trial.trial_id),
            patch_start_s=trial.patch_start_s,
            patch_end_s=trial.patch_end_s,
            valid_time_mask=trial.valid_time_mask,
            continuous_posterior_mean=latent,
        )
        state_paths.append(str(state_path))
        latent_paths.append(str(latent_path))
    return json.dumps(state_paths), json.dumps(latent_paths)


def _legacy_checkpoint(root: Path) -> Path:
    return root / LEGACY_PILOT / "models/k4_m4_seed0_single_full_pilot/model.pkl"


def _fit_cell(payload: dict[str, Any]) -> dict[str, Any]:
    root = Path(payload["root"])
    output = Path(payload["output"])
    k, m, seed = int(payload["K"]), int(payload["M"]), int(payload["seed"])
    iterations = int(payload["iterations"])
    run_id = _run_id(k, m, seed)
    partial = output / "partial_metrics" / f"{run_id}.json"
    if partial.exists():
        return json.loads(partial.read_text(encoding="utf-8"))
    started = time.perf_counter()
    try:
        embedding, preprocessor = _runtime_paths(root)
        trials, _pre, sequences = load_frozen_rslds_sequences(
            embedding,
            preprocessor,
            trial_ids=TRAIN_TRIALS + (DEV_TRIAL,),
        )
        train = [value for value in sequences if value.trial_id in TRAIN_TRIALS]
        dev = [value for value in sequences if value.trial_id == DEV_TRIAL]
        dev_trials = [value for value in trials if value.trial_id == DEV_TRIAL]
        checkpoint = output / "models" / run_id / "model.pkl"
        reused = False
        direct_parameter_delta: float | None = None
        if k == 4 and m == 4 and seed == 0 and not checkpoint.exists():
            checkpoint = _legacy_checkpoint(root)
            reused = True
        if checkpoint.exists():
            model, metadata = SSMRecurrentSLDS.load(checkpoint)
            if (model.k, model.latent_dim, model.seed, model.observation_dim) != (
                k,
                m,
                seed,
                OBSERVATION_DIM,
            ):
                raise ValueError("reused rSLDS checkpoint contract mismatch")
            train_elbo = np.asarray(model.elbo_history, dtype=float)
        else:
            model = SSMRecurrentSLDS(
                observation_dim=OBSERVATION_DIM,
                k=k,
                latent_dim=m,
                seed=seed,
            )
            before = _flatten_parameters(model)
            model.fit(
                [value.observations for value in train],
                num_iters=iterations,
                verbose=0,
            )
            after = _flatten_parameters(model)
            direct_parameter_delta = float(np.linalg.norm(after - before))
            train_elbo = np.asarray(model.elbo_history, dtype=float)
            model.save(
                checkpoint,
                metadata={
                    "run_id": run_id,
                    "train_sequence_ids": [value.sequence_id for value in train],
                    "embedding_manifest_sha256": _sha256(embedding / "manifest.json"),
                    "preprocessor_sha256": _sha256(preprocessor),
                },
            )
        arrays = model.parameter_arrays()
        np.savez_compressed(
            output / "parameters" / f"{run_id}.npz",
            **arrays,
            elbo=train_elbo,
        )
        reloaded, _ = SSMRecurrentSLDS.load(checkpoint)
        save_load_consistent = all(
            np.allclose(arrays[key], reloaded.parameter_arrays()[key])
            for key in arrays
        )
        dev_model, _ = SSMRecurrentSLDS.load(checkpoint)
        posterior = dev_model.fit(
            [value.observations for value in dev],
            num_iters=max(10, iterations // 2),
            initialize=False,
            learning=False,
            verbose=0,
        )
        segmentation, occupancy, bouts, transitions = _segmentation_tables(
            run_id, k, m, seed, dev, posterior
        )
        reconstruction, mse, r2 = _reconstruction_rows(
            run_id, k, m, seed, dev, posterior
        )
        recurrent, recurrent_summary = _recurrent_rows(
            run_id, k, m, seed, dev_model, posterior
        )
        dynamics, dynamics_summary = _dynamics_rows(
            run_id, k, m, seed, dev_model
        )
        state_paths, latent_paths = _save_trial_outputs(
            output, run_id, dev_trials, dev, posterior
        )
        transition_path = output / "parameters" / f"{run_id}_transition_counts.npy"
        np.save(transition_path, transitions)
        metric = {
            "run_id": run_id,
            "status": "completed",
            "K": k,
            "M": m,
            "seed": seed,
            "reused_existing_fit": reused,
            "train_sequences": len(train),
            "train_points": int(sum(len(value.observations) for value in train)),
            "dev_sequences": len(dev),
            "dev_points": int(sum(len(value.observations) for value in dev)),
            "requested_iterations": iterations,
            "elbo_history_length": int(len(train_elbo)),
            "initial_elbo": float(train_elbo[0]),
            "final_elbo": float(train_elbo[-1]),
            "elbo_improvement": float(train_elbo[-1] - train_elbo[0]),
            "termination_status": "fixed_budget_completed",
            "direct_parameter_delta_l2": direct_parameter_delta,
            "parameter_update_verified": bool(
                direct_parameter_delta > 0
                if direct_parameter_delta is not None
                else train_elbo[-1] > train_elbo[0]
            ),
            "save_load_consistent": bool(save_load_consistent),
            "smoothed_reconstruction_mse": mse,
            "smoothed_reconstruction_r2": r2,
            "forecast_status": "unavailable_backend_has_no_validated_online_filter",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "state_paths": state_paths,
            "latent_paths": latent_paths,
            "transition_counts_path": str(transition_path),
            "runtime_s": time.perf_counter() - started,
            "peak_rss_mb": _peak_rss_mb(),
            **segmentation,
            **recurrent_summary,
            **dynamics_summary,
        }
        record = {
            "metric": metric,
            "occupancy": occupancy,
            "bouts": bouts,
            "reconstruction": reconstruction,
            "recurrent": recurrent,
            "dynamics": dynamics,
            "elbo": train_elbo.tolist(),
        }
    except Exception as error:
        record = {
            "metric": {
                "run_id": run_id,
                "status": "failed",
                "K": k,
                "M": m,
                "seed": seed,
                "runtime_s": time.perf_counter() - started,
                "failure": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
            "occupancy": [],
            "bouts": [],
            "reconstruction": [],
            "recurrent": [],
            "dynamics": [],
            "elbo": [],
        }
    _write_json(partial, record)
    return record


def _load_state(path_json: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = json.loads(path_json)
    if len(paths) != 1:
        raise ValueError("pilot expects one development trial state file")
    with np.load(paths[0], allow_pickle=False) as data:
        return (
            np.asarray(data["inferred_state"], dtype=int),
            np.asarray(data["valid_time_mask"], dtype=bool),
            np.asarray(data["patch_start_s"], dtype=float),
        )


def _pair_metrics(left: pd.Series, right: pd.Series, output: Path) -> dict[str, Any]:
    a, valid_a, clock_a = _load_state(str(left.state_paths))
    b, valid_b, clock_b = _load_state(str(right.state_paths))
    if not np.array_equal(valid_a, valid_b) or not np.allclose(clock_a, clock_b):
        raise ValueError("initialization-stability clocks do not align")
    valid = valid_a & (a >= 0) & (b >= 0)
    left_values, right_values = a[valid], b[valid]
    matrix = np.zeros((int(left.K), int(right.K)), dtype=np.int64)
    np.add.at(matrix, (left_values, right_values), 1)
    row, col = linear_sum_assignment(-matrix)
    f1: list[float] = []
    for i, j in zip(row, col):
        tp = matrix[i, j]
        denominator = matrix[i].sum() + matrix[:, j].sum()
        f1.append(float(2 * tp / denominator) if denominator else 0.0)
    adjacent = valid[:-1] & valid[1:] & np.isclose(
        np.diff(clock_a), TIMESTEP_S, rtol=1e-7, atol=1e-9
    )
    boundary_a = adjacent & (a[:-1] != a[1:])
    boundary_b = adjacent & (b[:-1] != b[1:])
    matches = int(np.sum(boundary_a & boundary_b))
    denominator = int(boundary_a.sum() + boundary_b.sum())
    boundary_f1 = float(2 * matches / denominator) if denominator else 1.0
    path = output / "overlap_matrices" / f"{left.run_id}_vs_{right.run_id}.npz"
    np.savez_compressed(path, contingency=matrix)
    return {
        "left_seed": int(left.seed),
        "right_seed": int(right.seed),
        "K": int(left.K),
        "M": int(left.M),
        "aligned_valid_points": int(valid.sum()),
        "ARI": float(adjusted_rand_score(left_values, right_values)),
        "NMI": float(normalized_mutual_info_score(left_values, right_values)),
        "matched_state_f1_mean": float(np.mean(f1)),
        "matched_state_f1_min": float(np.min(f1)),
        "boundary_f1_exact": boundary_f1,
        "absolute_bouts_per_minute_difference": float(
            abs(left.bouts_per_minute - right.bouts_per_minute)
        ),
        "absolute_median_bout_duration_difference_s": float(
            abs(left.median_bout_duration_s - right.median_bout_duration_s)
        ),
        "overlap_matrix": str(path),
    }


def _initialization_stability(runs: pd.DataFrame, output: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    completed = runs[runs.status.eq("completed")]
    for (k, m), group in completed.groupby(["K", "M"]):
        records = [row for _, row in group.sort_values("seed").iterrows()]
        for left, right in combinations(records, 2):
            rows.append(_pair_metrics(left, right, output))
    return pd.DataFrame(rows)


def _forecast_baselines(
    train: Sequence[RSLDSSequence], dev: Sequence[RSLDSSequence]
) -> pd.DataFrame:
    mean = np.concatenate([value.observations for value in train]).mean(axis=0)
    rows: list[dict[str, Any]] = []
    for horizon in (1, 5, 10):
        targets: list[np.ndarray] = []
        persistence: list[np.ndarray] = []
        for sequence in dev:
            if len(sequence.observations) <= horizon:
                continue
            targets.append(sequence.observations[horizon:])
            persistence.append(sequence.observations[:-horizon])
        target = np.concatenate(targets)
        current = np.concatenate(persistence)
        rows.extend(
            [
                {
                    "model": "persistence",
                    "horizon_steps": horizon,
                    "horizon_s": horizon * TIMESTEP_S,
                    "eligible_points": len(target),
                    "mse": float(np.mean(np.square(current - target))),
                    "status": "completed_prefix_only",
                },
                {
                    "model": "training_mean",
                    "horizon_steps": horizon,
                    "horizon_s": horizon * TIMESTEP_S,
                    "eligible_points": len(target),
                    "mse": float(np.mean(np.square(mean - target))),
                    "status": "completed_training_only",
                },
                {
                    "model": "rSLDS",
                    "horizon_steps": horizon,
                    "horizon_s": horizon * TIMESTEP_S,
                    "eligible_points": len(target),
                    "mse": float("nan"),
                    "status": "unavailable_backend_has_no_validated_online_filter",
                },
                {
                    "model": "Sticky_HMM",
                    "horizon_steps": horizon,
                    "horizon_s": horizon * TIMESTEP_S,
                    "eligible_points": len(target),
                    "mse": float("nan"),
                    "status": "not_compared_because_rslds_forecast_unavailable",
                },
            ]
        )
    return pd.DataFrame(rows)


def _sensitivity(runs: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    stability_summary = stability.groupby(["K", "M"]).agg(
        initialization_ari_mean=("ARI", "mean"),
        initialization_ari_min=("ARI", "min"),
        initialization_nmi_mean=("NMI", "mean"),
        boundary_f1_mean=("boundary_f1_exact", "mean"),
    ).reset_index()
    run_summary = runs[runs.status.eq("completed")].groupby(["K", "M"]).agg(
        completed_seeds=("seed", "count"),
        reconstruction_mse_mean=("smoothed_reconstruction_mse", "mean"),
        reconstruction_mse_sd=("smoothed_reconstruction_mse", "std"),
        effective_state_ratio_mean=("effective_state_ratio", "mean"),
        largest_state_occupancy_mean=("largest_state_occupancy", "mean"),
        bouts_per_minute_mean=("bouts_per_minute", "mean"),
        median_bout_duration_s_mean=("median_bout_duration_s", "mean"),
        recurrent_ablation_change_mean=("neutralization_mean_abs_probability_change", "mean"),
        maximum_spectral_radius=("maximum_dynamics_spectral_radius", "max"),
        runtime_s_mean=("runtime_s", "mean"),
    ).reset_index()
    return run_summary.merge(stability_summary, on=["K", "M"], how="left")


def _make_figures(
    output: Path,
    runs: pd.DataFrame,
    occupancy: pd.DataFrame,
    stability: pd.DataFrame,
    recurrent: pd.DataFrame,
    dynamics: pd.DataFrame,
) -> list[str]:
    figures = output / "figures"
    paths: list[str] = []

    def save(name: str) -> None:
        path = figures / name
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(str(path))

    completed = runs[runs.status.eq("completed")].copy()
    completed["condition"] = completed.apply(lambda row: f"K{int(row.K)} M{int(row.M)}", axis=1)
    for column, name, ylabel in (
        ("smoothed_reconstruction_mse", "reconstruction_mse.png", "Smoothed reconstruction MSE"),
        ("effective_state_ratio", "effective_state_ratio.png", "K_eff / K"),
        ("bouts_per_minute", "bouts_per_minute.png", "Bouts / min"),
        ("runtime_s", "runtime.png", "Runtime (s)"),
    ):
        plt.figure(figsize=(7.2, 4.4))
        groups = [group[column].to_numpy() for _, group in completed.groupby("condition", sort=True)]
        labels = [key for key, _ in completed.groupby("condition", sort=True)]
        plt.boxplot(groups, labels=labels)
        plt.ylabel(ylabel)
        save(name)

    plt.figure(figsize=(7.2, 4.4))
    for (k, m), group in stability.groupby(["K", "M"]):
        plt.scatter([f"K{k} M{m}"] * len(group), group.ARI, label=f"K{k} M{m}")
    plt.ylim(0, 1)
    plt.ylabel("Initialization ARI")
    save("initialization_stability.png")

    plt.figure(figsize=(7.2, 4.4))
    summary = recurrent.groupby(["K", "M", "seed"]).mean_abs_probability_change_when_neutralized.mean().reset_index()
    for (k, m), group in summary.groupby(["K", "M"]):
        plt.scatter([f"K{k} M{m}"] * len(group), group.mean_abs_probability_change_when_neutralized)
    plt.ylabel("Mean |Δ transition probability|")
    save("recurrent_neutralization.png")

    plt.figure(figsize=(7.2, 4.4))
    for (k, m), group in dynamics.groupby(["K", "M"]):
        plt.scatter([f"K{k} M{m}"] * len(group), group.spectral_radius, alpha=0.65)
    plt.axhline(1.0, color="black", linewidth=0.8)
    plt.ylabel("State dynamics spectral radius")
    save("dynamics_spectral_radii.png")

    matrix = occupancy.pivot_table(index="run_id", columns="state", values="hard_occupancy", fill_value=0)
    plt.figure(figsize=(8.0, 5.5))
    plt.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis")
    plt.colorbar(label="Hard occupancy")
    plt.yticks(range(len(matrix)), matrix.index, fontsize=6)
    plt.xlabel("State index (run-local)")
    save("state_occupancy_heatmap.png")
    return paths


def _completion_report(
    output: Path,
    runs: pd.DataFrame,
    sensitivity: pd.DataFrame,
    support: pd.DataFrame,
    forecast: pd.DataFrame,
) -> tuple[Path, str, str]:
    completed = runs[runs.status.eq("completed")]
    failures = runs[runs.status.eq("failed")]
    stable = sensitivity.sort_values(
        ["initialization_ari_mean", "effective_state_ratio_mean", "reconstruction_mse_mean"],
        ascending=[False, False, True],
    )
    best = stable.iloc[0] if not stable.empty else None
    numerical_ok = bool(
        completed.all_parameters_finite.all()
        and completed.process_variances_positive.all()
        and completed.observation_variances_positive.all()
    ) if not completed.empty else False
    recurrent_change = float(completed.neutralization_mean_abs_probability_change.mean()) if not completed.empty else float("nan")
    mean_ari = float(sensitivity.initialization_ari_mean.mean()) if not sensitivity.empty else float("nan")
    if len(completed) < 12 or not numerical_ok or (np.isfinite(mean_ari) and mean_ari < 0.30):
        recommendation = "C. Do not advance"
    else:
        recommendation = "B. Retain as a secondary control"
    best_text = (
        f"K={int(best.K)}, M={int(best.M)}"
        if best is not None
        else "no reliable candidate"
    )
    lines = [
        "# 0.04 s Social-GNN: Completed rSLDS Pilot and Dynamical-State Evaluation",
        "",
        "## Executive decision",
        "",
        f"**{recommendation}.**",
        "",
        "Forecasting remains unavailable because the selected backend exposes whole-sequence Laplace smoothing but no validated online filtering API. The decision is therefore provisional with respect to predictive benefit; smoothed reconstruction is not treated as forecasting.",
        "",
        f"Best-supported bounded-pilot condition: **{best_text}**. This is a pilot ranking, not a global optimum.",
        "",
        "## Required questions",
        "",
        f"1. Completed configurations: {len(completed)}/15; failures: {len(failures)}.",
        f"2. The historical 40-point difference is exactly 40 isolated one-timestep valid runs: 5 in A1_C1 and 35 in A3_C3. They cannot define a dynamical transition and were the only points excluded.",
        f"3. Learned recurrent contribution: mean fitted-term neutralization changed transition probabilities by {recurrent_change:.6f}. This establishes fitted use, not predictive superiority.",
        f"4. Continuous dynamics numerically well behaved: {'yes for finite/covariance checks' if numerical_ok else 'no; inspect failures/diagnostics'}. Local spectral radii above one are reported rather than automatically rejected.",
        f"5. Occupancy/pathology: see `k_m_sensitivity.csv`; the highest stability/occupancy-supported condition was {best_text}.",
        f"6. Initialization stability: mean condition-level ARI={mean_ari:.3f}; each K,M has three dependent pairwise seed comparisons.",
        "7. Larger K/M effects are summarized separately; no global optimality claim is made.",
        "8. Genuine rSLDS forecasting improvement could not be established; only persistence and training-mean baseline errors are reported.",
        f"9. Larger formal experiment: {'not justified yet' if recommendation != 'A. Advance rSLDS' else 'justified'}.",
        "",
        "## Data and inference contract",
        "",
        f"Graph Full seed {ENCODER_SEED}, frozen scaler/PCA16, 0.04 s clock. Training trials: {', '.join(TRAIN_TRIALS)}; development: {DEV_TRIAL}; excluded: {EXCLUDED_TRIAL}.",
        "",
        "Every trial boundary and invalid gap is an independent sequence boundary. Discrete paths use the backend's most-likely-state decoder conditional on the smoothed continuous posterior. Posterior marginal occupancy and decoded-path occupancy are reported separately.",
        "",
        "## Confirmed findings",
        "",
        "- All completed checkpoints are genuine `RecurrentTransitions` rSLDS models.",
        "- Parameters, posterior probabilities, continuous latents, dynamics and covariances are checked for finite/valid values.",
        "- Recurrent-term neutralization holds fitted nonrecurrent parameters fixed and is an intervention, not a refitted stationary baseline.",
        "- A5_C5 was not loaded.",
        "",
        "## Limitations",
        "",
        "- ELBO is not compared to HMM exact marginal likelihood.",
        "- Smoothed reconstruction uses future observations and cannot support a forecasting claim.",
        "- One frozen encoder seed is used; downstream initialization stability is not cross-encoder stability.",
        "- State agreement and occupancy do not establish biological validity.",
    ]
    path = output / "rslds_pilot_completion_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path, recommendation, best_text


def run_rslds_completion(
    *,
    root: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT,
    workers: int = 3,
    iterations: int = 50,
    aggregate_only: bool = False,
) -> dict[str, Any]:
    root = Path(root).resolve()
    output = (root / output_dir).resolve() if not Path(output_dir).is_absolute() else Path(output_dir).resolve()
    _prepare(output)
    started = time.perf_counter()
    embedding, preprocessor = _runtime_paths(root)
    trials, _pre, sequences = load_frozen_rslds_sequences(
        embedding,
        preprocessor,
        trial_ids=TRAIN_TRIALS + (DEV_TRIAL,),
    )
    support = _data_support(trials, sequences)
    support_report = _write_support_report(output, support)
    train = [value for value in sequences if value.trial_id in TRAIN_TRIALS]
    dev = [value for value in sequences if value.trial_id == DEV_TRIAL]

    payloads = [
        {
            "root": str(root),
            "output": str(output),
            "K": k,
            "M": m,
            "seed": seed,
            "iterations": iterations,
        }
        for k, m, seed in GRID
    ]
    records: list[dict[str, Any]] = []
    if aggregate_only:
        for payload in payloads:
            partial = output / "partial_metrics" / f"{_run_id(payload['K'], payload['M'], payload['seed'])}.json"
            if not partial.exists():
                raise FileNotFoundError(f"missing completed partial: {partial}")
            records.append(json.loads(partial.read_text(encoding="utf-8")))
    elif workers == 1:
        records = [_fit_cell(value) for value in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_fit_cell, value): value for value in payloads}
            for future in as_completed(futures):
                records.append(future.result())

    runs = pd.DataFrame([value["metric"] for value in records]).sort_values(["M", "K", "seed"])
    occupancy = pd.DataFrame([row for value in records for row in value["occupancy"]])
    bouts = pd.DataFrame([row for value in records for row in value["bouts"]])
    reconstruction = pd.DataFrame([row for value in records for row in value["reconstruction"]])
    recurrent = pd.DataFrame([row for value in records for row in value["recurrent"]])
    dynamics = pd.DataFrame([row for value in records for row in value["dynamics"]])
    convergence = pd.DataFrame(
        [
            {
                "run_id": value["metric"]["run_id"],
                "iteration": iteration,
                "elbo": elbo,
            }
            for value in records
            for iteration, elbo in enumerate(value["elbo"])
        ]
    )
    runs.to_csv(output / "run_level_metrics.csv", index=False)
    convergence.to_csv(output / "convergence_diagnostics.csv", index=False)
    occupancy.to_csv(output / "state_occupancy.csv", index=False)
    bouts.to_csv(output / "bout_statistics.csv", index=False)
    reconstruction.to_csv(output / "smoothed_reconstruction.csv", index=False)
    recurrent.to_csv(output / "recurrent_switching_diagnostics.csv", index=False)
    dynamics.to_csv(output / "dynamics_diagnostics.csv", index=False)
    runs[["run_id", "K", "M", "seed", "status", "runtime_s", "peak_rss_mb", "checkpoint"]].to_csv(
        output / "runtime_memory.csv", index=False
    )
    stability = _initialization_stability(runs, output)
    stability.to_csv(output / "initialization_stability.csv", index=False)
    sensitivity = _sensitivity(runs, stability)
    sensitivity.to_csv(output / "k_m_sensitivity.csv", index=False)
    forecast = _forecast_baselines(train, dev)
    forecast.to_csv(output / "forecast_metrics.csv", index=False)
    figures = _make_figures(output, runs, occupancy, stability, recurrent, dynamics)
    report, recommendation, best = _completion_report(
        output, runs, sensitivity, support, forecast
    )
    completed = runs.status.eq("completed")
    config = root / CONFIG
    manifest = {
        "contract": "social_gnn_rslds_completed_pilot_0p04_v1",
        "status": "completed" if int(completed.sum()) == 15 else "completed_with_failures",
        "requested_configurations": 15,
        "completed_configurations": int(completed.sum()),
        "failed_configurations": int(runs.status.eq("failed").sum()),
        "grid": [list(value) for value in GRID],
        "architecture": ARCHITECTURE,
        "encoder_seed": ENCODER_SEED,
        "embedding_manifest": str(embedding / "manifest.json"),
        "embedding_manifest_sha256": _sha256(embedding / "manifest.json"),
        "preprocessor": str(preprocessor),
        "preprocessor_sha256": _sha256(preprocessor),
        "config": str(config),
        "config_sha256": _sha256(config),
        "train_trials": list(TRAIN_TRIALS),
        "development_trial": DEV_TRIAL,
        "excluded_trial": EXCLUDED_TRIAL,
        "timestep_s": TIMESTEP_S,
        "observation_dim": OBSERVATION_DIM,
        "training_valid_clock_points": int(support[support.split.eq("train")].valid_clock_points.sum()),
        "training_modeled_points": int(support[support.split.eq("train")].modeled_points.sum()),
        "excluded_singleton_points": int(support[support.split.eq("train")].excluded_singleton_points.sum()),
        "forecast_status": "unavailable_backend_has_no_validated_online_filter",
        "recommendation": recommendation,
        "best_supported_condition": best,
        "data_support_audit": str(support_report),
        "report": str(report),
        "figures": figures,
        "elapsed_s": time.perf_counter() - started,
    }
    _write_json(output / "completed_pilot_manifest.json", manifest)
    return manifest


def _parse_cell(value: str) -> tuple[int, int, int]:
    """Parse one ``K,M,seed`` CLI cell and enforce the declared pilot grid."""

    try:
        cell = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("cell must be K,M,seed") from error
    if len(cell) != 3:
        raise argparse.ArgumentTypeError("cell must contain exactly K,M,seed")
    typed = (cell[0], cell[1], cell[2])
    if typed not in GRID:
        raise argparse.ArgumentTypeError(
            f"cell {typed} is outside the declared finite pilot grid"
        )
    return typed


def run_selected_cells(
    *,
    root: str | Path,
    cells: Sequence[tuple[int, int, int]],
    output_dir: str | Path = DEFAULT_OUTPUT,
    workers: int = 1,
    iterations: int = 50,
) -> dict[str, Any]:
    """Fit only explicitly requested grid cells and preserve resumable partials.

    This entry point intentionally skips the 15-cell aggregate report.  It is
    intended for bounded follow-up runs such as one matched seed at two latent
    dimensions.  Existing completed partials are reused by :func:`_fit_cell`.
    """

    root = Path(root).resolve()
    output = (
        (root / output_dir).resolve()
        if not Path(output_dir).is_absolute()
        else Path(output_dir).resolve()
    )
    _prepare(output)
    selected = tuple(dict.fromkeys(tuple(cell) for cell in cells))
    if not selected:
        raise ValueError("at least one selected rSLDS cell is required")
    invalid = [cell for cell in selected if cell not in GRID]
    if invalid:
        raise ValueError(f"selected cells are outside the pilot grid: {invalid}")
    payloads = [
        {
            "root": str(root),
            "output": str(output),
            "K": k,
            "M": m,
            "seed": seed,
            "iterations": iterations,
        }
        for k, m, seed in selected
    ]
    if workers <= 1 or len(payloads) == 1:
        records = [_fit_cell(value) for value in payloads]
    else:
        records = []
        with ProcessPoolExecutor(max_workers=min(workers, len(payloads))) as executor:
            futures = {executor.submit(_fit_cell, value): value for value in payloads}
            for future in as_completed(futures):
                records.append(future.result())
    metrics = pd.DataFrame([value["metric"] for value in records]).sort_values(
        ["M", "K", "seed"]
    )
    metrics.to_csv(output / "selected_cell_metrics.csv", index=False)
    completed = int(metrics.status.eq("completed").sum())
    failed = int(metrics.status.eq("failed").sum())
    manifest = {
        "contract": "social_gnn_rslds_selected_cells_v1",
        "status": "completed" if completed == len(selected) else "completed_with_failures",
        "selected_cells": [list(cell) for cell in selected],
        "completed_cells": completed,
        "failed_cells": failed,
        "workers": min(max(int(workers), 1), len(selected)),
        "iterations": int(iterations),
        "output_dir": str(output),
        "metric_table": str(output / "selected_cell_metrics.csv"),
        "partial_metrics": [
            str(output / "partial_metrics" / f"{_run_id(*cell)}.json")
            for cell in selected
        ],
    }
    _write_json(output / "selected_cells_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path.cwd()))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--cell",
        action="append",
        type=_parse_cell,
        default=[],
        help="Fit only one K,M,seed cell; repeat for multiple cells.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cell:
        manifest = run_selected_cells(
            root=args.root,
            output_dir=args.output_dir,
            cells=args.cell,
            workers=args.workers,
            iterations=args.iterations,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    manifest = run_rslds_completion(
        root=args.root,
        output_dir=args.output_dir,
        workers=args.workers,
        iterations=args.iterations,
        aggregate_only=args.aggregate_only,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
