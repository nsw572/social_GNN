"""Unified KMeans/GMM/HMM/HSMM benchmark for one frozen representation."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits

from social_gnn.evaluation import TRAIT_NAMES, load_trials

from .embeddings import FrozenEmbeddingTrial, load_embedding_directory
from .evaluation import DecodedTrial, bout_rows, decode_trial, segmentation_metrics, stability_metrics
from .gaussian_hmm import GaussianHMM
from .gaussian_hsmm import GaussianHSMM
from .independent import GaussianMixtureStateModel, KMeansStateModel
from .preprocessing import EmbeddingPreprocessor


ALGORITHMS = frozenset({"kmeans", "gmm", "hmm", "hsmm"})


def _sequences(
    trials: Iterable[FrozenEmbeddingTrial], preprocessor: EmbeddingPreprocessor
) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for trial in trials:
        transformed, _indices = preprocessor.transform_trial_sequences(trial)
        result.extend(transformed)
    return result


def _run_name(algorithm: str, states: int, seed: int, max_duration: int | None) -> str:
    suffix = f"_dmax{max_duration}" if algorithm == "hsmm" else ""
    return f"{algorithm}_k{states}_seed{seed}{suffix}"


def _fit_model(
    algorithm: str,
    states: int,
    seed: int,
    sequences: list[np.ndarray],
    *,
    covariance_floor: float,
    hmm_max_iter: int,
    hsmm_max_iter: int,
    gmm_max_iter: int,
    tolerance: float,
    max_duration: int | None,
):
    if algorithm == "kmeans":
        return KMeansStateModel(states, seed=seed, n_init=20).fit(sequences)
    if algorithm == "gmm":
        return GaussianMixtureStateModel(
            states,
            seed=seed,
            covariance_floor=covariance_floor,
            max_iter=gmm_max_iter,
            tolerance=tolerance,
        ).fit(sequences)
    if algorithm == "hmm":
        return GaussianHMM(
            n_states=states,
            covariance_floor=covariance_floor,
            seed=seed,
        ).fit(sequences, max_iter=hmm_max_iter, tolerance=tolerance)
    if algorithm == "hsmm":
        if max_duration is None:
            raise ValueError("HSMM requires max_duration")
        return GaussianHSMM(
            n_states=states,
            max_duration=max_duration,
            covariance_floor=covariance_floor,
            seed=seed,
        ).fit(sequences, max_iter=hsmm_max_iter, tolerance=tolerance)
    raise ValueError(f"unsupported algorithm {algorithm!r}")


def _iterations(model: Any) -> int:
    if hasattr(model, "history_"):
        return len(model.history_)
    return int(getattr(model, "n_iter_", 0))


def _save_checkpoint(model: Any, path: Path, metadata: dict[str, Any]) -> Path:
    return model.save(path, metadata=metadata)


def _save_assignments(output: Path, decoded: list[DecodedTrial]) -> tuple[list[str], Path]:
    output.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    bouts: list[dict[str, Any]] = []
    for value in decoded:
        path = output / f"{value.trial.trial_id}_states.npz"
        np.savez_compressed(
            path,
            trial_id=np.asarray(value.trial.trial_id),
            patch_index=np.arange(value.predicted_state.size),
            patch_start_s=value.trial.patch_start_s,
            patch_end_s=value.trial.patch_end_s,
            patch_center_s=value.trial.patch_center_s,
            valid_time_mask=value.trial.valid_time_mask,
            predicted_state=value.predicted_state,
            state_posterior=value.posterior,
            state_confidence=value.state_confidence,
        )
        table = pd.DataFrame(
            {
                "trial_id": value.trial.trial_id,
                "patch_index": np.arange(value.predicted_state.size),
                "patch_start_s": value.trial.patch_start_s,
                "patch_end_s": value.trial.patch_end_s,
                "patch_center_s": value.trial.patch_center_s,
                "valid_time_mask": value.trial.valid_time_mask,
                "predicted_state": value.predicted_state,
                "state_confidence": value.state_confidence,
            }
        )
        csv_path = output / f"{value.trial.trial_id}_states.csv"
        table.to_csv(csv_path, index=False)
        files.extend((str(path), str(csv_path)))
        bouts.extend(bout_rows(value))
    bout_path = output / "bouts.csv"
    pd.DataFrame(bouts).to_csv(bout_path, index=False)
    return files, bout_path


def _measurement_rows(
    decoded: list[DecodedTrial],
    package_by_id: dict[str, Any],
    n_states: int,
    run_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state in range(n_states):
        node_parts: list[np.ndarray] = []
        edge_parts: list[np.ndarray] = []
        confidence_parts: list[np.ndarray] = []
        for value in decoded:
            package = package_by_id[value.trial.trial_id]
            index = np.flatnonzero(value.predicted_state == state)
            if index.size:
                node_parts.append(package.node_features[index])
                edge_parts.append(package.edge_values[index])
                confidence_parts.append(package.edge_confidence[index])
        row: dict[str, Any] = {**run_metadata, "state": state, "patches": 0}
        if node_parts:
            nodes = np.concatenate(node_parts, axis=0)
            edges = np.concatenate(edge_parts, axis=0)
            confidence = np.concatenate(confidence_parts, axis=0)
            row["patches"] = int(nodes.shape[0])
            for feature in range(nodes.shape[-1]):
                row[f"node_feature_{feature}_pair_mean"] = float(
                    nodes[:, :, feature].mean()
                )
                row[f"node_feature_{feature}_abs_pair_difference"] = float(
                    np.abs(nodes[:, 0, feature] - nodes[:, 1, feature]).mean()
                )
            for trait, name in enumerate(TRAIT_NAMES):
                trait_values = np.stack(
                    (edges[:, 0, 1, trait], edges[:, 1, 0, trait]), axis=1
                )
                trait_confidence = np.stack(
                    (
                        confidence[:, 0, 1, trait],
                        confidence[:, 1, 0, trait],
                    ),
                    axis=1,
                )
                denominator = float(trait_confidence.sum())
                row[f"edge_{name}_confidence_weighted_mean"] = (
                    float((trait_values * trait_confidence).sum() / denominator)
                    if denominator > 0
                    else float("nan")
                )
                row[f"edge_{name}_mean_confidence"] = float(trait_confidence.mean())
                row[f"edge_{name}_valid_coverage"] = float(
                    np.mean(trait_confidence > 0)
                )
                row[f"edge_{name}_directional_abs_difference"] = float(
                    np.abs(trait_values[:, 0] - trait_values[:, 1]).mean()
                )
        rows.append(row)
    return rows


def _silhouette(
    model: Any,
    validation_sequences: list[np.ndarray],
    *,
    seed: int,
    sample_size: int = 5000,
) -> float:
    values = np.concatenate(validation_sequences, axis=0)
    labels = model.predict(values)
    if np.unique(labels).size < 2 or np.unique(labels).size >= values.shape[0]:
        return float("nan")
    if values.shape[0] > sample_size:
        rng = np.random.default_rng(seed)
        index = np.sort(rng.choice(values.shape[0], sample_size, replace=False))
        values = values[index]
        labels = labels[index]
    return float(silhouette_score(values, labels, metric="euclidean"))


def _duration_diagnostics(
    algorithm: str,
    model: Any,
    decoded: list[DecodedTrial],
    n_states: int,
) -> dict[str, float]:
    bouts = [row for value in decoded for row in bout_rows(value)]
    observed = np.asarray([row["timesteps"] for row in bouts], dtype=np.float64)
    states = np.concatenate(
        [value.predicted_state[value.predicted_state >= 0] for value in decoded]
    )
    occupancy = np.bincount(states, minlength=n_states).astype(np.float64)
    occupancy /= max(occupancy.sum(), 1.0)
    expected = float("nan")
    if algorithm == "hmm":
        stay = np.clip(np.diag(model.transition_), 0.0, 1.0 - 1e-9)
        expected = float(np.dot(occupancy, 1.0 / (1.0 - stay)))
    elif algorithm == "hsmm":
        probability = np.exp(model.duration_log_probability())
        durations = np.arange(1, probability.shape[1] + 1, dtype=np.float64)
        expected = float(np.dot(occupancy, probability @ durations))
    observed_mean = float(observed.mean()) if observed.size else float("nan")
    return {
        "observed_mean_bout_patches": observed_mean,
        "fitted_duration_expected_patches": expected,
        "fitted_to_observed_duration_ratio": (
            expected / observed_mean
            if np.isfinite(expected) and observed_mean > 0
            else float("nan")
        ),
    }


def run_downstream_benchmark(
    *,
    embedding_dir: str | Path,
    output_dir: str | Path,
    experiment_config: str | Path,
    train_trial_ids: list[str],
    validation_trial_ids: list[str],
    algorithms: list[str],
    state_counts: list[int],
    seeds: list[int],
    max_durations: list[int],
    encoder_name: str,
    encoder_seed: int,
    pca_dim: int = 16,
    covariance_floor: float = 1e-3,
    hmm_max_iter: int = 40,
    hsmm_max_iter: int = 20,
    gmm_max_iter: int = 100,
    tolerance: float = 1e-4,
    thread_limit: int | None = 8,
) -> dict[str, Any]:
    """Fit all requested models while preserving failures and trial boundaries."""

    unknown = sorted(set(algorithms) - ALGORITHMS)
    if unknown:
        raise ValueError(f"unknown algorithms: {unknown}")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    trials = load_embedding_directory(embedding_dir)
    by_id = {trial.trial_id: trial for trial in trials}
    requested = set(train_trial_ids) | set(validation_trial_ids)
    missing = sorted(requested - set(by_id))
    if missing:
        raise ValueError(f"embedding directory is missing trials {missing}")
    if set(train_trial_ids) & set(validation_trial_ids):
        raise ValueError("train and validation trials overlap")
    train_trials = [by_id[value] for value in train_trial_ids]
    validation_trials = [by_id[value] for value in validation_trial_ids]
    preprocessor = EmbeddingPreprocessor().fit(
        train_trials,
        pca_dim=pca_dim,
        seed=0,
    )
    preprocess_path = preprocessor.save(output / "preprocessing" / f"pca_{pca_dim}.npz")
    train_sequences = _sequences(train_trials, preprocessor)
    validation_sequences = _sequences(validation_trials, preprocessor)
    train_steps = int(sum(len(value) for value in train_sequences))
    validation_steps = int(sum(len(value) for value in validation_sequences))
    _config, packages = load_trials(experiment_config)
    package_by_id = {value.trial_id: value for value in packages}

    records: list[dict[str, Any]] = []
    measurement_rows: list[dict[str, Any]] = []
    states_by_group: dict[tuple[str, int, int | None], list[np.ndarray]] = {}
    for algorithm in algorithms:
        durations: list[int | None] = max_durations if algorithm == "hsmm" else [None]
        for states in state_counts:
            for max_duration in durations:
                group = (algorithm, states, max_duration)
                states_by_group[group] = []
                for seed in seeds:
                    name = _run_name(algorithm, states, seed, max_duration)
                    run_dir = output / algorithm / name
                    base = {
                        "encoder": encoder_name,
                        "encoder_seed": int(encoder_seed),
                        "algorithm": algorithm,
                        "downstream_seed": int(seed),
                        "K": int(states),
                        "max_duration_patches": max_duration,
                        "pca_dim": int(preprocessor.output_dim),
                        "status": "failed",
                    }
                    started = time.perf_counter()
                    try:
                        with threadpool_limits(limits=thread_limit):
                            model = _fit_model(
                                algorithm,
                                states,
                                seed,
                                train_sequences,
                                covariance_floor=covariance_floor,
                                hmm_max_iter=hmm_max_iter,
                                hsmm_max_iter=hsmm_max_iter,
                                gmm_max_iter=gmm_max_iter,
                                tolerance=tolerance,
                                max_duration=max_duration,
                            )
                        train_fit = float(model.score(train_sequences))
                        validation_fit = float(model.score(validation_sequences))
                        decoded = [
                            decode_trial(model, preprocessor, trial)
                            for trial in validation_trials
                        ]
                        joined_state = np.concatenate(
                            [value.predicted_state[value.predicted_state >= 0] for value in decoded]
                        )
                        states_by_group[group].append(joined_state)
                        metrics = segmentation_metrics(decoded, states)
                        duration_metrics = _duration_diagnostics(
                            algorithm, model, decoded, states
                        )
                        if algorithm == "kmeans":
                            metrics["mean_posterior_entropy"] = float("nan")
                            metrics["mean_state_confidence"] = float("nan")
                        checkpoint = _save_checkpoint(
                            model,
                            run_dir / "model.npz",
                            {
                                **base,
                                "embedding_dir": str(Path(embedding_dir).resolve()),
                                "preprocessor": str(preprocess_path),
                                "train_trials": train_trial_ids,
                                "validation_trials": validation_trial_ids,
                            },
                        )
                        assignment_files, bout_path = _save_assignments(
                            run_dir / "assignments", decoded
                        )
                        parameter_count = int(model.parameter_count())
                        aic = float("nan")
                        bic = float("nan")
                        if algorithm == "gmm":
                            aic = model.aic(train_sequences)
                            bic = model.bic(train_sequences)
                        elif algorithm in {"hmm", "hsmm"}:
                            aic = 2 * parameter_count - 2 * train_fit
                            bic = np.log(train_steps) * parameter_count - 2 * train_fit
                        cap_fraction = float("nan")
                        if algorithm == "hsmm":
                            bouts = [row for value in decoded for row in bout_rows(value)]
                            cap_fraction = float(
                                np.mean(
                                    [row["timesteps"] == max_duration for row in bouts]
                                )
                            )
                        record = {
                            **base,
                            "status": "completed",
                            "train_fit_total": train_fit,
                            "validation_fit_total": validation_fit,
                            "train_fit_per_patch": train_fit / train_steps,
                            "validation_fit_per_patch": validation_fit / validation_steps,
                            "train_inertia": (
                                float(model.inertia(train_sequences))
                                if algorithm == "kmeans"
                                else float("nan")
                            ),
                            "validation_inertia": (
                                float(model.inertia(validation_sequences))
                                if algorithm == "kmeans"
                                else float("nan")
                            ),
                            "silhouette_validation": _silhouette(
                                model, validation_sequences, seed=seed
                            ),
                            "aic_train": aic,
                            "bic_train": bic,
                            "parameter_count": parameter_count,
                            "iterations": _iterations(model),
                            "converged": bool(getattr(model, "converged_", True)),
                            "posterior_probabilities_available": algorithm != "kmeans",
                            "fraction_bouts_at_duration_cap": cap_fraction,
                            "checkpoint": str(checkpoint),
                            "assignment_files": json.dumps(assignment_files),
                            "bout_table": str(bout_path),
                            "elapsed_s": time.perf_counter() - started,
                            "failure": "",
                            **metrics,
                            **duration_metrics,
                        }
                        measurement_rows.extend(
                            _measurement_rows(
                                decoded,
                                package_by_id,
                                states,
                                {**base, "status": "completed"},
                            )
                        )
                    except Exception as error:  # preserve failed runs for audit
                        record = {
                            **base,
                            "elapsed_s": time.perf_counter() - started,
                            "failure": f"{type(error).__name__}: {error}",
                            "traceback": traceback.format_exc(),
                        }
                    records.append(record)
                    pd.DataFrame(records).to_csv(output / "runs.csv", index=False)
                    pd.DataFrame(measurement_rows).to_csv(
                        output / "state_measurements.csv", index=False
                    )

    frame = pd.DataFrame(records)
    summaries: list[dict[str, Any]] = []
    for group, labels in states_by_group.items():
        algorithm, states, max_duration = group
        subset = frame[
            (frame["algorithm"] == algorithm)
            & (frame["K"] == states)
            & (frame["max_duration_patches"].fillna(-1) == (max_duration or -1))
            & (frame["status"] == "completed")
        ]
        stability = stability_metrics(labels)
        summaries.append(
            {
                "encoder": encoder_name,
                "encoder_seed": encoder_seed,
                "algorithm": algorithm,
                "K": states,
                "max_duration_patches": max_duration,
                "completed_seeds": int(subset.shape[0]),
                "failed_seeds": int(len(seeds) - subset.shape[0]),
                **stability,
                **{
                    f"{column}_mean": float(subset[column].mean())
                    for column in (
                        "validation_fit_per_patch",
                        "silhouette_validation",
                        "effective_entropy_states",
                        "largest_state_occupancy",
                        "bouts_per_minute",
                        "median_bout_duration_s",
                        "fraction_one_timestep_bouts",
                        "fraction_bouts_at_duration_cap",
                    )
                    if column in subset and not subset.empty
                },
            }
        )
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    manifest = {
        "contract": "social_downstream_benchmark_v1",
        "encoder": encoder_name,
        "encoder_seed": encoder_seed,
        "embedding_dir": str(Path(embedding_dir).resolve()),
        "experiment_config": str(Path(experiment_config).resolve()),
        "output_dir": str(output),
        "preprocessor": str(preprocess_path),
        "preprocessing_fit_trials": train_trial_ids,
        "validation_trials": validation_trial_ids,
        "excluded_trials": sorted(set(by_id) - requested),
        "algorithms": algorithms,
        "state_counts": state_counts,
        "downstream_seeds": seeds,
        "max_durations_patches": max_durations,
        "pca_dim": preprocessor.output_dim,
        "thread_limit": thread_limit,
        "pca_explained_variance_ratio": (
            None
            if preprocessor.explained_variance_ratio_ is None
            else preprocessor.explained_variance_ratio_.tolist()
        ),
        "notes": [
            "Scaling and PCA were fitted on training trials only.",
            "Every trial and invalid-gap-delimited run remains a separate sequence.",
            "A4_C4 is reused development/validation data, not an untouched test.",
            "A5_C5 is excluded from fitting, selection, and performance claims.",
            "Raw likelihoods are comparable across algorithms only within one fixed encoder/PCA observation space.",
            "KMeans one-hot outputs are hard assignments, not posterior probabilities.",
            "State signatures are descriptive measurements and have no assigned behavior names.",
        ],
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--encoder-name", required=True)
    parser.add_argument("--encoder-seed", required=True, type=int)
    parser.add_argument("--train-trial", action="append", required=True)
    parser.add_argument("--validation-trial", action="append", required=True)
    parser.add_argument("--algorithm", action="append", choices=sorted(ALGORITHMS), required=True)
    parser.add_argument("--states", nargs="+", type=int, default=[4, 6, 8, 10, 12])
    parser.add_argument("--seed", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-duration", nargs="+", type=int, default=[25, 50, 100])
    parser.add_argument("--pca-dim", type=int, default=16)
    parser.add_argument("--hmm-max-iter", type=int, default=40)
    parser.add_argument("--hsmm-max-iter", type=int, default=20)
    parser.add_argument("--gmm-max-iter", type=int, default=100)
    parser.add_argument("--covariance-floor", type=float, default=1e-3)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--thread-limit", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_downstream_benchmark(
        embedding_dir=args.embedding_dir,
        output_dir=args.output_dir,
        experiment_config=args.experiment_config,
        train_trial_ids=args.train_trial,
        validation_trial_ids=args.validation_trial,
        algorithms=args.algorithm,
        state_counts=args.states,
        seeds=args.seed,
        max_durations=args.max_duration,
        encoder_name=args.encoder_name,
        encoder_seed=args.encoder_seed,
        pca_dim=args.pca_dim,
        covariance_floor=args.covariance_floor,
        hmm_max_iter=args.hmm_max_iter,
        hsmm_max_iter=args.hsmm_max_iter,
        gmm_max_iter=args.gmm_max_iter,
        tolerance=args.tolerance,
        thread_limit=args.thread_limit,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
