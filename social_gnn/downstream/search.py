"""Trial-level HMM/HSMM hyperparameter search on frozen embeddings."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .embeddings import FrozenEmbeddingTrial, load_embedding_directory
from .evaluation import decode_trial, segmentation_metrics, stability_metrics
from .gaussian_hmm import GaussianHMM
from .gaussian_hsmm import GaussianHSMM
from .preprocessing import EmbeddingPreprocessor


def _trials_by_id(trials: Iterable[FrozenEmbeddingTrial]) -> dict[str, FrozenEmbeddingTrial]:
    return {trial.trial_id: trial for trial in trials}


def _sequences(
    trials: Iterable[FrozenEmbeddingTrial], preprocessor: EmbeddingPreprocessor
) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for trial in trials:
        sequences, _indices = preprocessor.transform_trial_sequences(trial)
        result.extend(sequences)
    return result


def _finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.nanmean(array)) if array.size else float("nan")


def run_temporal_search(
    *,
    embedding_dir: str | Path,
    output_dir: str | Path,
    train_trial_ids: list[str],
    validation_trial_ids: list[str],
    model_types: list[str],
    state_counts: list[int],
    seeds: list[int],
    pca_dims: list[int],
    hmm_max_iter: int = 40,
    hsmm_max_iter: int = 20,
    covariance_floor: float = 1e-3,
    max_duration: int = 50,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    trials = load_embedding_directory(embedding_dir)
    by_id = _trials_by_id(trials)
    requested = set(train_trial_ids) | set(validation_trial_ids)
    missing = sorted(requested - set(by_id))
    if missing:
        raise ValueError(f"unknown trial IDs: {missing}")
    overlap = set(train_trial_ids) & set(validation_trial_ids)
    if overlap:
        raise ValueError(f"train/validation overlap: {sorted(overlap)}")
    if any(value not in {"hmm", "hsmm"} for value in model_types):
        raise ValueError("model_types may contain only 'hmm' and 'hsmm'")
    train_trials = [by_id[value] for value in train_trial_ids]
    validation_trials = [by_id[value] for value in validation_trial_ids]
    input_dim = train_trials[0].graph_embeddings.shape[1]
    records: list[dict[str, Any]] = []
    decoded_by_group: dict[tuple[str, int, int], list[np.ndarray]] = {}

    for requested_dim in pca_dims:
        pca_dim = None if requested_dim == input_dim else requested_dim
        preprocessor = EmbeddingPreprocessor().fit(
            train_trials, pca_dim=pca_dim, seed=seeds[0]
        )
        preprocess_path = preprocessor.save(
            output / f"preprocessing_d{preprocessor.output_dim}.npz"
        )
        train_sequences = _sequences(train_trials, preprocessor)
        validation_sequences = _sequences(validation_trials, preprocessor)
        train_steps = sum(value.shape[0] for value in train_sequences)
        validation_steps = sum(value.shape[0] for value in validation_sequences)
        for model_type in model_types:
            for states in state_counts:
                group_key = (model_type, preprocessor.output_dim, states)
                decoded_by_group[group_key] = []
                for seed in seeds:
                    started = time.perf_counter()
                    if model_type == "hmm":
                        model = GaussianHMM(
                            n_states=states,
                            covariance_floor=covariance_floor,
                            seed=seed,
                        ).fit(
                            train_sequences,
                            max_iter=hmm_max_iter,
                            tolerance=tolerance,
                        )
                    else:
                        model = GaussianHSMM(
                            n_states=states,
                            max_duration=max_duration,
                            covariance_floor=covariance_floor,
                            seed=seed,
                        ).fit(
                            train_sequences,
                            max_iter=hsmm_max_iter,
                            tolerance=tolerance,
                        )
                    train_ll = model.score(train_sequences)
                    validation_ll = model.score(validation_sequences)
                    decoded = [
                        decode_trial(model, preprocessor, trial)
                        for trial in validation_trials
                    ]
                    metrics = segmentation_metrics(decoded, states)
                    joined_state = np.concatenate(
                        [value.predicted_state[value.predicted_state >= 0] for value in decoded]
                    )
                    decoded_by_group[group_key].append(joined_state)
                    parameter_count = model.parameter_count()
                    checkpoint_dir = output / "models" / model_type / f"d{preprocessor.output_dim}_k{states}"
                    checkpoint_path = checkpoint_dir / f"seed_{seed}.npz"
                    metadata = {
                        "embedding_dir": str(Path(embedding_dir).resolve()),
                        "preprocessor": str(preprocess_path),
                        "train_trials": train_trial_ids,
                        "validation_trials": validation_trial_ids,
                        "pca_dim": preprocessor.output_dim,
                        "seed": seed,
                    }
                    model.save(checkpoint_path, metadata=metadata)
                    elapsed = time.perf_counter() - started
                    record = {
                        "model": model_type,
                        "embedding_dim": preprocessor.output_dim,
                        "K": states,
                        "seed": seed,
                        "train_log_likelihood": train_ll,
                        "validation_log_likelihood": validation_ll,
                        "train_ll_per_timestep": train_ll / train_steps,
                        "validation_ll_per_timestep": validation_ll / validation_steps,
                        "parameter_count": parameter_count,
                        "aic_train": 2 * parameter_count - 2 * train_ll,
                        "bic_train": np.log(train_steps) * parameter_count - 2 * train_ll,
                        "iterations": len(model.history_),
                        "converged": bool(model.converged_),
                        "elapsed_s": elapsed,
                        "checkpoint": str(checkpoint_path),
                        "preprocessor": str(preprocess_path),
                        **metrics,
                    }
                    records.append(record)
                    pd.DataFrame(records).to_csv(output / "search_runs.csv", index=False)

    summaries: list[dict[str, Any]] = []
    frame = pd.DataFrame(records)
    for (model_type, dimension, states), group in frame.groupby(
        ["model", "embedding_dim", "K"], sort=True
    ):
        stability = stability_metrics(decoded_by_group[(model_type, dimension, states)])
        best_index = group["validation_ll_per_timestep"].idxmax()
        best = frame.loc[best_index]
        summaries.append(
            {
                "model": model_type,
                "embedding_dim": int(dimension),
                "K": int(states),
                "seeds": int(group.shape[0]),
                "validation_ll_per_timestep_mean": float(
                    group["validation_ll_per_timestep"].mean()
                ),
                "validation_ll_per_timestep_std": float(
                    group["validation_ll_per_timestep"].std(ddof=0)
                ),
                "mean_pairwise_ari": stability["mean_pairwise_ari"],
                "mean_pairwise_nmi": stability["mean_pairwise_nmi"],
                "median_bout_duration_s_mean": float(
                    group["median_bout_duration_s"].mean()
                ),
                "bouts_per_minute_mean": float(group["bouts_per_minute"].mean()),
                "effective_states_mean": float(
                    group["effective_entropy_states"].mean()
                ),
                "states_over_one_percent_mean": float(
                    group["states_over_one_percent"].mean()
                ),
                "mean_state_confidence": float(group["mean_state_confidence"].mean()),
                "converged_fraction": float(group["converged"].mean()),
                "best_seed": int(best["seed"]),
                "best_validation_ll_per_timestep": float(
                    best["validation_ll_per_timestep"]
                ),
                "best_checkpoint": str(best["checkpoint"]),
                "preprocessor": str(best["preprocessor"]),
            }
        )
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(output / "search_summary.csv", index=False)
    manifest = {
        "contract": "social_temporal_search_v1",
        "embedding_dir": str(Path(embedding_dir).resolve()),
        "output_dir": str(output),
        "train_trials": train_trial_ids,
        "validation_trials": validation_trial_ids,
        "excluded_trials": sorted(set(by_id) - requested),
        "model_types": model_types,
        "state_counts": state_counts,
        "seeds": seeds,
        "pca_dims": pca_dims,
        "hmm_max_iter": hmm_max_iter,
        "hsmm_max_iter": hsmm_max_iter,
        "max_duration_timesteps": max_duration,
        "notes": [
            "Every trial and every valid run is passed as an independent sequence.",
            "No transitions are counted across trial boundaries or invalid clock gaps.",
            "A5_C5 is excluded from model selection because it was previously inspected.",
            "Likelihoods across different PCA dimensions are not directly rank-comparable.",
        ],
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-trial", action="append", required=True)
    parser.add_argument("--validation-trial", action="append", required=True)
    parser.add_argument("--model", action="append", choices=("hmm", "hsmm"), required=True)
    parser.add_argument("--states", nargs="+", type=int, default=[4, 6, 8, 10, 12])
    parser.add_argument("--seed", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--pca-dim", nargs="+", type=int, default=[64, 16])
    parser.add_argument("--hmm-max-iter", type=int, default=40)
    parser.add_argument("--hsmm-max-iter", type=int, default=20)
    parser.add_argument("--max-duration", type=int, default=50)
    parser.add_argument("--covariance-floor", type=float, default=1e-3)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_temporal_search(
        embedding_dir=args.embedding_dir,
        output_dir=args.output_dir,
        train_trial_ids=args.train_trial,
        validation_trial_ids=args.validation_trial,
        model_types=args.model,
        state_counts=args.states,
        seeds=args.seed,
        pca_dims=args.pca_dim,
        hmm_max_iter=args.hmm_max_iter,
        hsmm_max_iter=args.hsmm_max_iter,
        covariance_floor=args.covariance_floor,
        max_duration=args.max_duration,
        tolerance=args.tolerance,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
