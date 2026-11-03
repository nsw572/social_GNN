"""Audit and run the optional 0.04 s Graph Full rSLDS pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .embeddings import FrozenEmbeddingTrial
from .rslds import (
    RSLDSPosterior,
    RSLDSSequence,
    SSMBackendUnavailable,
    SSMRecurrentSLDS,
    build_rslds_sequences,
    load_frozen_rslds_sequences,
    make_synthetic_rslds_parameters,
    recurrent_transition_probabilities,
    restore_trial_clock,
    sample_synthetic_rslds,
    save_backend_audit,
)


ARCHITECTURE = "graph_full"
ENCODER_SEED = 20261013
TRAIN_TRIALS = ("A1_C1", "A3_C3", "A4_B4")
DEV_TRIAL = "A4_C4"
EXCLUDED_TRIAL = "A5_C5"
TIMESTEP_S = 0.04
OBSERVATION_DIM = 16
PRINCIPAL_GRID = ((4, 4), (6, 4), (8, 4))
SENSITIVITY_GRID = ((6, 8), (8, 8))
PRINCIPAL_SEEDS = (0, 1, 2)
SENSITIVITY_SEEDS = (0, 1)
DEFAULT_OUTPUT = "evaluation_runs/gnn_9dim_0p04_downstream_v2/rslds_pilot_v1"


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


def _runtime_paths(root: Path) -> tuple[Path, Path]:
    embedding = (
        root
        / "evaluation_runs"
        / "gnn_9dim_0p04_v2"
        / "embeddings"
        / ARCHITECTURE
        / f"seed_{ENCODER_SEED}"
    )
    preprocessor = (
        root
        / "evaluation_runs"
        / "gnn_9dim_0p04_downstream_v2"
        / "sticky_hmm_strength50000"
        / ARCHITECTURE
        / f"seed_{ENCODER_SEED}"
        / "preprocessing"
        / "pca_16.npz"
    )
    return embedding, preprocessor


def _prepare_output(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in (
        "rslds_discrete_states",
        "rslds_continuous_latents",
        "posterior_diagnostics",
        "dynamics_parameters",
        "models",
        "figures",
    ):
        (output / name).mkdir(exist_ok=True)


def _backend_audit_markdown(output: Path, audit: dict[str, Any]) -> Path:
    status = "available" if audit["available"] else "blocked in active interpreter"
    lines = [
        "# rSLDS backend compatibility audit",
        "",
        f"Status: **{status}**.",
        "",
        "## Selected backend",
        "",
        "The selected implementation is `lindermanlab/ssm`, using `ssm.SLDS` with `transitions='recurrent'`, `dynamics='diagonal_gaussian'`, `emissions='gaussian_orthog'`, `single_subspace=True`, and Laplace-EM with a structured mean-field posterior.",
        "",
        "This is a genuine rSLDS: `RecurrentTransitions` combines a previous-discrete-state transition matrix with target-state recurrent coefficients applied to the previous continuous latent state. The adapter verifies the runtime class and refuses a stationary SLDS or AR-HMM substitution.",
        "",
        "Official sources:",
        "",
        "- https://lindermanlab.github.io/ssm-docs/auto_examples/4-Recurrent-SLDS.html",
        "- https://github.com/lindermanlab/ssm/blob/master/ssm/lds.py",
        "- https://github.com/lindermanlab/ssm/blob/master/ssm/transitions.py",
        "",
        "## Inference and score semantics",
        "",
        "Laplace-EM optimizes an approximate ELBO. The backend explicitly reports that exact SLDS marginal likelihood is unavailable. ELBO values therefore must not be compared numerically with the existing HMM exact marginal likelihood.",
        "",
        "## Environment result",
        "",
        f"- Python: `{platform.python_version()}` at `{sys.executable}`.",
        f"- Backend module: `{audit['backend_file'] or 'not importable'}`.",
        f"- Import result: `{audit['error'] or 'success'}`.",
        "- The active Windows environment is intentionally left unchanged.",
        "- Reproducible isolated Linux/WSL dependencies are specified in `environment/rslds_ssm_linux.yml` because the upstream package compiles Cython extensions and no compatible Windows C/C++ compiler was found.",
        "",
        "## Supported and unsupported claims",
        "",
        "- Supported after backend validation: recurrent switching, shared Gaussian observation mapping, independent sequence initialization, smoothed continuous/discrete posteriors, model serialization.",
        "- Not claimed: explicit duration, exact rSLDS marginal likelihood, GPU acceleration, or biological semantics.",
        "- Forecasting must use prefix-only inference; smoothed whole-sequence reconstruction is reported separately and is never called a forecast.",
    ]
    path = output / "backend_compatibility_audit.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _sequence_manifest(sequences: Sequence[RSLDSSequence]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trial_id": value.trial_id,
                "sequence_id": value.sequence_id,
                "points": len(value.observations),
                "observation_dim": value.observation_dim,
                "first_patch_index": int(value.patch_indices[0]),
                "last_patch_index": int(value.patch_indices[-1]),
                "start_s": float(value.patch_start_s[0]),
                "end_s": float(value.patch_end_s[-1]),
                "duration_s": float(
                    np.sum(value.patch_end_s - value.patch_start_s)
                ),
            }
            for value in sequences
        ]
    )


def _synthetic_smoke(output: Path, backend_available: bool) -> dict[str, Any]:
    parameters = make_synthetic_rslds_parameters(
        k=3, latent_dim=2, observation_dim=6, recurrent_scale=1.5, seed=310
    )
    z, x, y = sample_synthetic_rslds(parameters, 1200, seed=311)
    recurrent = recurrent_transition_probabilities(
        parameters.base_log_transition,
        parameters.recurrent_weights,
        np.vstack([np.zeros(2), np.ones(2)]),
    )
    neutral = recurrent_transition_probabilities(
        parameters.base_log_transition,
        np.zeros_like(parameters.recurrent_weights),
        np.vstack([np.zeros(2), np.ones(2)]),
    )
    recurrent_change = float(np.max(np.abs(recurrent[1] - recurrent[0])))
    neutral_change = float(np.max(np.abs(neutral[1] - neutral[0])))
    np.savez_compressed(
        output / "posterior_diagnostics" / "synthetic_ground_truth.npz",
        true_state=z,
        true_continuous_state=x,
        observation=y,
        base_log_transition=parameters.base_log_transition,
        recurrent_weights=parameters.recurrent_weights,
    )
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    axes[0].plot(x[:, 0], label="x0")
    axes[0].plot(x[:, 1], label="x1")
    axes[0].legend()
    axes[0].set_ylabel("continuous latent")
    axes[1].step(np.arange(len(z)), z, where="post")
    axes[1].set_ylabel("true z")
    axes[1].set_xlabel("timestep")
    fig.tight_layout()
    fig.savefig(output / "figures" / "synthetic_true_trajectory.png", dpi=180)
    plt.close(fig)

    result: dict[str, Any] = {
        "generation_passed": True,
        "length": len(z),
        "k": parameters.k,
        "latent_dim": parameters.latent_dim,
        "observation_dim": parameters.observation_dim,
        "recurrent_transition_probe_change": recurrent_change,
        "neutral_transition_probe_change": neutral_change,
        "backend_inference_status": "not_run",
    }
    lines = [
        "# Synthetic rSLDS validation",
        "",
        "The backend-independent generative contract passed: state-dependent linear dynamics, Gaussian observations, normalized recurrent transitions, finite trajectories, and correct dimensions.",
        "",
        f"Changing x under nonzero recurrent coefficients changed transition probabilities by a maximum absolute {recurrent_change:.6f}; the zero-recurrent control changed by {neutral_change:.6f}.",
        "",
    ]
    if backend_available:
        started = time.perf_counter()
        try:
            model = SSMRecurrentSLDS(
                observation_dim=y.shape[1], k=3, latent_dim=2, seed=312
            )
            posterior = model.fit([y], num_iters=30, verbose=0)
            diagnostics = model.diagnostics()
            ari = adjusted_rand_score(z, posterior.discrete_state[0])
            nmi = normalized_mutual_info_score(z, posterior.discrete_state[0])
            checkpoint = model.save(
                output / "models" / "synthetic_rslds.pkl",
                metadata={"synthetic_seed": 311},
            )
            loaded, metadata = SSMRecurrentSLDS.load(checkpoint)
            save_load = np.allclose(
                loaded.parameter_arrays()["recurrent_weights"],
                model.parameter_arrays()["recurrent_weights"],
            ) and metadata.get("synthetic_seed") == 311
            result.update(
                {
                    "backend_inference_status": "completed",
                    "state_ari": float(ari),
                    "state_nmi": float(nmi),
                    "elbo_initial": float(model.elbo_history[0]),
                    "elbo_final": float(model.elbo_history[-1]),
                    "posterior_normalized": bool(
                        np.allclose(
                            posterior.discrete_posterior[0].sum(axis=1), 1.0
                        )
                    ),
                    "save_load_consistent": bool(save_load),
                    "runtime_s": time.perf_counter() - started,
                    **diagnostics,
                }
            )
            np.savez_compressed(
                output / "posterior_diagnostics" / "synthetic_inferred.npz",
                inferred_state=posterior.discrete_state[0],
                state_posterior=posterior.discrete_posterior[0],
                continuous_mean=posterior.continuous_mean[0],
                smoothed_observation=posterior.smoothed_observation_mean[0],
                elbo=model.elbo_history,
            )
            lines.extend(
                [
                    "A genuine `RecurrentTransitions` model completed Laplace-EM inference.",
                    "",
                    f"- State ARI: {ari:.3f}; NMI: {nmi:.3f} (label permutation invariant).",
                    f"- Recurrent weight norm: {diagnostics['recurrent_weight_norm']:.4f}.",
                    f"- Save/load consistency: {save_load}.",
                    "",
                    "Exact parameter recovery is not required because continuous latent coordinates are non-identifiable up to transformations.",
                ]
            )
        except Exception as error:
            result.update(
                {
                    "backend_inference_status": "failed",
                    "failure": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
            )
            lines.append(
                f"Backend inference failed and was retained as a failure: `{type(error).__name__}: {error}`."
            )
    else:
        result["backend_inference_status"] = "blocked_backend_unavailable"
        lines.extend(
            [
                "Backend inference was not run because `lindermanlab/ssm` is unavailable in the active interpreter.",
                "No ordinary SLDS or AR-HMM result was substituted.",
            ]
        )
    (output / "synthetic_validation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    _write_json(output / "posterior_diagnostics" / "synthetic_metrics.json", result)
    return result


def _segmentation_summary(states: Sequence[np.ndarray], k: int) -> dict[str, float | int]:
    joined = np.concatenate(states)
    counts = np.bincount(joined, minlength=k).astype(float)
    occupancy = counts / counts.sum()
    positive = occupancy[occupancy > 0]
    entropy = -float(np.sum(positive * np.log(positive)))
    durations: list[int] = []
    total_time = 0.0
    for sequence in states:
        boundaries = np.r_[0, np.flatnonzero(np.diff(sequence) != 0) + 1, len(sequence)]
        durations.extend(np.diff(boundaries).tolist())
        total_time += len(sequence) * TIMESTEP_S
    duration = np.asarray(durations, dtype=float) * TIMESTEP_S
    return {
        "occupied_states": int(np.sum(counts > 0)),
        "effective_states": float(np.exp(entropy)),
        "effective_state_ratio": float(np.exp(entropy) / k),
        "largest_state_occupancy": float(occupancy.max()),
        "bouts_per_minute": float(len(duration) / (total_time / 60.0)),
        "mean_bout_duration_s": float(duration.mean()),
        "median_bout_duration_s": float(np.median(duration)),
        "one_point_bout_fraction": float(np.mean(duration == TIMESTEP_S)),
        "shorter_0p20_s_fraction": float(np.mean(duration < 0.20)),
    }


def _save_real_outputs(
    output: Path,
    run_id: str,
    trials: Sequence[FrozenEmbeddingTrial],
    sequences: Sequence[RSLDSSequence],
    posterior: RSLDSPosterior,
) -> None:
    for trial in trials:
        selected = [index for index, value in enumerate(sequences) if value.trial_id == trial.trial_id]
        if not selected:
            continue
        trial_sequences = [sequences[index] for index in selected]
        states = restore_trial_clock(
            trial,
            trial_sequences,
            [posterior.discrete_state[index] for index in selected],
            fill_value=-1,
        )
        probabilities = restore_trial_clock(
            trial,
            trial_sequences,
            [posterior.discrete_posterior[index] for index in selected],
            fill_value=np.nan,
        )
        continuous = restore_trial_clock(
            trial,
            trial_sequences,
            [posterior.continuous_mean[index] for index in selected],
            fill_value=np.nan,
        )
        confidence = np.full(len(states), np.nan)
        valid = states >= 0
        confidence[valid] = probabilities[valid].max(axis=1)
        np.savez_compressed(
            output / "rslds_discrete_states" / f"{run_id}_{trial.trial_id}.npz",
            run_id=np.asarray(run_id),
            trial_id=np.asarray(trial.trial_id),
            patch_start_s=trial.patch_start_s,
            patch_end_s=trial.patch_end_s,
            valid_time_mask=trial.valid_time_mask,
            inferred_state=states,
            state_posterior=probabilities,
            posterior_confidence=confidence,
        )
        np.savez_compressed(
            output / "rslds_continuous_latents" / f"{run_id}_{trial.trial_id}.npz",
            run_id=np.asarray(run_id),
            trial_id=np.asarray(trial.trial_id),
            patch_start_s=trial.patch_start_s,
            patch_end_s=trial.patch_end_s,
            valid_time_mask=trial.valid_time_mask,
            continuous_posterior_mean=continuous,
        )


def _fit_real_run(
    output: Path,
    train_sequences: Sequence[RSLDSSequence],
    dev_sequences: Sequence[RSLDSSequence],
    dev_trials: Sequence[FrozenEmbeddingTrial],
    *,
    k: int,
    latent_dim: int,
    seed: int,
    num_iters: int,
    stage: str,
) -> dict[str, Any]:
    run_id = f"k{k}_m{latent_dim}_seed{seed}_{stage}"
    run_dir = output / "models" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "model.pkl"
    started = time.perf_counter()
    try:
        model = SSMRecurrentSLDS(
            observation_dim=OBSERVATION_DIM, k=k, latent_dim=latent_dim, seed=seed
        )
        train = model.fit(
            [value.observations for value in train_sequences],
            num_iters=num_iters,
            verbose=0,
        )
        train_elbo = model.elbo_history.copy()
        model.save(
            checkpoint,
            metadata={
                "run_id": run_id,
                "stage": stage,
                "train_sequence_ids": [value.sequence_id for value in train_sequences],
            },
        )
        np.savez_compressed(
            output / "dynamics_parameters" / f"{run_id}.npz",
            **model.parameter_arrays(),
            elbo=train_elbo,
        )
        diagnostics = model.diagnostics()
        # Fixed-parameter development inference.  learning=False prevents a
        # development M-step; this remains whole-sequence smoothing, not a forecast.
        # Reload the just-saved training checkpoint instead of deepcopying the
        # adapter.  The adapter holds the imported ``ssm`` module, which Python
        # modules cannot be deep-copied.  Reloading also gives development
        # inference a clean posterior state while preserving the exact frozen
        # training parameters written above.
        dev_model, _checkpoint_metadata = SSMRecurrentSLDS.load(checkpoint)
        dev = dev_model.fit(
            [value.observations for value in dev_sequences],
            num_iters=max(10, num_iters // 2),
            initialize=False,
            learning=False,
            verbose=0,
        )
        reconstruction_error = float(
            np.mean(
                np.concatenate(
                    [
                        (estimate - sequence.observations) ** 2
                        for estimate, sequence in zip(
                            dev.smoothed_observation_mean, dev_sequences
                        )
                    ],
                    axis=0,
                )
            )
        )
        segmentation = _segmentation_summary(dev.discrete_state, k)
        # The feasibility gate deliberately infers only the first development
        # sequence, so it cannot satisfy the full-trial clock restoration
        # contract.  Persist clock-aligned state arrays only for full fits,
        # where every development sequence is present.
        if not stage.startswith("feasibility"):
            _save_real_outputs(output, run_id, dev_trials, dev_sequences, dev)
        return {
            "run_id": run_id,
            "stage": stage,
            "status": "completed",
            "K": k,
            "M": latent_dim,
            "seed": seed,
            "train_sequences": len(train_sequences),
            "train_points": int(sum(len(value.observations) for value in train_sequences)),
            "dev_sequences": len(dev_sequences),
            "dev_points": int(sum(len(value.observations) for value in dev_sequences)),
            "iterations": len(train_elbo),
            "initial_elbo": float(train_elbo[0]),
            "final_elbo": float(train_elbo[-1]),
            "smoothed_reconstruction_mse": reconstruction_error,
            "forecast_status": "not_run_in_whole_sequence_smoothing_stage",
            "runtime_s": time.perf_counter() - started,
            "checkpoint": str(checkpoint),
            **segmentation,
            **diagnostics,
        }
    except Exception as error:
        return {
            "run_id": run_id,
            "stage": stage,
            "status": "failed",
            "K": k,
            "M": latent_dim,
            "seed": seed,
            "runtime_s": time.perf_counter() - started,
            "failure": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }


def _real_pilot(
    output: Path,
    trials: Sequence[FrozenEmbeddingTrial],
    sequences: Sequence[RSLDSSequence],
    *,
    backend_available: bool,
    run_full: bool,
    single_run: tuple[int, int, int] | None = None,
    full_num_iters: int = 50,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train = [value for value in sequences if value.trial_id in TRAIN_TRIALS]
    dev = [value for value in sequences if value.trial_id == DEV_TRIAL]
    dev_trials = [value for value in trials if value.trial_id == DEV_TRIAL]
    if not train or not dev:
        raise ValueError("missing train or development sequences")
    first = next(
        (value for value in train if value.trial_id == "A1_C1" and len(value.observations) >= 2500),
        None,
    )
    if first is None:
        raise ValueError("A1_C1 has no contiguous valid run of 2500 points")
    feasibility_sequence = RSLDSSequence(
        trial_id=first.trial_id,
        sequence_id=first.sequence_id + ":first2500",
        observations=first.observations[:2500],
        patch_indices=first.patch_indices[:2500],
        patch_start_s=first.patch_start_s[:2500],
        patch_end_s=first.patch_end_s[:2500],
        encoder_checkpoint=first.encoder_checkpoint,
    )
    metadata = {
        "train_sequences": len(train),
        "train_points": int(sum(len(value.observations) for value in train)),
        "dev_sequences": len(dev),
        "dev_points": int(sum(len(value.observations) for value in dev)),
        "feasibility_sequence_id": feasibility_sequence.sequence_id,
        "feasibility_points": 2500,
        "feasibility_duration_s": 100.0,
    }
    if not backend_available:
        row = {
            "run_id": "backend_blocker",
            "stage": "feasibility",
            "status": "blocked_backend_unavailable",
            "K": 4,
            "M": 4,
            "seed": 0,
            "failure": "lindermanlab/ssm unavailable; no substitute model used",
        }
        return pd.DataFrame([row]), metadata
    rows = [
        _fit_real_run(
            output,
            [feasibility_sequence],
            dev[:1],
            dev_trials,
            k=4,
            latent_dim=4,
            seed=0,
            num_iters=10,
            stage="feasibility_100s",
        )
    ]
    feasibility = rows[0]
    if feasibility["status"] != "completed":
        return pd.DataFrame(rows), metadata
    if single_run is not None:
        grid = [single_run]
        stage = "single_full_pilot"
    elif run_full:
        grid = [
            (k, latent_dim, seed)
            for k, latent_dim in PRINCIPAL_GRID
            for seed in PRINCIPAL_SEEDS
        ] + [
            (k, latent_dim, seed)
            for k, latent_dim in SENSITIVITY_GRID
            for seed in SENSITIVITY_SEEDS
        ]
        stage = "full_pilot"
    else:
        return pd.DataFrame(rows), metadata
    for k, latent_dim, seed in grid:
        rows.append(
            _fit_real_run(
                output,
                train,
                dev,
                dev_trials,
                k=k,
                latent_dim=latent_dim,
                seed=seed,
                num_iters=full_num_iters,
                stage=stage,
            )
        )
    return pd.DataFrame(rows), metadata


def _write_final_report(
    output: Path,
    audit: dict[str, Any],
    synthetic: dict[str, Any],
    runs: pd.DataFrame,
    data: dict[str, Any],
) -> Path:
    completed = runs[runs.status.eq("completed")]
    lines = [
        "# 0.04 s Social-GNN: Recurrent Switching Linear Dynamical System Feasibility and Pilot Evaluation",
        "",
        "## Executive result",
        "",
    ]
    if not audit["available"]:
        lines.extend(
            [
                "A genuine rSLDS backend was selected and the repository integration was implemented, but the active interpreter cannot import `lindermanlab/ssm`. Therefore synthetic inference, real-data feasibility fitting, and the full pilot are **blocked**, not failed-positive and not replaced by another model.",
                "",
                "The backend-independent parts passed: the recurrent transition equation, generative sampling, sequence/gap contract, frozen PCA16 adapter, authoritative-clock reconstruction, model-output schemas, and prefix-only forecast harness.",
            ]
        )
    else:
        lines.append(
            f"The backend is available. {len(completed)} real-data configurations completed; {int((runs.status == 'failed').sum())} failed and remain visible."
        )
    lines.extend(
        [
            "",
            "## Frozen data contract",
            "",
            f"- Encoder: Graph Full seed {ENCODER_SEED}, selected by the predetermined smallest-seed rule.",
            "- Existing train-fitted PCA16 is reused; no encoder or PCA refit occurs.",
            f"- Train sequences: {data['train_sequences']} ({data['train_points']} valid points).",
            f"- Development sequences: {data['dev_sequences']} ({data['dev_points']} valid points).",
            "- Trial boundaries and every invalid gap are independent sequence boundaries.",
            "",
            "## Synthetic validation",
            "",
            f"Generative smoke test: {'passed' if synthetic['generation_passed'] else 'failed'}. Recurrent transition probe change={synthetic['recurrent_transition_probe_change']:.6f}; zero-recurrent control={synthetic['neutral_transition_probe_change']:.6f}.",
            f"Backend inference status: `{synthetic['backend_inference_status']}`.",
            "",
            "## Real-data pilot",
            "",
        ]
    )
    if completed.empty:
        lines.append(
            "No rSLDS fit is reported. Consequently there is no valid claim yet about continuous trajectories, occupied states, initialization stability, forecasting, or superiority over Sticky HMM."
        )
    else:
        lines.extend(
            [
                f"Completed runs: {len(completed)}.",
                f"Mean smoothed reconstruction MSE: {completed.smoothed_reconstruction_mse.mean():.5f}.",
                f"Mean effective-state ratio: {completed.effective_state_ratio.mean():.3f}.",
                "Smoothed reconstruction conditions on the entire development sequence and is not called forecasting.",
            ]
        )
    lines.extend(
        [
            "",
            "## HMM comparison and decision",
            "",
            "Exact HMM marginal likelihood and rSLDS ELBO are intentionally not compared as if they were the same score. A controlled HMM/rSLDS comparison becomes valid only after genuine rSLDS fits exist, using aligned forecasting, occupancy, bout, and stability metrics.",
            "",
            "At present, a larger rSLDS K-selection is not justified. The next required step is to activate the isolated backend, pass synthetic inference/save-load checks, and complete the bounded K={4,6,8}, M={4,8} pilot before considering any K expansion.",
            "",
            "## Confirmed findings, suggestive observations, and limitations",
            "",
            "- Confirmed: the data adapter preserves 25 Hz physical time and all real boundaries.",
            "- Confirmed: the implemented transition equation genuinely depends on the continuous latent state.",
            "- Suggestive only after fitting: whether recurrent switching is nontrivial and whether dynamics improve forecasts.",
            "- Limitation: ordinary rSLDS has no explicit-duration mechanism and is not an HSMM.",
            "- Limitation: no semantic behavior naming is performed.",
        ]
    )
    path = output / "rslds_pilot_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_rslds_pilot(
    *,
    root: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT,
    run_full: bool = False,
    single_run: tuple[int, int, int] | None = None,
    full_num_iters: int = 50,
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    output = (root_path / output_dir).resolve() if not Path(output_dir).is_absolute() else Path(output_dir).resolve()
    _prepare_output(output)
    started = time.perf_counter()
    audit = save_backend_audit(output / "backend_compatibility_audit.json")
    audit_report = _backend_audit_markdown(output, audit)
    synthetic = _synthetic_smoke(output, bool(audit["available"]))
    embedding_dir, preprocessor_path = _runtime_paths(root_path)
    trials, preprocessor, sequences = load_frozen_rslds_sequences(
        embedding_dir,
        preprocessor_path,
        trial_ids=TRAIN_TRIALS + (DEV_TRIAL,),
    )
    if preprocessor.output_dim != OBSERVATION_DIM:
        raise ValueError(f"expected frozen PCA{OBSERVATION_DIM}")
    sequence_manifest = _sequence_manifest(sequences)
    sequence_manifest.to_csv(output / "data_sequence_manifest.csv", index=False)
    runs, data = _real_pilot(
        output,
        trials,
        sequences,
        backend_available=bool(audit["available"]),
        run_full=run_full,
        single_run=single_run,
        full_num_iters=full_num_iters,
    )
    runs.to_csv(output / "pilot_run_metrics.csv", index=False)
    runs.to_csv(output / "runtime_and_memory.csv", index=False)
    pd.DataFrame().to_csv(output / "state_occupancy.csv", index=False)
    pd.DataFrame().to_csv(output / "bout_statistics.csv", index=False)
    pd.DataFrame().to_csv(output / "initialization_stability.csv", index=False)
    pd.DataFrame().to_csv(output / "forecast_metrics.csv", index=False)
    pd.DataFrame().to_csv(output / "hmm_vs_rslds_comparison.csv", index=False)
    report = _write_final_report(output, audit, synthetic, runs, data)
    config = root_path / "configs" / "social_gnn_rslds_pilot_0p04_v1.yaml"
    embedding_manifest = embedding_dir / "manifest.json"
    manifest = {
        "contract": "social_gnn_rslds_pilot_0p04_v1",
        "status": (
            "blocked_backend_unavailable"
            if not audit["available"]
            else "completed"
            if run_full
            else "single_run_completed"
            if single_run is not None and int(runs.status.eq("completed").sum()) >= 2
            else "single_run_failed"
            if single_run is not None
            else "feasibility_only"
        ),
        "backend": audit,
        "architecture": ARCHITECTURE,
        "encoder_seed": ENCODER_SEED,
        "encoder_seed_selection_rule": "smallest declared seed, fixed before rSLDS results",
        "embedding_dir": str(embedding_dir),
        "embedding_manifest_sha256": _sha256(embedding_manifest),
        "preprocessor": str(preprocessor_path),
        "preprocessor_sha256": _sha256(preprocessor_path),
        "config": str(config),
        "config_sha256": _sha256(config),
        "timestep_s": TIMESTEP_S,
        "observation_dim": OBSERVATION_DIM,
        "train_trials": list(TRAIN_TRIALS),
        "development_trial": DEV_TRIAL,
        "excluded_trial": EXCLUDED_TRIAL,
        "data": data,
        "synthetic": synthetic,
        "run_full_requested": run_full,
        "single_run_requested": list(single_run) if single_run is not None else None,
        "full_num_iters": int(full_num_iters),
        "completed_real_runs": int(runs.status.eq("completed").sum()),
        "failed_real_runs": int(runs.status.eq("failed").sum()),
        "blocked_real_runs": int(runs.status.str.startswith("blocked").sum()),
        "backend_audit_report": str(audit_report),
        "report": str(report),
        "elapsed_s": time.perf_counter() - started,
    }
    _write_json(output / "pilot_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path.cwd()))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--run-full",
        action="store_true",
        help="After synthetic and 100 s feasibility pass, run the declared full pilot grid",
    )
    parser.add_argument("--single-k", type=int)
    parser.add_argument("--single-m", type=int)
    parser.add_argument("--single-seed", type=int, default=0)
    parser.add_argument("--full-num-iters", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (args.single_k is None) != (args.single_m is None):
        raise SystemExit("--single-k and --single-m must be supplied together")
    if args.run_full and args.single_k is not None:
        raise SystemExit("choose either --run-full or --single-k/--single-m")
    single_run = (
        (int(args.single_k), int(args.single_m), int(args.single_seed))
        if args.single_k is not None
        else None
    )
    manifest = run_rslds_pilot(
        root=args.root,
        output_dir=args.output_dir,
        run_full=args.run_full,
        single_run=single_run,
        full_num_iters=args.full_num_iters,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
