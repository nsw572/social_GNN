"""Data contract and optional genuine-rSLDS backend for frozen Social-GNN data.

The backend is intentionally optional.  Importing this module never imports
``ssm``.  When available, :class:`SSMRecurrentSLDS` constructs an actual
``ssm.SLDS`` with ``transitions='recurrent'``; it never falls back to a
stationary SLDS or AR-HMM under the rSLDS name.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .embeddings import FrozenEmbeddingTrial, load_embedding_directory
from .preprocessing import EmbeddingPreprocessor


@dataclass(frozen=True)
class RSLDSSequence:
    """One independent contiguous PCA observation sequence."""

    trial_id: str
    sequence_id: str
    observations: np.ndarray
    patch_indices: np.ndarray
    patch_start_s: np.ndarray
    patch_end_s: np.ndarray
    encoder_checkpoint: str

    def __post_init__(self) -> None:
        y = np.asarray(self.observations, dtype=np.float64)
        index = np.asarray(self.patch_indices, dtype=np.int64)
        start = np.asarray(self.patch_start_s, dtype=np.float64)
        end = np.asarray(self.patch_end_s, dtype=np.float64)
        if y.ndim != 2 or y.shape[0] < 2:
            raise ValueError(f"{self.sequence_id}: observations must be [L,D], L>=2")
        length = y.shape[0]
        if index.shape != (length,) or start.shape != (length,) or end.shape != (length,):
            raise ValueError(f"{self.sequence_id}: clock/index arrays must be [L]")
        if not np.isfinite(y).all() or not np.isfinite(start).all() or not np.isfinite(end).all():
            raise ValueError(f"{self.sequence_id}: nonfinite observations or clock")
        if np.any(np.diff(index) != 1):
            raise ValueError(f"{self.sequence_id}: sequence crosses an invalid gap")
        if np.any(np.diff(start) <= 0) or np.any(end <= start):
            raise ValueError(f"{self.sequence_id}: invalid physical clock")
        for value in (y, index, start, end):
            value.setflags(write=False)
        object.__setattr__(self, "observations", y)
        object.__setattr__(self, "patch_indices", index)
        object.__setattr__(self, "patch_start_s", start)
        object.__setattr__(self, "patch_end_s", end)

    @property
    def patch_center_s(self) -> np.ndarray:
        return 0.5 * (self.patch_start_s + self.patch_end_s)

    @property
    def observation_dim(self) -> int:
        return int(self.observations.shape[1])


def build_rslds_sequences(
    trials: Iterable[FrozenEmbeddingTrial],
    preprocessor: EmbeddingPreprocessor,
    *,
    trial_ids: Sequence[str] | None = None,
) -> list[RSLDSSequence]:
    """Transform trials with the frozen PCA and preserve every real boundary."""

    allowed = None if trial_ids is None else set(trial_ids)
    sequences: list[RSLDSSequence] = []
    seen_trials: set[str] = set()
    for trial in trials:
        if allowed is not None and trial.trial_id not in allowed:
            continue
        seen_trials.add(trial.trial_id)
        arrays, indices = preprocessor.transform_trial_sequences(trial)
        for number, (values, patch_index) in enumerate(zip(arrays, indices)):
            if values.shape[0] < 2:
                continue
            sequence_id = (
                f"{trial.trial_id}:run{number}:"
                f"patch{int(patch_index[0])}-{int(patch_index[-1])}"
            )
            sequences.append(
                RSLDSSequence(
                    trial_id=trial.trial_id,
                    sequence_id=sequence_id,
                    observations=values,
                    patch_indices=patch_index,
                    patch_start_s=trial.patch_start_s[patch_index],
                    patch_end_s=trial.patch_end_s[patch_index],
                    encoder_checkpoint=trial.encoder_checkpoint,
                )
            )
    if allowed is not None and seen_trials != allowed:
        raise ValueError(f"missing requested trials {sorted(allowed - seen_trials)}")
    if not sequences:
        raise ValueError("no contiguous valid observation sequences")
    dimensions = {value.observation_dim for value in sequences}
    if len(dimensions) != 1:
        raise ValueError(f"inconsistent observation dimensions {sorted(dimensions)}")
    return sequences


def load_frozen_rslds_sequences(
    embedding_dir: str | Path,
    preprocessor_path: str | Path,
    *,
    trial_ids: Sequence[str],
) -> tuple[list[FrozenEmbeddingTrial], EmbeddingPreprocessor, list[RSLDSSequence]]:
    trials = load_embedding_directory(embedding_dir)
    preprocessor = EmbeddingPreprocessor.load(preprocessor_path)
    sequences = build_rslds_sequences(trials, preprocessor, trial_ids=trial_ids)
    return trials, preprocessor, sequences


def restore_trial_clock(
    trial: FrozenEmbeddingTrial,
    sequences: Sequence[RSLDSSequence],
    values: Sequence[np.ndarray],
    *,
    fill_value: float | int,
) -> np.ndarray:
    """Restore run-local outputs to the authoritative clock without interpolation."""

    if len(sequences) != len(values):
        raise ValueError("sequences and values must have equal length")
    selected = [
        (sequence, np.asarray(value))
        for sequence, value in zip(sequences, values)
        if sequence.trial_id == trial.trial_id
    ]
    if not selected:
        raise ValueError(f"{trial.trial_id}: no sequence output to restore")
    trailing = selected[0][1].shape[1:]
    output = np.full((trial.graph_embeddings.shape[0],) + trailing, fill_value)
    occupied = np.zeros(trial.graph_embeddings.shape[0], dtype=bool)
    for sequence, value in selected:
        if value.shape[0] != sequence.observations.shape[0] or value.shape[1:] != trailing:
            raise ValueError(f"{sequence.sequence_id}: output shape mismatch")
        if occupied[sequence.patch_indices].any():
            raise ValueError(f"{sequence.sequence_id}: overlapping restored patch indices")
        output[sequence.patch_indices] = value
        occupied[sequence.patch_indices] = True
    expected = trial.valid_time_mask.copy()
    # One-point runs are deliberately not modeled and remain unassigned.
    for run in trial.contiguous_valid_runs():
        if len(run) < 2:
            expected[run] = False
    if not np.array_equal(occupied, expected):
        raise ValueError(f"{trial.trial_id}: restored output does not match valid runs")
    return output


def recurrent_transition_probabilities(
    base_log_transition: np.ndarray,
    recurrent_weights: np.ndarray,
    previous_continuous_state: np.ndarray,
) -> np.ndarray:
    """Evaluate P(z_t=j | z_t-1=i, x_t-1) for the SSM parameterization."""

    base = np.asarray(base_log_transition, dtype=np.float64)
    weights = np.asarray(recurrent_weights, dtype=np.float64)
    state = np.asarray(previous_continuous_state, dtype=np.float64)
    if base.ndim != 2 or base.shape[0] != base.shape[1]:
        raise ValueError("base_log_transition must be [K,K]")
    k = base.shape[0]
    if weights.ndim != 2 or weights.shape[0] != k:
        raise ValueError("recurrent_weights must be [K,M]")
    if state.shape[-1:] != (weights.shape[1],):
        raise ValueError("continuous state dimension mismatch")
    flat = state.reshape(-1, weights.shape[1])
    recurrent = flat @ weights.T
    logits = base[None, :, :] + recurrent[:, None, :]
    logits -= logits.max(axis=2, keepdims=True)
    probability = np.exp(logits)
    probability /= probability.sum(axis=2, keepdims=True)
    return probability.reshape(state.shape[:-1] + (k, k))


@dataclass(frozen=True)
class SyntheticRSLDSParameters:
    initial_state_probability: np.ndarray
    initial_continuous_mean: np.ndarray
    initial_continuous_covariance: np.ndarray
    base_log_transition: np.ndarray
    recurrent_weights: np.ndarray
    dynamics_matrices: np.ndarray
    dynamics_offsets: np.ndarray
    process_variances: np.ndarray
    observation_matrix: np.ndarray
    observation_offset: np.ndarray
    observation_variances: np.ndarray

    def __post_init__(self) -> None:
        pi = np.asarray(self.initial_state_probability, dtype=float)
        base = np.asarray(self.base_log_transition, dtype=float)
        recurrent = np.asarray(self.recurrent_weights, dtype=float)
        matrices = np.asarray(self.dynamics_matrices, dtype=float)
        offsets = np.asarray(self.dynamics_offsets, dtype=float)
        q = np.asarray(self.process_variances, dtype=float)
        c = np.asarray(self.observation_matrix, dtype=float)
        d = np.asarray(self.observation_offset, dtype=float)
        r = np.asarray(self.observation_variances, dtype=float)
        k, m = recurrent.shape
        n = c.shape[0]
        if pi.shape != (k,) or base.shape != (k, k):
            raise ValueError("synthetic discrete-state dimensions mismatch")
        if matrices.shape != (k, m, m) or offsets.shape != (k, m) or q.shape != (k, m):
            raise ValueError("synthetic dynamics dimensions mismatch")
        if c.shape[1] != m or d.shape != (n,) or r.shape != (n,):
            raise ValueError("synthetic observation dimensions mismatch")
        if self.initial_continuous_mean.shape != (m,) or self.initial_continuous_covariance.shape != (m, m):
            raise ValueError("synthetic initial continuous dimensions mismatch")
        if not np.isclose(pi.sum(), 1.0) or np.any(pi < 0):
            raise ValueError("initial state probability must be a simplex")
        if np.any(q <= 0) or np.any(r <= 0):
            raise ValueError("all covariance diagonals must be positive")

    @property
    def k(self) -> int:
        return int(self.recurrent_weights.shape[0])

    @property
    def latent_dim(self) -> int:
        return int(self.recurrent_weights.shape[1])

    @property
    def observation_dim(self) -> int:
        return int(self.observation_matrix.shape[0])


def make_synthetic_rslds_parameters(
    *,
    k: int = 3,
    latent_dim: int = 2,
    observation_dim: int = 4,
    recurrent_scale: float = 1.5,
    seed: int = 0,
) -> SyntheticRSLDSParameters:
    if k < 2 or latent_dim < 1 or observation_dim <= latent_dim:
        raise ValueError("require K>=2 and observation_dim>latent_dim>=1")
    rng = np.random.default_rng(seed)
    base_probability = np.full((k, k), 0.08 / (k - 1))
    np.fill_diagonal(base_probability, 0.92)
    matrices = np.empty((k, latent_dim, latent_dim))
    for state in range(k):
        raw = rng.normal(size=(latent_dim, latent_dim))
        radius = max(float(np.max(np.abs(np.linalg.eigvals(raw)))), 1e-6)
        matrices[state] = raw / radius * (0.55 + 0.30 * state / max(k - 1, 1))
    offsets = rng.normal(scale=0.35, size=(k, latent_dim))
    recurrent = rng.normal(scale=recurrent_scale, size=(k, latent_dim))
    observation = rng.normal(size=(observation_dim, latent_dim))
    return SyntheticRSLDSParameters(
        initial_state_probability=np.full(k, 1.0 / k),
        initial_continuous_mean=np.zeros(latent_dim),
        initial_continuous_covariance=np.eye(latent_dim) * 0.20,
        base_log_transition=np.log(base_probability),
        recurrent_weights=recurrent,
        dynamics_matrices=matrices,
        dynamics_offsets=offsets,
        process_variances=np.full((k, latent_dim), 0.03),
        observation_matrix=observation,
        observation_offset=np.zeros(observation_dim),
        observation_variances=np.full(observation_dim, 0.05),
    )


def sample_synthetic_rslds(
    parameters: SyntheticRSLDSParameters,
    length: int,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if length < 2:
        raise ValueError("length must be >=2")
    rng = np.random.default_rng(seed)
    z = np.empty(length, dtype=np.int64)
    x = np.empty((length, parameters.latent_dim), dtype=np.float64)
    y = np.empty((length, parameters.observation_dim), dtype=np.float64)
    z[0] = rng.choice(parameters.k, p=parameters.initial_state_probability)
    x[0] = rng.multivariate_normal(
        parameters.initial_continuous_mean,
        parameters.initial_continuous_covariance,
    )
    y[0] = (
        parameters.observation_matrix @ x[0]
        + parameters.observation_offset
        + rng.normal(scale=np.sqrt(parameters.observation_variances))
    )
    for timestep in range(1, length):
        transition = recurrent_transition_probabilities(
            parameters.base_log_transition,
            parameters.recurrent_weights,
            x[timestep - 1],
        )
        z[timestep] = rng.choice(parameters.k, p=transition[z[timestep - 1]])
        state = z[timestep]
        x[timestep] = (
            parameters.dynamics_matrices[state] @ x[timestep - 1]
            + parameters.dynamics_offsets[state]
            + rng.normal(scale=np.sqrt(parameters.process_variances[state]))
        )
        y[timestep] = (
            parameters.observation_matrix @ x[timestep]
            + parameters.observation_offset
            + rng.normal(scale=np.sqrt(parameters.observation_variances))
        )
    return z, x, y


@dataclass
class RSLDSPosterior:
    continuous_mean: list[np.ndarray]
    discrete_posterior: list[np.ndarray]
    discrete_state: list[np.ndarray]
    smoothed_observation_mean: list[np.ndarray]

    def validate(self, sequences: Sequence[np.ndarray], k: int, latent_dim: int) -> None:
        collections = (
            self.continuous_mean,
            self.discrete_posterior,
            self.discrete_state,
            self.smoothed_observation_mean,
        )
        if any(len(value) != len(sequences) for value in collections):
            raise ValueError("posterior sequence count mismatch")
        for index, observation in enumerate(sequences):
            length, n = observation.shape
            if self.continuous_mean[index].shape != (length, latent_dim):
                raise ValueError("continuous posterior shape mismatch")
            if self.discrete_posterior[index].shape != (length, k):
                raise ValueError("discrete posterior shape mismatch")
            if self.discrete_state[index].shape != (length,):
                raise ValueError("discrete state shape mismatch")
            if self.smoothed_observation_mean[index].shape != (length, n):
                raise ValueError("smoothed observation shape mismatch")
            if not np.isfinite(self.continuous_mean[index]).all():
                raise ValueError("nonfinite continuous posterior")
            if not np.isfinite(self.discrete_posterior[index]).all():
                raise ValueError("nonfinite discrete posterior")
            if not np.allclose(self.discrete_posterior[index].sum(axis=1), 1.0, atol=1e-6):
                raise ValueError("discrete posterior is not normalized")


class SSMBackendUnavailable(RuntimeError):
    pass


class SSMRecurrentSLDS:
    """Thin, verified adapter around lindermanlab/ssm's recurrent SLDS."""

    def __init__(self, *, observation_dim: int, k: int, latent_dim: int, seed: int) -> None:
        if observation_dim <= latent_dim or k < 2 or latent_dim < 1:
            raise ValueError("require observation_dim>latent_dim>=1 and K>=2")
        try:
            self.ssm = importlib.import_module("ssm")
        except Exception as error:
            raise SSMBackendUnavailable(
                "lindermanlab/ssm is unavailable; use the isolated rslds environment"
            ) from error
        np.random.seed(seed)
        self.observation_dim = int(observation_dim)
        self.k = int(k)
        self.latent_dim = int(latent_dim)
        self.seed = int(seed)
        self.model = self.ssm.SLDS(
            self.observation_dim,
            self.k,
            self.latent_dim,
            transitions="recurrent",
            dynamics="diagonal_gaussian",
            emissions="gaussian_orthog",
            single_subspace=True,
        )
        transition_name = type(self.model.transitions).__name__
        if transition_name != "RecurrentTransitions" or not hasattr(self.model.transitions, "Rs"):
            raise RuntimeError(
                f"backend constructed {transition_name}, not genuine RecurrentTransitions"
            )
        if not bool(self.model.emissions.single_subspace):
            raise RuntimeError("pilot requires a shared observation mapping")
        self.elbo_history: np.ndarray | None = None
        self.posterior_object: Any = None

    @property
    def recurrent_weights(self) -> np.ndarray:
        return np.asarray(self.model.transitions.Rs, dtype=np.float64)

    @property
    def base_log_transition(self) -> np.ndarray:
        return np.asarray(self.model.transitions.log_Ps, dtype=np.float64)

    def fit(
        self,
        sequences: Sequence[np.ndarray],
        *,
        num_iters: int = 50,
        initialize: bool = True,
        learning: bool = True,
        verbose: int = 0,
    ) -> RSLDSPosterior:
        values = [np.asarray(value, dtype=np.float64) for value in sequences]
        if not values or any(value.ndim != 2 or value.shape[1] != self.observation_dim for value in values):
            raise ValueError("every sequence must be [T,observation_dim]")
        if learning:
            # The public 0.0.1 fit API always supplies learning=True to the
            # selected optimizer; passing it again raises a duplicate-keyword
            # error.  Keep that policy in the backend and do not override it.
            result = self.model.fit(
                values,
                method="laplace_em",
                variational_posterior="structured_meanfield",
                initialize=initialize,
                num_iters=int(num_iters),
            )
            if not isinstance(result, tuple) or len(result) != 2:
                raise RuntimeError("unexpected ssm SLDS.fit return contract")
            elbos, posterior = result
        else:
            # SLDS.approximate_posterior in ssm 0.0.1 returns only its ELBO
            # history, not the posterior object it created.  Construct the
            # same structured posterior explicitly and call the package's
            # Laplace-EM implementation with learning disabled so development
            # observations cannot update model parameters.
            inputs = [np.zeros((len(value), 0), dtype=np.float64) for value in values]
            masks = [np.ones_like(value, dtype=bool) for value in values]
            tags = [None] * len(values)
            posterior = self.model._make_variational_posterior(
                "structured_meanfield",
                values,
                inputs,
                masks,
                tags,
                "laplace_em",
            )
            elbos = self.model._fit_laplace_em(
                posterior,
                values,
                inputs,
                masks,
                tags,
                learning=False,
                num_iters=int(num_iters),
            )
        self.elbo_history = np.asarray(elbos, dtype=np.float64)
        self.posterior_object = posterior
        if self.elbo_history.ndim != 1 or not np.isfinite(self.elbo_history).all():
            raise RuntimeError("nonfinite or malformed Laplace-EM ELBO history")
        x_means = [np.asarray(value, dtype=np.float64) for value in posterior.mean_continuous_states]
        probabilities: list[np.ndarray] = []
        states: list[np.ndarray] = []
        reconstructions: list[np.ndarray] = []
        for x, observation in zip(x_means, values):
            expected, _joints, _normalizer = self.model.expected_states(x, observation)
            probabilities.append(np.asarray(expected, dtype=np.float64))
            states.append(
                np.asarray(self.model.most_likely_states(x, observation), dtype=np.int64)
            )
            reconstructions.append(
                np.asarray(self.model.smooth(x, observation), dtype=np.float64)
            )
        output = RSLDSPosterior(x_means, probabilities, states, reconstructions)
        output.validate(values, self.k, self.latent_dim)
        return output

    def parameter_arrays(self) -> dict[str, np.ndarray]:
        return {
            "base_log_transition": self.base_log_transition,
            "recurrent_weights": self.recurrent_weights,
            "dynamics_matrices": np.asarray(self.model.dynamics.As, dtype=np.float64),
            "dynamics_offsets": np.asarray(self.model.dynamics.bs, dtype=np.float64),
            "process_variances": np.asarray(self.model.dynamics.sigmasq, dtype=np.float64),
            "observation_matrix": np.asarray(self.model.emissions.Cs, dtype=np.float64),
            "observation_offset": np.asarray(self.model.emissions.ds, dtype=np.float64),
            "observation_variances": np.exp(
                np.asarray(self.model.emissions.inv_etas, dtype=np.float64)
            ),
        }

    def diagnostics(self) -> dict[str, Any]:
        arrays = self.parameter_arrays()
        radii = np.asarray(
            [max(abs(np.linalg.eigvals(value))) for value in arrays["dynamics_matrices"]],
            dtype=float,
        )
        transition_probe = recurrent_transition_probabilities(
            arrays["base_log_transition"],
            arrays["recurrent_weights"],
            np.vstack([np.zeros(self.latent_dim), np.ones(self.latent_dim)]),
        )
        return {
            "backend": "lindermanlab/ssm",
            "transition_class": type(self.model.transitions).__name__,
            "inference": "Laplace-EM structured mean field",
            "exact_marginal_likelihood_available": False,
            "elbo_finite": bool(
                self.elbo_history is not None and np.isfinite(self.elbo_history).all()
            ),
            "recurrent_weight_norm": float(np.linalg.norm(arrays["recurrent_weights"])),
            "transition_probe_max_abs_change": float(
                np.max(np.abs(transition_probe[1] - transition_probe[0]))
            ),
            "dynamics_spectral_radii": radii.tolist(),
            "maximum_dynamics_spectral_radius": float(radii.max()),
            "process_variances_positive": bool(np.all(arrays["process_variances"] > 0)),
            "observation_variances_positive": bool(
                np.all(arrays["observation_variances"] > 0)
            ),
        }

    def save(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "contract": "lindermanlab_ssm_recurrent_slds_v1",
            "observation_dim": self.observation_dim,
            "k": self.k,
            "latent_dim": self.latent_dim,
            "seed": self.seed,
            "model": self.model,
            "elbo_history": self.elbo_history,
            "metadata": metadata or {},
        }
        with output.open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        return output

    @classmethod
    def load(cls, path: str | Path) -> tuple["SSMRecurrentSLDS", dict[str, Any]]:
        with Path(path).open("rb") as stream:
            payload = pickle.load(stream)
        if payload.get("contract") != "lindermanlab_ssm_recurrent_slds_v1":
            raise ValueError(f"{path}: unsupported rSLDS checkpoint")
        instance = cls(
            observation_dim=int(payload["observation_dim"]),
            k=int(payload["k"]),
            latent_dim=int(payload["latent_dim"]),
            seed=int(payload["seed"]),
        )
        instance.model = payload["model"]
        instance.elbo_history = payload["elbo_history"]
        if type(instance.model.transitions).__name__ != "RecurrentTransitions":
            raise ValueError("checkpoint is not a genuine recurrent-transition SLDS")
        return instance, dict(payload.get("metadata", {}))


def plugin_mean_forecast(
    model: SSMRecurrentSLDS,
    continuous_state: np.ndarray,
    discrete_probability: np.ndarray,
    horizons: Sequence[int],
) -> dict[int, np.ndarray]:
    """Deterministic moment plug-in forecast; this is not a predictive density."""

    x = np.asarray(continuous_state, dtype=float).copy()
    probability = np.asarray(discrete_probability, dtype=float).copy()
    if x.shape != (model.latent_dim,) or probability.shape != (model.k,):
        raise ValueError("forecast initial state shape mismatch")
    requested = sorted(set(int(value) for value in horizons))
    if not requested or requested[0] < 1:
        raise ValueError("forecast horizons must be positive")
    arrays = model.parameter_arrays()
    output: dict[int, np.ndarray] = {}
    for step in range(1, requested[-1] + 1):
        transition = recurrent_transition_probabilities(
            arrays["base_log_transition"], arrays["recurrent_weights"], x
        )
        probability = probability @ transition
        next_by_state = np.einsum("kij,j->ki", arrays["dynamics_matrices"], x)
        next_by_state += arrays["dynamics_offsets"]
        x = probability @ next_by_state
        if step in requested:
            c = arrays["observation_matrix"][0]
            d = arrays["observation_offset"][0]
            output[step] = c @ x + d
    return output


def evaluate_prefix_forecasts(
    observations: np.ndarray,
    origins: Sequence[int],
    horizons: Sequence[int],
    infer_prefix: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    forecast: Callable[[np.ndarray, np.ndarray, Sequence[int]], dict[int, np.ndarray]],
) -> list[dict[str, float | int]]:
    """Leakage-safe forecast harness: inference receives only y[:origin+1]."""

    values = np.asarray(observations, dtype=float)
    rows: list[dict[str, float | int]] = []
    for origin in sorted(set(int(value) for value in origins)):
        if origin < 0 or origin >= len(values) - 1:
            raise ValueError("forecast origin outside sequence")
        x, probability = infer_prefix(values[: origin + 1])
        predictions = forecast(x, probability, horizons)
        for horizon, prediction in predictions.items():
            target_index = origin + int(horizon)
            if target_index >= len(values):
                continue
            difference = np.asarray(prediction) - values[target_index]
            rows.append(
                {
                    "origin": origin,
                    "horizon_steps": int(horizon),
                    "target_index": target_index,
                    "squared_error": float(np.mean(difference * difference)),
                }
            )
    return rows


def save_backend_audit(path: str | Path) -> dict[str, Any]:
    """Record whether the active interpreter can instantiate genuine rSLDS."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        module = importlib.import_module("ssm")
        backend_file = str(Path(module.__file__).resolve())
        backend_version = importlib.metadata.version("ssm")
        probe = module.SLDS(
            6,
            3,
            2,
            transitions="recurrent",
            dynamics="diagonal_gaussian",
            emissions="gaussian_orthog",
            single_subspace=True,
        )
        transition_class = type(probe.transitions).__name__
        if transition_class != "RecurrentTransitions":
            raise RuntimeError(
                f"backend constructed {transition_class}, not RecurrentTransitions"
            )
        available = True
        error = ""
    except Exception as exception:
        backend_file = ""
        backend_version = ""
        transition_class = ""
        available = False
        error = f"{type(exception).__name__}: {exception}"
    record = {
        "backend": "lindermanlab/ssm",
        "available": available,
        "backend_file": backend_file,
        "backend_version": backend_version,
        "validated_transition_class": transition_class,
        "error": error,
        "required_transition": "RecurrentTransitions",
        "required_inference": "Laplace-EM structured mean field",
        "exact_marginal_likelihood": False,
    }
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
