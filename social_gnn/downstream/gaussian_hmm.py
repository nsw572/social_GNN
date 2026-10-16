"""Reproducible diagonal-Gaussian HMM for independent trial sequences."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from numba import njit
from sklearn.cluster import KMeans


LOG_2PI = float(np.log(2.0 * np.pi))


@njit(cache=True)
def _logsumexp(values: np.ndarray) -> float:
    maximum = np.max(values)
    if not np.isfinite(maximum):
        return maximum
    return maximum + np.log(np.exp(values - maximum).sum())


@njit(cache=True)
def _forward_backward(
    log_emission: np.ndarray,
    log_initial: np.ndarray,
    log_transition: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    length, states = log_emission.shape
    alpha = np.empty((length, states))
    beta = np.empty((length, states))
    alpha[0] = log_initial + log_emission[0]
    work = np.empty(states)
    for time in range(1, length):
        for target in range(states):
            for source in range(states):
                work[source] = alpha[time - 1, source] + log_transition[source, target]
            alpha[time, target] = log_emission[time, target] + _logsumexp(work)
    likelihood = _logsumexp(alpha[length - 1])
    beta[length - 1] = 0.0
    for time in range(length - 2, -1, -1):
        for source in range(states):
            for target in range(states):
                work[target] = (
                    log_transition[source, target]
                    + log_emission[time + 1, target]
                    + beta[time + 1, target]
                )
            beta[time, source] = _logsumexp(work)
    gamma = np.exp(alpha + beta - likelihood)
    transition_counts = np.zeros((states, states))
    for time in range(length - 1):
        for source in range(states):
            for target in range(states):
                transition_counts[source, target] += np.exp(
                    alpha[time, source]
                    + log_transition[source, target]
                    + log_emission[time + 1, target]
                    + beta[time + 1, target]
                    - likelihood
                )
    return alpha, gamma, likelihood, transition_counts


@njit(cache=True)
def _viterbi(
    log_emission: np.ndarray,
    log_initial: np.ndarray,
    log_transition: np.ndarray,
) -> tuple[np.ndarray, float]:
    length, states = log_emission.shape
    score = np.empty((length, states))
    back = np.empty((length, states), dtype=np.int64)
    score[0] = log_initial + log_emission[0]
    back[0] = -1
    for time in range(1, length):
        for target in range(states):
            values = score[time - 1] + log_transition[:, target]
            source = int(np.argmax(values))
            score[time, target] = values[source] + log_emission[time, target]
            back[time, target] = source
    path = np.empty(length, dtype=np.int64)
    path[-1] = int(np.argmax(score[-1]))
    for time in range(length - 2, -1, -1):
        path[time] = back[time + 1, path[time + 1]]
    return path, float(np.max(score[-1]))


def _validate_sequences(sequences: Iterable[np.ndarray]) -> list[np.ndarray]:
    values = [np.asarray(sequence, dtype=np.float64) for sequence in sequences]
    if not values:
        raise ValueError("at least one sequence is required")
    dimension = values[0].shape[1] if values[0].ndim == 2 else -1
    for index, sequence in enumerate(values):
        if sequence.ndim != 2 or sequence.shape[0] < 1 or sequence.shape[1] != dimension:
            raise ValueError(f"sequence {index} has invalid shape {sequence.shape}")
        if not np.isfinite(sequence).all():
            raise ValueError(f"sequence {index} contains nonfinite values")
    return values


@dataclass
class GaussianHMM:
    n_states: int
    covariance_floor: float = 1e-3
    transition_pseudocount: float = 1e-2
    initial_pseudocount: float = 1e-2
    seed: int = 0
    initial_: np.ndarray | None = None
    transition_: np.ndarray | None = None
    means_: np.ndarray | None = None
    variances_: np.ndarray | None = None
    history_: list[float] = field(default_factory=list)
    converged_: bool = False

    def _check_fitted(self) -> None:
        if any(
            value is None
            for value in (self.initial_, self.transition_, self.means_, self.variances_)
        ):
            raise RuntimeError("GaussianHMM is not fitted")

    @property
    def n_features(self) -> int:
        self._check_fitted()
        return int(self.means_.shape[1])

    def _initialize(self, sequences: list[np.ndarray]) -> None:
        if self.n_states < 2:
            raise ValueError("n_states must be >= 2")
        joined = np.concatenate(sequences, axis=0)
        if joined.shape[0] < self.n_states:
            raise ValueError("fewer observations than states")
        labels = KMeans(
            n_clusters=self.n_states,
            n_init=1,
            random_state=self.seed,
            max_iter=100,
        ).fit_predict(joined)
        self.means_ = np.stack(
            [joined[labels == state].mean(axis=0) for state in range(self.n_states)]
        )
        global_variance = np.maximum(joined.var(axis=0), self.covariance_floor)
        self.variances_ = np.stack(
            [
                np.maximum(
                    joined[labels == state].var(axis=0), self.covariance_floor
                )
                if np.any(labels == state)
                else global_variance
                for state in range(self.n_states)
            ]
        )
        self.initial_ = np.full(self.n_states, 1.0 / self.n_states)
        off = 0.10 / max(self.n_states - 1, 1)
        self.transition_ = np.full((self.n_states, self.n_states), off)
        np.fill_diagonal(self.transition_, 0.90)
        self.transition_ /= self.transition_.sum(axis=1, keepdims=True)

    def emission_log_probability(self, sequence: np.ndarray) -> np.ndarray:
        self._check_fitted()
        values = np.asarray(sequence, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.n_features:
            raise ValueError(f"expected [T,{self.n_features}], got {values.shape}")
        difference = values[:, None, :] - self.means_[None, :, :]
        return -0.5 * (
            np.sum(difference * difference / self.variances_[None, :, :], axis=2)
            + np.sum(np.log(self.variances_), axis=1)[None, :]
            + self.n_features * LOG_2PI
        )

    def fit(
        self,
        sequences: Iterable[np.ndarray],
        *,
        max_iter: int = 50,
        tolerance: float = 1e-4,
    ) -> "GaussianHMM":
        values = _validate_sequences(sequences)
        self._initialize(values)
        self.history_ = []
        previous = -np.inf
        for _iteration in range(max_iter):
            initial_counts = np.full(self.n_states, self.initial_pseudocount)
            transition_counts = np.full(
                (self.n_states, self.n_states), self.transition_pseudocount
            )
            state_weight = np.zeros(self.n_states)
            first_moment = np.zeros_like(self.means_)
            second_moment = np.zeros_like(self.means_)
            total_likelihood = 0.0
            log_initial = np.log(np.maximum(self.initial_, 1e-300))
            log_transition = np.log(np.maximum(self.transition_, 1e-300))
            for sequence in values:
                emission = self.emission_log_probability(sequence)
                _alpha, gamma, likelihood, xi = _forward_backward(
                    emission, log_initial, log_transition
                )
                total_likelihood += likelihood
                initial_counts += gamma[0]
                transition_counts += xi
                state_weight += gamma.sum(axis=0)
                first_moment += gamma.T @ sequence
                second_moment += gamma.T @ (sequence * sequence)
            self.initial_ = initial_counts / initial_counts.sum()
            self.transition_ = transition_counts / transition_counts.sum(
                axis=1, keepdims=True
            )
            safe_weight = np.maximum(state_weight, 1e-12)[:, None]
            old_means = self.means_.copy()
            old_variances = self.variances_.copy()
            occupied = state_weight > 1e-8
            self.means_[occupied] = (first_moment / safe_weight)[occupied]
            variance = second_moment / safe_weight - self.means_ * self.means_
            self.variances_[occupied] = np.maximum(
                variance[occupied], self.covariance_floor
            )
            self.means_[~occupied] = old_means[~occupied]
            self.variances_[~occupied] = old_variances[~occupied]
            self.history_.append(float(total_likelihood))
            if np.isfinite(previous):
                relative = abs(total_likelihood - previous) / max(1.0, abs(previous))
                if relative < tolerance:
                    self.converged_ = True
                    break
            previous = total_likelihood
        return self

    def score(self, sequences: Iterable[np.ndarray]) -> float:
        values = _validate_sequences(sequences)
        self._check_fitted()
        log_initial = np.log(np.maximum(self.initial_, 1e-300))
        log_transition = np.log(np.maximum(self.transition_, 1e-300))
        total = 0.0
        for sequence in values:
            _alpha, _gamma, likelihood, _xi = _forward_backward(
                self.emission_log_probability(sequence), log_initial, log_transition
            )
            total += likelihood
        return float(total)

    def posterior(self, sequence: np.ndarray) -> np.ndarray:
        self._check_fitted()
        _alpha, gamma, _likelihood, _xi = _forward_backward(
            self.emission_log_probability(sequence),
            np.log(np.maximum(self.initial_, 1e-300)),
            np.log(np.maximum(self.transition_, 1e-300)),
        )
        return gamma / np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)

    def predict(self, sequence: np.ndarray) -> np.ndarray:
        self._check_fitted()
        path, _score = _viterbi(
            self.emission_log_probability(sequence),
            np.log(np.maximum(self.initial_, 1e-300)),
            np.log(np.maximum(self.transition_, 1e-300)),
        )
        return path

    def parameter_count(self) -> int:
        self._check_fitted()
        return (
            self.n_states - 1
            + self.n_states * (self.n_states - 1)
            + 2 * self.n_states * self.n_features
        )

    def save(self, path: str | Path, *, metadata: dict | None = None) -> Path:
        self._check_fitted()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            model_type=np.asarray("gaussian_hmm_v1"),
            n_states=np.asarray(self.n_states),
            covariance_floor=np.asarray(self.covariance_floor),
            transition_pseudocount=np.asarray(self.transition_pseudocount),
            initial_pseudocount=np.asarray(self.initial_pseudocount),
            seed=np.asarray(self.seed),
            initial=self.initial_,
            transition=self.transition_,
            means=self.means_,
            variances=self.variances_,
            history=np.asarray(self.history_),
            converged=np.asarray(self.converged_),
            metadata_json=np.asarray(json.dumps(metadata or {}, ensure_ascii=False)),
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> tuple["GaussianHMM", dict]:
        with np.load(path, allow_pickle=False) as data:
            if str(data["model_type"].item()) != "gaussian_hmm_v1":
                raise ValueError(f"{path} is not a GaussianHMM checkpoint")
            model = cls(
                n_states=int(data["n_states"]),
                covariance_floor=float(data["covariance_floor"]),
                transition_pseudocount=float(data["transition_pseudocount"]),
                initial_pseudocount=float(data["initial_pseudocount"]),
                seed=int(data["seed"]),
            )
            model.initial_ = np.asarray(data["initial"], dtype=np.float64)
            model.transition_ = np.asarray(data["transition"], dtype=np.float64)
            model.means_ = np.asarray(data["means"], dtype=np.float64)
            model.variances_ = np.asarray(data["variances"], dtype=np.float64)
            model.history_ = np.asarray(data["history"], dtype=np.float64).tolist()
            model.converged_ = bool(data["converged"])
            metadata = json.loads(str(data["metadata_json"].item()))
        return model, metadata
