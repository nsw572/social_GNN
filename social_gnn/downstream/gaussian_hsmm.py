"""Explicit-duration diagonal-Gaussian hidden semi-Markov model.

Durations follow state-specific shifted Poisson distributions truncated at a
configured maximum. Self-transitions are excluded because persistence is
represented by the duration distribution, not by repeated one-step states.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from numba import njit
from scipy.optimize import minimize_scalar
from scipy.special import gammaln, logsumexp

from .gaussian_hmm import GaussianHMM, LOG_2PI, _validate_sequences


@njit(cache=True)
def _lse(values: np.ndarray) -> float:
    maximum = np.max(values)
    if not np.isfinite(maximum):
        return maximum
    return maximum + np.log(np.exp(values - maximum).sum())


@njit(cache=True)
def _logadd(a: float, b: float) -> float:
    if not np.isfinite(a):
        return b
    if not np.isfinite(b):
        return a
    maximum = max(a, b)
    return maximum + np.log(np.exp(a - maximum) + np.exp(b - maximum))


@njit(cache=True)
def _segment_sum(cumulative: np.ndarray, start: int, stop: int, state: int) -> float:
    return cumulative[stop, state] - cumulative[start, state]


@njit(cache=True)
def _hsmm_forward(
    log_emission: np.ndarray,
    log_initial: np.ndarray,
    log_transition: np.ndarray,
    log_duration: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    length, states = log_emission.shape
    max_duration = log_duration.shape[1]
    cumulative = np.zeros((length + 1, states))
    for time in range(length):
        cumulative[time + 1] = cumulative[time] + log_emission[time]
    alpha = np.full((length, states), -np.inf)
    incoming = np.full((length, states), -np.inf)
    incoming[0] = log_initial
    work = np.empty(states)
    for end in range(length):
        upper = min(max_duration, end + 1)
        for state in range(states):
            value = -np.inf
            for duration in range(1, upper + 1):
                start = end - duration + 1
                candidate = (
                    incoming[start, state]
                    + log_duration[state, duration - 1]
                    + _segment_sum(cumulative, start, end + 1, state)
                )
                value = _logadd(value, candidate)
            alpha[end, state] = value
        if end + 1 < length:
            for target in range(states):
                for source in range(states):
                    work[source] = alpha[end, source] + log_transition[source, target]
                incoming[end + 1, target] = _lse(work)
    return alpha, cumulative, _lse(alpha[length - 1])


@njit(cache=True)
def _hsmm_backward(
    cumulative: np.ndarray,
    log_transition: np.ndarray,
    log_duration: np.ndarray,
) -> np.ndarray:
    length = cumulative.shape[0] - 1
    states = cumulative.shape[1]
    max_duration = log_duration.shape[1]
    beta = np.full((length, states), -np.inf)
    beta[length - 1] = 0.0
    for end in range(length - 2, -1, -1):
        start = end + 1
        upper = min(max_duration, length - start)
        for current in range(states):
            value = -np.inf
            for target in range(states):
                if target == current:
                    continue
                for duration in range(1, upper + 1):
                    next_end = start + duration - 1
                    candidate = (
                        log_transition[current, target]
                        + log_duration[target, duration - 1]
                        + _segment_sum(cumulative, start, next_end + 1, target)
                        + beta[next_end, target]
                    )
                    value = _logadd(value, candidate)
            beta[end, current] = value
    return beta


@njit(cache=True)
def _hsmm_expectations(
    log_emission: np.ndarray,
    log_initial: np.ndarray,
    log_transition: np.ndarray,
    log_duration: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    alpha, cumulative, likelihood = _hsmm_forward(
        log_emission, log_initial, log_transition, log_duration
    )
    beta = _hsmm_backward(cumulative, log_transition, log_duration)
    length, states = log_emission.shape
    max_duration = log_duration.shape[1]
    incoming = np.full((length, states), -np.inf)
    incoming[0] = log_initial
    work = np.empty(states)
    for start in range(1, length):
        for target in range(states):
            for source in range(states):
                work[source] = alpha[start - 1, source] + log_transition[source, target]
            incoming[start, target] = _lse(work)

    gamma_difference = np.zeros((length + 1, states))
    initial_counts = np.zeros(states)
    duration_counts = np.zeros((states, max_duration))
    entry_log_mass = np.full((length, states), -np.inf)
    for start in range(length):
        upper = min(max_duration, length - start)
        for state in range(states):
            for duration in range(1, upper + 1):
                end = start + duration - 1
                local = (
                    log_duration[state, duration - 1]
                    + _segment_sum(cumulative, start, end + 1, state)
                    + beta[end, state]
                )
                posterior = np.exp(incoming[start, state] + local - likelihood)
                gamma_difference[start, state] += posterior
                gamma_difference[end + 1, state] -= posterior
                duration_counts[state, duration - 1] += posterior
                entry_log_mass[start, state] = _logadd(
                    entry_log_mass[start, state], local
                )
                if start == 0:
                    initial_counts[state] += posterior

    gamma = np.empty((length, states))
    running = np.zeros(states)
    for time in range(length):
        running += gamma_difference[time]
        gamma[time] = running

    transition_counts = np.zeros((states, states))
    for start in range(1, length):
        for source in range(states):
            for target in range(states):
                if source == target:
                    continue
                transition_counts[source, target] += np.exp(
                    alpha[start - 1, source]
                    + log_transition[source, target]
                    + entry_log_mass[start, target]
                    - likelihood
                )
    return gamma, initial_counts, transition_counts, duration_counts, likelihood


@njit(cache=True)
def _hsmm_viterbi(
    log_emission: np.ndarray,
    log_initial: np.ndarray,
    log_transition: np.ndarray,
    log_duration: np.ndarray,
) -> tuple[np.ndarray, float]:
    length, states = log_emission.shape
    max_duration = log_duration.shape[1]
    cumulative = np.zeros((length + 1, states))
    for time in range(length):
        cumulative[time + 1] = cumulative[time] + log_emission[time]
    delta = np.full((length, states), -np.inf)
    incoming = np.full((length, states), -np.inf)
    incoming_state = np.full((length, states), -1, dtype=np.int64)
    incoming[0] = log_initial
    back_state = np.full((length, states), -1, dtype=np.int64)
    back_duration = np.ones((length, states), dtype=np.int64)
    for end in range(length):
        upper = min(max_duration, end + 1)
        for state in range(states):
            best = -np.inf
            best_duration = 1
            best_state = -1
            for duration in range(1, upper + 1):
                start = end - duration + 1
                candidate = (
                    incoming[start, state]
                    + log_duration[state, duration - 1]
                    + _segment_sum(cumulative, start, end + 1, state)
                )
                if candidate > best:
                    best = candidate
                    best_duration = duration
                    best_state = incoming_state[start, state]
            delta[end, state] = best
            back_duration[end, state] = best_duration
            back_state[end, state] = best_state
        if end + 1 < length:
            for target in range(states):
                best_source = -1
                best = -np.inf
                for source in range(states):
                    value = delta[end, source] + log_transition[source, target]
                    if value > best:
                        best = value
                        best_source = source
                incoming[end + 1, target] = best
                incoming_state[end + 1, target] = best_source
    path = np.empty(length, dtype=np.int64)
    state = int(np.argmax(delta[length - 1]))
    score = float(delta[length - 1, state])
    end = length - 1
    while end >= 0:
        duration = back_duration[end, state]
        start = end - duration + 1
        for time in range(start, end + 1):
            path[time] = state
        state = back_state[end, state]
        end = start - 1
    return path, score


@dataclass
class GaussianHSMM:
    n_states: int
    max_duration: int = 50
    covariance_floor: float = 1e-3
    transition_pseudocount: float = 1e-2
    duration_pseudocount: float = 1e-3
    initial_pseudocount: float = 1e-2
    seed: int = 0
    initial_: np.ndarray | None = None
    transition_: np.ndarray | None = None
    means_: np.ndarray | None = None
    variances_: np.ndarray | None = None
    duration_lambda_: np.ndarray | None = None
    history_: list[float] = field(default_factory=list)
    converged_: bool = False

    def _check_fitted(self) -> None:
        if any(
            value is None
            for value in (
                self.initial_,
                self.transition_,
                self.means_,
                self.variances_,
                self.duration_lambda_,
            )
        ):
            raise RuntimeError("GaussianHSMM is not fitted")

    @property
    def n_features(self) -> int:
        self._check_fitted()
        return int(self.means_.shape[1])

    def duration_log_probability(self) -> np.ndarray:
        self._check_fitted()
        duration = np.arange(1, self.max_duration + 1, dtype=np.float64)
        count = duration - 1.0
        result = np.empty((self.n_states, self.max_duration))
        for state, rate in enumerate(self.duration_lambda_):
            raw = count * np.log(max(rate, 1e-8)) - rate - gammaln(count + 1.0)
            result[state] = raw - logsumexp(raw)
        return result

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

    def _initialize(self, sequences: list[np.ndarray]) -> None:
        if self.max_duration < 2:
            raise ValueError("max_duration must be >= 2")
        hmm = GaussianHMM(
            n_states=self.n_states,
            covariance_floor=self.covariance_floor,
            seed=self.seed,
        ).fit(sequences, max_iter=12)
        self.means_ = hmm.means_.copy()
        self.variances_ = hmm.variances_.copy()
        initial = np.full(self.n_states, self.initial_pseudocount)
        transition = np.full(
            (self.n_states, self.n_states), self.transition_pseudocount
        )
        np.fill_diagonal(transition, 0.0)
        durations: list[list[int]] = [[] for _ in range(self.n_states)]
        for sequence in sequences:
            path = hmm.predict(sequence)
            initial[path[0]] += 1.0
            start = 0
            for stop in range(1, path.size + 1):
                if stop == path.size or path[stop] != path[start]:
                    state = int(path[start])
                    durations[state].append(min(stop - start, self.max_duration))
                    if stop < path.size:
                        transition[state, int(path[stop])] += 1.0
                    start = stop
        self.initial_ = initial / initial.sum()
        self.transition_ = transition / np.maximum(
            transition.sum(axis=1, keepdims=True), 1e-300
        )
        self.duration_lambda_ = np.asarray(
            [
                max(0.05, float(np.mean(value) - 1.0)) if value else 1.0
                for value in durations
            ],
            dtype=np.float64,
        )

    def _update_duration(self, counts: np.ndarray) -> np.ndarray:
        duration = np.arange(1, self.max_duration + 1, dtype=np.float64)
        shifted = duration - 1.0
        updated = np.empty(self.n_states)
        for state in range(self.n_states):
            weight = counts[state] + self.duration_pseudocount

            def objective(rate: float) -> float:
                raw = shifted * np.log(rate) - rate - gammaln(shifted + 1.0)
                log_probability = raw - logsumexp(raw)
                return -float(np.dot(weight, log_probability))

            result = minimize_scalar(
                objective,
                bounds=(0.01, max(2.0, self.max_duration * 3.0)),
                method="bounded",
                options={"xatol": 1e-4},
            )
            updated[state] = float(result.x)
        return updated

    def fit(
        self,
        sequences: Iterable[np.ndarray],
        *,
        max_iter: int = 30,
        tolerance: float = 1e-4,
    ) -> "GaussianHSMM":
        values = _validate_sequences(sequences)
        self._initialize(values)
        self.history_ = []
        previous = -np.inf
        for _iteration in range(max_iter):
            initial_counts = np.full(self.n_states, self.initial_pseudocount)
            transition_counts = np.full(
                (self.n_states, self.n_states), self.transition_pseudocount
            )
            np.fill_diagonal(transition_counts, 0.0)
            duration_counts = np.zeros((self.n_states, self.max_duration))
            state_weight = np.zeros(self.n_states)
            first_moment = np.zeros_like(self.means_)
            second_moment = np.zeros_like(self.means_)
            total_likelihood = 0.0
            log_initial = np.log(np.maximum(self.initial_, 1e-300))
            log_transition = np.full_like(self.transition_, -np.inf)
            allowed = ~np.eye(self.n_states, dtype=bool)
            log_transition[allowed] = np.log(
                np.maximum(self.transition_[allowed], 1e-300)
            )
            log_duration = self.duration_log_probability()
            for sequence in values:
                gamma, initial, transition, duration, likelihood = _hsmm_expectations(
                    self.emission_log_probability(sequence),
                    log_initial,
                    log_transition,
                    log_duration,
                )
                total_likelihood += likelihood
                initial_counts += initial
                transition_counts += transition
                duration_counts += duration
                state_weight += gamma.sum(axis=0)
                first_moment += gamma.T @ sequence
                second_moment += gamma.T @ (sequence * sequence)
            self.initial_ = initial_counts / initial_counts.sum()
            np.fill_diagonal(transition_counts, 0.0)
            self.transition_ = transition_counts / np.maximum(
                transition_counts.sum(axis=1, keepdims=True), 1e-300
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
            self.duration_lambda_ = self._update_duration(duration_counts)
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
        log_transition = np.full_like(self.transition_, -np.inf)
        allowed = ~np.eye(self.n_states, dtype=bool)
        log_transition[allowed] = np.log(
            np.maximum(self.transition_[allowed], 1e-300)
        )
        duration = self.duration_log_probability()
        total = 0.0
        for sequence in values:
            _alpha, _cumulative, likelihood = _hsmm_forward(
                self.emission_log_probability(sequence),
                log_initial,
                log_transition,
                duration,
            )
            total += likelihood
        return float(total)

    def posterior(self, sequence: np.ndarray) -> np.ndarray:
        self._check_fitted()
        transition = np.full_like(self.transition_, -np.inf)
        allowed = ~np.eye(self.n_states, dtype=bool)
        transition[allowed] = np.log(np.maximum(self.transition_[allowed], 1e-300))
        gamma, _initial, _xi, _duration, _likelihood = _hsmm_expectations(
            self.emission_log_probability(sequence),
            np.log(np.maximum(self.initial_, 1e-300)),
            transition,
            self.duration_log_probability(),
        )
        return gamma / np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)

    def predict(self, sequence: np.ndarray) -> np.ndarray:
        self._check_fitted()
        transition = np.full_like(self.transition_, -np.inf)
        allowed = ~np.eye(self.n_states, dtype=bool)
        transition[allowed] = np.log(np.maximum(self.transition_[allowed], 1e-300))
        path, _score = _hsmm_viterbi(
            self.emission_log_probability(sequence),
            np.log(np.maximum(self.initial_, 1e-300)),
            transition,
            self.duration_log_probability(),
        )
        return path

    def parameter_count(self) -> int:
        self._check_fitted()
        transition_parameters = self.n_states * max(self.n_states - 2, 0)
        return (
            self.n_states - 1
            + transition_parameters
            + 2 * self.n_states * self.n_features
            + self.n_states
        )

    def save(self, path: str | Path, *, metadata: dict | None = None) -> Path:
        self._check_fitted()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            model_type=np.asarray("gaussian_hsmm_v1"),
            n_states=np.asarray(self.n_states),
            max_duration=np.asarray(self.max_duration),
            covariance_floor=np.asarray(self.covariance_floor),
            transition_pseudocount=np.asarray(self.transition_pseudocount),
            duration_pseudocount=np.asarray(self.duration_pseudocount),
            initial_pseudocount=np.asarray(self.initial_pseudocount),
            seed=np.asarray(self.seed),
            initial=self.initial_,
            transition=self.transition_,
            means=self.means_,
            variances=self.variances_,
            duration_lambda=self.duration_lambda_,
            history=np.asarray(self.history_),
            converged=np.asarray(self.converged_),
            metadata_json=np.asarray(json.dumps(metadata or {}, ensure_ascii=False)),
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> tuple["GaussianHSMM", dict]:
        with np.load(path, allow_pickle=False) as data:
            if str(data["model_type"].item()) != "gaussian_hsmm_v1":
                raise ValueError(f"{path} is not a GaussianHSMM checkpoint")
            model = cls(
                n_states=int(data["n_states"]),
                max_duration=int(data["max_duration"]),
                covariance_floor=float(data["covariance_floor"]),
                transition_pseudocount=float(data["transition_pseudocount"]),
                duration_pseudocount=float(data["duration_pseudocount"]),
                initial_pseudocount=float(data["initial_pseudocount"]),
                seed=int(data["seed"]),
            )
            model.initial_ = np.asarray(data["initial"], dtype=np.float64)
            model.transition_ = np.asarray(data["transition"], dtype=np.float64)
            model.means_ = np.asarray(data["means"], dtype=np.float64)
            model.variances_ = np.asarray(data["variances"], dtype=np.float64)
            model.duration_lambda_ = np.asarray(
                data["duration_lambda"], dtype=np.float64
            )
            model.history_ = np.asarray(data["history"], dtype=np.float64).tolist()
            model.converged_ = bool(data["converged"])
            metadata = json.loads(str(data["metadata_json"].item()))
        return model, metadata
