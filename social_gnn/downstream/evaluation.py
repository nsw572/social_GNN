"""Label-invariant stability and physical-time segmentation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable, Protocol

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .embeddings import FrozenEmbeddingTrial
from .preprocessing import EmbeddingPreprocessor


class TemporalModel(Protocol):
    n_states: int

    def predict(self, sequence: np.ndarray) -> np.ndarray: ...

    def posterior(self, sequence: np.ndarray) -> np.ndarray: ...

    def score(self, sequences: Iterable[np.ndarray]) -> float: ...


@dataclass(frozen=True)
class DecodedTrial:
    trial: FrozenEmbeddingTrial
    predicted_state: np.ndarray
    posterior: np.ndarray
    state_confidence: np.ndarray
    run_indices: tuple[np.ndarray, ...]


def decode_trial(
    model: TemporalModel,
    preprocessor: EmbeddingPreprocessor,
    trial: FrozenEmbeddingTrial,
) -> DecodedTrial:
    sequences, indices = preprocessor.transform_trial_sequences(trial)
    length = trial.graph_embeddings.shape[0]
    states = np.full(length, -1, dtype=np.int64)
    posterior = np.zeros((length, model.n_states), dtype=np.float64)
    for sequence, index in zip(sequences, indices):
        state = np.asarray(model.predict(sequence), dtype=np.int64)
        probability = np.asarray(model.posterior(sequence), dtype=np.float64)
        if state.shape != (index.size,) or probability.shape != (
            index.size,
            model.n_states,
        ):
            raise ValueError(f"{trial.trial_id}: temporal model returned wrong shape")
        states[index] = state
        posterior[index] = probability
    confidence = posterior.max(axis=1)
    confidence[~trial.valid_time_mask] = 0.0
    return DecodedTrial(
        trial=trial,
        predicted_state=states,
        posterior=posterior,
        state_confidence=confidence,
        run_indices=tuple(indices),
    )


def _run_boundaries(trial: FrozenEmbeddingTrial, indices: np.ndarray) -> np.ndarray:
    centers = trial.patch_center_s[indices]
    if centers.size == 1:
        return np.asarray(
            [trial.patch_start_s[indices[0]], trial.patch_end_s[indices[0]]],
            dtype=np.float64,
        )
    boundaries = np.empty(centers.size + 1, dtype=np.float64)
    boundaries[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    boundaries[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    boundaries[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return boundaries


def bout_rows(decoded: DecodedTrial) -> list[dict[str, Any]]:
    """Convert Viterbi runs to non-overlap-counting physical-time bouts."""

    rows: list[dict[str, Any]] = []
    bout_id = 0
    for run_id, indices in enumerate(decoded.run_indices):
        states = decoded.predicted_state[indices]
        boundaries = _run_boundaries(decoded.trial, indices)
        start = 0
        for stop in range(1, states.size + 1):
            if stop == states.size or states[stop] != states[start]:
                original = indices[start:stop]
                rows.append(
                    {
                        "trial_id": decoded.trial.trial_id,
                        "run_id": run_id,
                        "bout_id": bout_id,
                        "state": int(states[start]),
                        "start_index": int(original[0]),
                        "end_index_inclusive": int(original[-1]),
                        "timesteps": int(stop - start),
                        "bout_start_s": float(boundaries[start]),
                        "bout_end_s": float(boundaries[stop]),
                        "bout_duration_s": float(boundaries[stop] - boundaries[start]),
                        "mean_state_confidence": float(
                            decoded.state_confidence[original].mean()
                        ),
                        "left_truncated": bool(start == 0),
                        "right_truncated": bool(stop == states.size),
                    }
                )
                bout_id += 1
                start = stop
    return rows


def transition_counts(
    decoded_trials: Iterable[DecodedTrial], n_states: int
) -> np.ndarray:
    counts = np.zeros((n_states, n_states), dtype=np.float64)
    for decoded in decoded_trials:
        for indices in decoded.run_indices:
            state = decoded.predicted_state[indices]
            for source, target in zip(state[:-1], state[1:]):
                counts[int(source), int(target)] += 1.0
    return counts


def normalized_transition_matrix(counts: np.ndarray) -> np.ndarray:
    denominator = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, denominator, out=np.zeros_like(counts), where=denominator > 0)


def segmentation_metrics(
    decoded_trials: Iterable[DecodedTrial],
    n_states: int,
    *,
    long_bout_s: float = 30.0,
) -> dict[str, float]:
    decoded = list(decoded_trials)
    bouts = [row for trial in decoded for row in bout_rows(trial)]
    valid_states = np.concatenate(
        [trial.predicted_state[trial.predicted_state >= 0] for trial in decoded]
    )
    occupancy = np.bincount(valid_states, minlength=n_states).astype(np.float64)
    occupancy /= max(occupancy.sum(), 1.0)
    positive = occupancy[occupancy > 0]
    effective_entropy_states = float(np.exp(-np.sum(positive * np.log(positive))))
    duration = np.asarray([row["bout_duration_s"] for row in bouts])
    timesteps = np.asarray([row["timesteps"] for row in bouts])
    total_minutes = float(duration.sum() / 60.0)
    posterior_entropy = []
    for trial in decoded:
        probability = trial.posterior[trial.trial.valid_time_mask]
        posterior_entropy.append(
            -np.sum(probability * np.log(np.maximum(probability, 1e-300)), axis=1)
        )
    entropy = np.concatenate(posterior_entropy)
    return {
        "observed_timesteps": float(valid_states.size),
        "number_of_bouts": float(len(bouts)),
        "bouts_per_minute": float(len(bouts) / max(total_minutes, 1e-12)),
        "mean_bout_duration_s": float(duration.mean()),
        "median_bout_duration_s": float(np.median(duration)),
        "fraction_one_timestep_bouts": float(np.mean(timesteps == 1)),
        "fraction_long_bouts": float(np.mean(duration >= long_bout_s)),
        "occupied_states": float(np.sum(occupancy > 0)),
        "states_over_one_percent": float(np.sum(occupancy >= 0.01)),
        "effective_entropy_states": effective_entropy_states,
        "mean_posterior_entropy": float(entropy.mean()),
        "mean_state_confidence": float(
            np.mean(
                np.concatenate(
                    [
                        trial.state_confidence[trial.trial.valid_time_mask]
                        for trial in decoded
                    ]
                )
            )
        ),
    }


def stability_metrics(state_sequences: Iterable[np.ndarray]) -> dict[str, float]:
    sequences = [np.asarray(value, dtype=np.int64) for value in state_sequences]
    if len(sequences) < 2:
        return {"mean_pairwise_ari": float("nan"), "mean_pairwise_nmi": float("nan")}
    ari: list[float] = []
    nmi: list[float] = []
    for left, right in combinations(sequences, 2):
        if left.shape != right.shape:
            raise ValueError("stability sequences must cover identical timesteps")
        valid = (left >= 0) & (right >= 0)
        ari.append(adjusted_rand_score(left[valid], right[valid]))
        nmi.append(normalized_mutual_info_score(left[valid], right[valid]))
    return {
        "mean_pairwise_ari": float(np.mean(ari)),
        "mean_pairwise_nmi": float(np.mean(nmi)),
    }
