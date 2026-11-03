from __future__ import annotations

import numpy as np

from social_gnn.downstream.embeddings import FrozenEmbeddingTrial
from social_gnn.downstream.preprocessing import EmbeddingPreprocessor
from social_gnn.downstream.rslds import (
    build_rslds_sequences,
    evaluate_prefix_forecasts,
    make_synthetic_rslds_parameters,
    recurrent_transition_probabilities,
    restore_trial_clock,
    sample_synthetic_rslds,
)


def _trial() -> FrozenEmbeddingTrial:
    length = 8
    return FrozenEmbeddingTrial(
        trial_id="trial",
        graph_embeddings=np.arange(length * 3, dtype=np.float32).reshape(length, 3),
        patch_start_s=np.arange(length) * 0.04,
        patch_end_s=(np.arange(length) + 1) * 0.04,
        valid_time_mask=np.array([True, True, False, True, True, True, False, True]),
        encoder_checkpoint="fixed",
        encoder_config="config",
    )


def test_recurrent_term_changes_transition_probabilities() -> None:
    base = np.log(np.array([[0.9, 0.1], [0.2, 0.8]]))
    weights = np.array([[2.0, 0.0], [-2.0, 0.0]])
    probability = recurrent_transition_probabilities(
        base, weights, np.array([[-1.0, 0.0], [1.0, 0.0]])
    )
    assert probability.shape == (2, 2, 2)
    np.testing.assert_allclose(probability.sum(axis=2), 1.0)
    assert not np.allclose(probability[0], probability[1])
    neutral = recurrent_transition_probabilities(base, np.zeros_like(weights), np.array([[-1.0, 0.0], [1.0, 0.0]]))
    np.testing.assert_allclose(neutral[0], neutral[1])


def test_synthetic_rslds_generation_shapes_and_finite_values() -> None:
    parameters = make_synthetic_rslds_parameters(k=3, latent_dim=2, observation_dim=5, seed=4)
    z, x, y = sample_synthetic_rslds(parameters, 200, seed=5)
    assert z.shape == (200,)
    assert x.shape == (200, 2)
    assert y.shape == (200, 5)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert set(np.unique(z)).issubset({0, 1, 2})


def test_sequence_adapter_splits_invalid_gaps_and_restores_clock() -> None:
    trial = _trial()
    preprocessor = EmbeddingPreprocessor().fit([trial])
    sequences = build_rslds_sequences([trial], preprocessor)
    assert [value.patch_indices.tolist() for value in sequences] == [[0, 1], [3, 4, 5]]
    states = [np.full(len(value.patch_indices), index, dtype=int) for index, value in enumerate(sequences)]
    restored = restore_trial_clock(trial, sequences, states, fill_value=-1)
    np.testing.assert_array_equal(restored, np.array([0, 0, -1, 1, 1, 1, -1, -1]))


def test_forecast_harness_never_passes_future_observations() -> None:
    observations = np.arange(40, dtype=float).reshape(20, 2)
    seen_lengths: list[int] = []

    def infer(prefix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        seen_lengths.append(len(prefix))
        return np.array([prefix[-1, 0]]), np.array([1.0])

    def forecast(_x: np.ndarray, _p: np.ndarray, horizons):
        return {int(h): np.zeros(2) for h in horizons}

    rows = evaluate_prefix_forecasts(observations, [3, 7], [1, 5], infer, forecast)
    assert seen_lengths == [4, 8]
    assert all(row["target_index"] > row["origin"] for row in rows)
