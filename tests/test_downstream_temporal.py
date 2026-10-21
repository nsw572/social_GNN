import itertools
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from social_gnn.downstream.embeddings import FrozenEmbeddingTrial
from social_gnn.downstream.evaluation import (
    DecodedTrial,
    bout_rows,
    transition_counts,
)
from social_gnn.downstream.gaussian_hmm import GaussianHMM
from social_gnn.downstream.gaussian_hsmm import GaussianHSMM
from social_gnn.downstream.preprocessing import EmbeddingPreprocessor


def make_trial(trial_id="trial", values=None, valid=None, duration=0.32, stride=0.16):
    if values is None:
        values = np.arange(20, dtype=np.float32).reshape(10, 2)
    length = values.shape[0]
    start = np.arange(length, dtype=np.float64) * stride
    if valid is None:
        valid = np.ones(length, dtype=bool)
    return FrozenEmbeddingTrial(
        trial_id=trial_id,
        graph_embeddings=np.asarray(values, dtype=np.float32),
        patch_start_s=start,
        patch_end_s=start + duration,
        valid_time_mask=np.asarray(valid, dtype=bool),
        encoder_checkpoint="checkpoint.pt",
        encoder_config="config.yaml",
    )


class DownstreamTemporalTest(unittest.TestCase):
    def test_embedding_dimensions_and_chronology_are_strict(self):
        with self.assertRaisesRegex(ValueError, r"shape \[T,D\]"):
            make_trial(values=np.zeros(3, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "strictly chronological"):
            FrozenEmbeddingTrial(
                trial_id="bad-clock",
                graph_embeddings=np.zeros((3, 2), dtype=np.float32),
                patch_start_s=np.asarray([0.0, 0.16, 0.16]),
                patch_end_s=np.asarray([0.32, 0.48, 0.48]),
                valid_time_mask=np.ones(3, dtype=bool),
                encoder_checkpoint="checkpoint.pt",
                encoder_config="config.yaml",
            )

    def test_invalid_patch_splits_sequence_without_reordering(self):
        trial = make_trial(valid=[True, True, False, True, True, True, False, True, True, True])
        sequences, indices = trial.model_sequences()
        self.assertEqual([value.tolist() for value in indices], [[0, 1], [3, 4, 5], [7, 8, 9]])
        np.testing.assert_array_equal(sequences[1], trial.graph_embeddings[[3, 4, 5]])

    def test_embedding_confidence_metadata_checkpoint_roundtrip(self):
        trial = make_trial()
        enriched = FrozenEmbeddingTrial(
            trial_id=trial.trial_id,
            graph_embeddings=trial.graph_embeddings,
            patch_start_s=trial.patch_start_s,
            patch_end_s=trial.patch_end_s,
            valid_time_mask=trial.valid_time_mask,
            encoder_checkpoint=trial.encoder_checkpoint,
            encoder_config=trial.encoder_config,
            node_valid_fraction=np.linspace(0.5, 1.0, 10, dtype=np.float32),
            edge_confidence_mean=np.full((10, 8), 0.75, dtype=np.float32),
        )
        directory = Path("social_gnn") / f"_test_downstream_{uuid.uuid4().hex}"
        directory.mkdir(parents=True)
        try:
            restored = FrozenEmbeddingTrial.load(enriched.save(directory / "embedding.npz"))
            np.testing.assert_array_equal(
                restored.node_valid_fraction, enriched.node_valid_fraction
            )
            np.testing.assert_array_equal(
                restored.edge_confidence_mean, enriched.edge_confidence_mean
            )
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_train_only_preprocessing(self):
        train = make_trial("train", np.asarray([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]]))
        validation = make_trial("validation", np.full((3, 2), 1000.0))
        preprocessing = EmbeddingPreprocessor().fit([train])
        transformed = preprocessing.transform_array(train.graph_embeddings)
        np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-12)
        self.assertGreater(preprocessing.transform_array(validation.graph_embeddings).mean(), 100.0)

    def test_hmm_forward_and_viterbi_match_exact_enumeration(self):
        model = GaussianHMM(n_states=2)
        model.initial_ = np.asarray([0.6, 0.4])
        model.transition_ = np.asarray([[0.75, 0.25], [0.35, 0.65]])
        model.means_ = np.asarray([[-1.0], [1.0]])
        model.variances_ = np.asarray([[0.7], [0.8]])
        sequence = np.asarray([[-0.8], [0.2], [1.1]])
        emission = model.emission_log_probability(sequence)
        scores = []
        paths = []
        for path in itertools.product(range(2), repeat=3):
            score = np.log(model.initial_[path[0]]) + emission[0, path[0]]
            for time in range(1, 3):
                score += np.log(model.transition_[path[time - 1], path[time]])
                score += emission[time, path[time]]
            scores.append(score)
            paths.append(path)
        self.assertAlmostEqual(model.score([sequence]), float(logsumexp(scores)), places=9)
        np.testing.assert_array_equal(model.predict(sequence), paths[int(np.argmax(scores))])

    def test_diagonal_gaussian_emission_matches_closed_form(self):
        model = GaussianHMM(n_states=1)
        model.initial_ = np.ones(1)
        model.transition_ = np.ones((1, 1))
        model.means_ = np.asarray([[0.0]])
        model.variances_ = np.asarray([[4.0]])
        actual = model.emission_log_probability(np.asarray([[2.0]]))[0, 0]
        expected = -0.5 * (np.log(2.0 * np.pi) + np.log(4.0) + 1.0)
        self.assertAlmostEqual(actual, expected, places=12)

    def _enumerate_hsmm(self, model, sequence):
        emission = model.emission_log_probability(sequence)
        log_duration = model.duration_log_probability()
        candidates = []

        def recurse(position, previous, score, path):
            if position == len(sequence):
                candidates.append((score, np.asarray(path, dtype=np.int64)))
                return
            for state in range(model.n_states):
                if state == previous:
                    continue
                prefix = np.log(model.initial_[state]) if position == 0 else np.log(model.transition_[previous, state])
                for duration in range(1, min(model.max_duration, len(sequence) - position) + 1):
                    local = log_duration[state, duration - 1] + emission[position : position + duration, state].sum()
                    recurse(position + duration, state, score + prefix + local, path + [state] * duration)

        recurse(0, -1, 0.0, [])
        return candidates

    def test_hsmm_forward_posterior_and_viterbi_match_enumeration(self):
        model = GaussianHSMM(n_states=2, max_duration=3)
        model.initial_ = np.asarray([0.55, 0.45])
        model.transition_ = np.asarray([[0.0, 1.0], [1.0, 0.0]])
        model.means_ = np.asarray([[-1.0], [1.0]])
        model.variances_ = np.asarray([[1.0], [1.0]])
        model.duration_lambda_ = np.asarray([1.2, 0.7])
        sequence = np.asarray([[-1.2], [-0.7], [0.8], [1.1]])
        candidates = self._enumerate_hsmm(model, sequence)
        scores = np.asarray([value[0] for value in candidates])
        likelihood = float(logsumexp(scores))
        self.assertAlmostEqual(model.score([sequence]), likelihood, places=8)
        np.testing.assert_array_equal(
            model.predict(sequence), candidates[int(np.argmax(scores))][1]
        )
        exact = np.zeros((len(sequence), 2))
        weight = np.exp(scores - likelihood)
        for probability, (_score, path) in zip(weight, candidates):
            exact[np.arange(len(sequence)), path] += probability
        np.testing.assert_allclose(model.posterior(sequence), exact, atol=1e-8)

    def test_bout_seconds_use_center_stride_not_overlapping_window_duration(self):
        trial = make_trial(values=np.zeros((3, 2)), duration=0.32, stride=0.16)
        decoded = DecodedTrial(
            trial=trial,
            predicted_state=np.asarray([1, 1, 1]),
            posterior=np.asarray([[0.0, 1.0]] * 3),
            state_confidence=np.ones(3),
            run_indices=(np.arange(3),),
        )
        rows = bout_rows(decoded)
        self.assertEqual(rows[0]["timesteps"], 3)
        self.assertAlmostEqual(rows[0]["bout_duration_s"], 0.48, places=12)

    def test_transition_counts_never_cross_trials_or_invalid_runs(self):
        first = make_trial("first", values=np.zeros((2, 2)))
        second = make_trial("second", values=np.zeros((2, 2)))
        decoded = [
            DecodedTrial(first, np.asarray([0, 1]), np.ones((2, 2)) / 2, np.ones(2) / 2, (np.arange(2),)),
            DecodedTrial(second, np.asarray([1, 0]), np.ones((2, 2)) / 2, np.ones(2) / 2, (np.arange(2),)),
        ]
        counts = transition_counts(decoded, 2)
        np.testing.assert_array_equal(counts, np.asarray([[0.0, 1.0], [1.0, 0.0]]))

    def test_fixed_seed_fit_and_checkpoint_roundtrip(self):
        rng = np.random.default_rng(5)
        sequences = [rng.normal(size=(30, 2)), rng.normal(loc=1.0, size=(25, 2))]
        first = GaussianHMM(3, seed=9).fit(sequences, max_iter=3)
        second = GaussianHMM(3, seed=9).fit(sequences, max_iter=3)
        np.testing.assert_allclose(first.means_, second.means_)
        directory = Path("social_gnn") / f"_test_downstream_{uuid.uuid4().hex}"
        directory.mkdir(parents=True)
        try:
            path = directory / "model.npz"
            first.save(path, metadata={"purpose": "test"})
            loaded, metadata = GaussianHMM.load(path)
            np.testing.assert_allclose(first.predict(sequences[0]), loaded.predict(sequences[0]))
            self.assertEqual(metadata["purpose"], "test")
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_hsmm_checkpoint_roundtrip(self):
        model = GaussianHSMM(n_states=2, max_duration=4, seed=7)
        model.initial_ = np.asarray([0.6, 0.4])
        model.transition_ = np.asarray([[0.0, 1.0], [1.0, 0.0]])
        model.means_ = np.asarray([[-1.0], [1.0]])
        model.variances_ = np.asarray([[0.7], [0.9]])
        model.duration_lambda_ = np.asarray([1.5, 2.5])
        sequence = np.asarray([[-1.2], [-0.8], [0.7], [1.3]])
        directory = Path("social_gnn") / f"_test_downstream_{uuid.uuid4().hex}"
        directory.mkdir(parents=True)
        try:
            path = directory / "hsmm.npz"
            model.save(path, metadata={"purpose": "hsmm-test"})
            loaded, metadata = GaussianHSMM.load(path)
            np.testing.assert_array_equal(model.predict(sequence), loaded.predict(sequence))
            self.assertAlmostEqual(model.score([sequence]), loaded.score([sequence]), places=12)
            self.assertEqual(metadata["purpose"], "hsmm-test")
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_malformed_embedding_is_rejected(self):
        values = np.zeros((3, 2), dtype=np.float32)
        values[1, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            make_trial(values=values)


if __name__ == "__main__":
    unittest.main()
