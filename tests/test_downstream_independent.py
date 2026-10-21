import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from social_gnn.downstream.evaluation import segmentation_metrics, stability_metrics
from social_gnn.downstream.independent import (
    GaussianMixtureStateModel,
    KMeansStateModel,
)

from test_downstream_temporal import make_trial
from social_gnn.downstream.evaluation import DecodedTrial


def _sequences():
    rng = np.random.default_rng(17)
    return [
        np.r_[rng.normal(-2.0, 0.2, (30, 2)), rng.normal(2.0, 0.2, (30, 2))],
        np.r_[rng.normal(2.0, 0.2, (25, 2)), rng.normal(-2.0, 0.2, (25, 2))],
    ]


class IndependentStateModelTest(unittest.TestCase):
    def test_kmeans_seed_and_checkpoint_are_reproducible(self):
        sequences = _sequences()
        first = KMeansStateModel(2, seed=5).fit(sequences)
        second = KMeansStateModel(2, seed=5).fit(sequences)
        np.testing.assert_array_equal(first.predict(sequences[0]), second.predict(sequences[0]))
        directory = Path("social_gnn") / f"_test_cluster_{uuid.uuid4().hex}"
        directory.mkdir(parents=True)
        try:
            path = first.save(directory / "kmeans.npz", metadata={"trial": "unit"})
            restored, metadata = KMeansStateModel.load(path)
            np.testing.assert_array_equal(
                first.predict(sequences[1]), restored.predict(sequences[1])
            )
            self.assertEqual(metadata["trial"], "unit")
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_gmm_seed_and_checkpoint_are_reproducible(self):
        sequences = _sequences()
        first = GaussianMixtureStateModel(2, seed=7, max_iter=20).fit(sequences)
        second = GaussianMixtureStateModel(2, seed=7, max_iter=20).fit(sequences)
        np.testing.assert_allclose(first.posterior(sequences[0]), second.posterior(sequences[0]))
        directory = Path("social_gnn") / f"_test_cluster_{uuid.uuid4().hex}"
        directory.mkdir(parents=True)
        try:
            path = first.save(directory / "gmm.npz", metadata={"trial": "unit"})
            restored, metadata = GaussianMixtureStateModel.load(path)
            np.testing.assert_allclose(
                first.posterior(sequences[1]), restored.posterior(sequences[1])
            )
            self.assertAlmostEqual(first.score(sequences), restored.score(sequences))
            self.assertEqual(metadata["trial"], "unit")
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_ari_and_stability_ignore_label_names(self):
        first = np.asarray([0, 0, 1, 1, 2, 2])
        renamed = np.asarray([2, 2, 0, 0, 1, 1])
        self.assertAlmostEqual(adjusted_rand_score(first, renamed), 1.0)
        metrics = stability_metrics([first, renamed])
        self.assertAlmostEqual(metrics["mean_pairwise_ari"], 1.0)
        self.assertAlmostEqual(metrics["mean_pairwise_nmi"], 1.0)

    def test_effective_state_count_matches_entropy_definition(self):
        trial = make_trial(values=np.zeros((4, 2)))
        labels = np.asarray([0, 0, 1, 1])
        posterior = np.eye(2)[labels]
        decoded = DecodedTrial(trial, labels, posterior, np.ones(4), (np.arange(4),))
        metrics = segmentation_metrics([decoded], 2)
        self.assertAlmostEqual(metrics["effective_entropy_states"], 2.0)
        self.assertAlmostEqual(metrics["largest_state_occupancy"], 0.5)
        self.assertAlmostEqual(metrics["smallest_nonempty_state_occupancy"], 0.5)


if __name__ == "__main__":
    unittest.main()
