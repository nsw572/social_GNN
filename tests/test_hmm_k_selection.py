import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from social_gnn.downstream.embeddings import FrozenEmbeddingTrial
from social_gnn.downstream.gaussian_hmm import GaussianHMM
from social_gnn.downstream.k_selection import (
    FIXED_PRIOR_STRENGTH,
    _mode_lower,
    _partition_pair,
    _summarize_pairs,
    _validate_model_contract,
    audit_run_grid,
    hmm_parameter_count,
    information_criteria,
    likelihood_per_point,
    occupancy_metrics,
    prior_strength_for_k,
    rectangular_contingency,
    rectangular_overlap_summary,
)


def _assignment(path: Path, labels: np.ndarray, valid: np.ndarray | None = None) -> Path:
    labels = np.asarray(labels, dtype=np.int64)
    if valid is None:
        valid = np.ones(labels.size, dtype=bool)
    start = np.arange(labels.size, dtype=float) * 0.04
    np.savez_compressed(
        path,
        patch_start_s=start,
        patch_end_s=start + 0.04,
        valid_time_mask=np.asarray(valid, dtype=bool),
        predicted_state=labels,
    )
    return path


class HMMKSelectionTest(unittest.TestCase):
    def test_parameter_count_and_information_criteria(self):
        self.assertEqual(hmm_parameter_count(6, 16), 6**2 + 32 * 6 - 1)
        result = information_criteria(-100.0, k=6, dimension=16, observations=1000)
        self.assertEqual(result["parameter_count"], 227)
        self.assertAlmostEqual(result["aic_train"], 654.0)
        self.assertAlmostEqual(result["bic_train"], 200.0 + np.log(1000) * 227)

    def test_likelihood_normalization_uses_actual_valid_count(self):
        self.assertAlmostEqual(likelihood_per_point(-125.0, 50), -2.5)
        with self.assertRaises(ValueError):
            likelihood_per_point(-1.0, 0)

    def test_rectangular_contingency_and_information_metrics(self):
        left = np.asarray([0, 0, 1, 1, 1, 0])
        right = np.asarray([0, 2, 1, 1, 2, 0])
        matrix = rectangular_contingency(left, right, 2, 3)
        self.assertEqual(matrix.shape, (2, 3))
        self.assertEqual(int(matrix.sum()), left.size)
        metrics = rectangular_overlap_summary(matrix)
        self.assertGreaterEqual(metrics["variation_of_information"], 0.0)
        self.assertLessEqual(metrics["left_to_right_purity"], 1.0)

    def test_cross_k_split_is_not_forced_into_bijection(self):
        directory = Path("social_gnn") / f"_test_k_selection_{uuid.uuid4().hex}"
        directory.mkdir(parents=True)
        try:
            left = _assignment(directory / "left.npz", np.asarray([0] * 10 + [1] * 10))
            right = _assignment(
                directory / "right.npz", np.asarray([0] * 5 + [2] * 5 + [1] * 10)
            )
            metrics, matrix, matches, diagnostic = _partition_pair(left, right, 2, 3)
            self.assertEqual(matrix.shape, (2, 3))
            self.assertEqual(len(matches), 2)
            split = diagnostic.query(
                "direction == 'left_to_right' and source_state == 0"
            ).iloc[0]
            self.assertTrue(split.split_candidate)
            self.assertEqual(metrics["left_split_candidate_count"], 1)
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_invalid_gap_is_excluded_from_exact_boundary_count(self):
        directory = Path("social_gnn") / f"_test_k_selection_{uuid.uuid4().hex}"
        directory.mkdir(parents=True)
        try:
            valid = np.asarray([True, True, False, True, True])
            left = _assignment(directory / "left.npz", [0, 0, -1, 1, 1], valid)
            right = _assignment(directory / "right.npz", [1, 1, -1, 0, 0], valid)
            metrics, *_ = _partition_pair(left, right, 2, 2)
            self.assertEqual(metrics["eligible_adjacent_patch_pairs"], 2)
            self.assertEqual(metrics["left_boundary_count"], 0)
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_occupancy_and_effective_state_warnings(self):
        labels = np.asarray([0] * 98 + [1] + [2])
        posterior = np.eye(3)[labels]
        summary, rows = occupancy_metrics(labels, posterior, 3)
        self.assertEqual(summary["occupied_states"], 3)
        self.assertLess(summary["effective_state_ratio"], 0.5)
        self.assertIn("dominant_state", summary["occupancy_warning"])
        self.assertEqual(len(rows), 3)

    def test_pairwise_group_aggregation_reports_mean_min_and_median(self):
        frame = pd.DataFrame(
            {
                "architecture": ["a"] * 3,
                "K": [6] * 3,
                "ARI": [0.2, 0.5, 0.8],
                "NMI": [0.3, 0.6, 0.9],
                "mean_hungarian_f1": [0.4, 0.7, 1.0],
                "boundary_f1_exact": [0.1, 0.2, 0.3],
            }
        )
        result = _summarize_pairs(frame, ["architecture", "K"]).iloc[0]
        self.assertAlmostEqual(result.ARI_mean, 0.5)
        self.assertAlmostEqual(result.ARI_median, 0.5)
        self.assertAlmostEqual(result.ARI_min, 0.2)

    def test_prior_policy_is_k_aware_and_k6_identical(self):
        self.assertEqual(prior_strength_for_k("fixed", 12), FIXED_PRIOR_STRENGTH)
        self.assertEqual(
            prior_strength_for_k("normalized_6_over_k", 6), FIXED_PRIOR_STRENGTH
        )
        self.assertEqual(prior_strength_for_k("normalized_6_over_k", 12), 25000.0)

    def test_tie_breaking_is_deterministic_and_prefers_lower_k(self):
        self.assertEqual(_mode_lower([4, 4, 6, 6]), 4)

    def test_expected_transition_counts_do_not_cross_sequences(self):
        sequences = [np.asarray([[0.0], [0.1]]), np.asarray([[4.0], [4.1], [4.2]])]
        model = GaussianHMM(2, sticky_prior_strength=0.0, seed=3).fit(
            sequences, max_iter=3
        )
        self.assertAlmostEqual(model.expected_transition_counts(sequences).sum(), 3.0)

    def test_frozen_trial_splits_invalid_runs_without_padding(self):
        valid = np.asarray([True, True, False, True, True])
        trial = FrozenEmbeddingTrial(
            trial_id="trial",
            graph_embeddings=np.zeros((5, 2), dtype=np.float32),
            patch_start_s=np.arange(5) * 0.04,
            patch_end_s=np.arange(5) * 0.04 + 0.04,
            valid_time_mask=valid,
            encoder_checkpoint="frozen.pt",
            encoder_config="frozen.yaml",
        )
        self.assertEqual([run.tolist() for run in trial.contiguous_valid_runs()], [[0, 1], [3, 4]])

    def test_grid_audit_reports_missing_and_nonconverged_cells(self):
        runs = pd.DataFrame(
            [
                {
                    "architecture": "graph_full",
                    "encoder_seed": 1,
                    "hmm_seed": 0,
                    "K": 4,
                    "status": "completed",
                    "converged": False,
                    "failure": "",
                }
            ]
        )
        audit, counts = audit_run_grid(
            runs,
            architectures=["graph_full"],
            encoder_seeds=[1],
            hmm_seeds=[0],
            k_values=[4, 6],
        )
        self.assertEqual(len(audit), 2)
        self.assertEqual(counts["missing"], 1)
        self.assertEqual(counts["nonconverged_completed"], 1)

    def test_checkpoint_contract_locks_frozen_embedding_and_pca_paths(self):
        model = GaussianHMM(
            6,
            timestep_s=0.04,
            sticky_expected_duration_s=2.0,
            sticky_prior_strength=50000.0,
            seed=0,
        )
        model.initial_ = np.ones(6) / 6
        model.transition_ = np.ones((6, 6)) / 6
        model.means_ = np.zeros((6, 16))
        model.variances_ = np.ones((6, 16))
        embedding = Path("frozen_embedding").resolve()
        pca = Path("frozen_pca.npz").resolve()
        metadata = {
            "encoder": "graph_full",
            "encoder_seed": 1,
            "embedding_dir": str(embedding),
            "preprocessor": str(pca),
        }
        _validate_model_contract(
            model,
            metadata,
            architecture="graph_full",
            encoder_seed=1,
            hmm_seed=0,
            k=6,
            strength=50000.0,
            embedding_dir=embedding,
            preprocessor_path=pca,
        )
        with self.assertRaisesRegex(ValueError, "PCA path mismatch"):
            _validate_model_contract(
                model,
                metadata,
                architecture="graph_full",
                encoder_seed=1,
                hmm_seed=0,
                k=6,
                strength=50000.0,
                embedding_dir=embedding,
                preprocessor_path=Path("other_pca.npz").resolve(),
            )


if __name__ == "__main__":
    unittest.main()
