import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from social_gnn.downstream.gaussian_hmm import GaussianHMM
from social_gnn.downstream.sticky_sensitivity import (
    BASELINE_DWELL_S,
    BASELINE_STRENGTH,
    N_STATES,
    TIMESTEP_S,
    _add_likelihood_deltas,
    _partition_comparison,
    _validate_model_contract,
    sensitivity_grid,
)


def assignment(labels: np.ndarray, valid: np.ndarray | None = None) -> dict[str, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    if valid is None:
        valid = np.ones(labels.size, dtype=bool)
    start = np.arange(labels.size, dtype=float) * TIMESTEP_S
    return {
        "predicted_state": labels,
        "valid_time_mask": np.asarray(valid, dtype=bool),
        "patch_start_s": start,
        "patch_end_s": start + TIMESTEP_S,
    }


class StickySensitivityTest(unittest.TestCase):
    def test_unique_factorial_grid_contains_ten_settings(self):
        grid = sensitivity_grid()
        self.assertEqual(len(grid), 10)
        self.assertEqual(len(set(grid)), 10)
        self.assertIn((BASELINE_STRENGTH, BASELINE_DWELL_S), grid)

    def test_expected_transition_counts_do_not_bridge_sequences(self):
        sequences = [
            np.asarray([[0.0], [0.1], [0.2]], dtype=float),
            np.asarray([[4.0], [4.1]], dtype=float),
        ]
        model = GaussianHMM(
            2,
            sticky_expected_duration_s=None,
            sticky_prior_strength=0.0,
            seed=3,
        ).fit(sequences, max_iter=4)
        counts = model.expected_transition_counts(sequences)
        self.assertAlmostEqual(float(counts.sum()), 3.0, places=6)
        self.assertTrue(np.isfinite(counts).all())

    def test_prior_strength_changes_learned_transition_probabilities(self):
        rng = np.random.default_rng(2)
        sequence = rng.normal(size=(160, 2))
        weak = GaussianHMM(
            3,
            timestep_s=0.04,
            sticky_expected_duration_s=2.0,
            sticky_prior_strength=0.0,
            seed=4,
        ).fit([sequence], max_iter=6)
        strong = GaussianHMM(
            3,
            timestep_s=0.04,
            sticky_expected_duration_s=2.0,
            sticky_prior_strength=10000.0,
            seed=4,
        ).fit([sequence], max_iter=6)
        self.assertGreater(
            float(np.diag(strong.transition_).mean()),
            float(np.diag(weak.transition_).mean()),
        )
        np.testing.assert_allclose(strong.transition_.sum(axis=1), 1.0)

    def test_partition_comparison_is_label_permutation_invariant(self):
        left = assignment(np.asarray([0, 0, 1, 1, 2, 2]))
        right = assignment(np.asarray([2, 2, 0, 0, 1, 1]))
        result, _overlap, matches, diagnostic = _partition_comparison(
            left, right, k=3
        )
        self.assertAlmostEqual(result["ARI"], 1.0)
        self.assertAlmostEqual(result["mean_hungarian_f1"], 1.0)
        self.assertAlmostEqual(result["boundary_f1_exact"], 1.0)
        self.assertEqual(int(diagnostic.split_candidate.sum()), 0)
        self.assertTrue((matches.f1 == 1.0).all())

    def test_partition_comparison_flags_known_split(self):
        left = assignment(np.asarray([0] * 10 + [1] * 10))
        right = assignment(np.asarray([0] * 5 + [2] * 5 + [1] * 10))
        result, _overlap, _matches, diagnostic = _partition_comparison(
            left, right, k=3
        )
        split = diagnostic[
            diagnostic.direction.eq("left_to_right")
            & diagnostic.source_state.eq(0)
        ].iloc[0]
        self.assertTrue(split.split_candidate)
        self.assertLess(result["ARI"], 1.0)

    def test_invalid_gap_is_not_a_boundary(self):
        valid = np.asarray([True, True, False, True, True])
        left = assignment(np.asarray([0, 0, -1, 1, 1]), valid)
        right = assignment(np.asarray([2, 2, -1, 0, 0]), valid)
        result, *_rest = _partition_comparison(left, right, k=3)
        self.assertEqual(result["eligible_adjacent_patch_pairs"], 2)
        self.assertEqual(result["left_boundary_count"], 0)
        self.assertEqual(result["right_boundary_count"], 0)

    def test_likelihood_deltas_are_matched_by_both_seed_ids(self):
        rows = []
        for encoder_seed in (11, 22):
            for downstream_seed in (0, 1):
                baseline = -10.0 - encoder_seed - downstream_seed
                rows.extend(
                    [
                        {
                            "status": "completed",
                            "encoder_seed": encoder_seed,
                            "downstream_seed": downstream_seed,
                            "prior_strength": BASELINE_STRENGTH,
                            "target_dwell_s": BASELINE_DWELL_S,
                            "train_log_likelihood_per_point": baseline - 1,
                            "validation_log_likelihood_per_point": baseline,
                        },
                        {
                            "status": "completed",
                            "encoder_seed": encoder_seed,
                            "downstream_seed": downstream_seed,
                            "prior_strength": 1000.0,
                            "target_dwell_s": 2.0,
                            "train_log_likelihood_per_point": baseline - 0.5,
                            "validation_log_likelihood_per_point": baseline + 0.25,
                        },
                    ]
                )
        result = _add_likelihood_deltas(pd.DataFrame(rows))
        weak = result[result.prior_strength.eq(1000.0)]
        np.testing.assert_allclose(
            weak.validation_log_likelihood_delta_vs_50000_2s, 0.25
        )

    def test_checkpoint_contract_rejects_wrong_pca_or_prior(self):
        embedding = Path("contract_test_embedding").resolve()
        pca = Path("contract_test_pca.npz").resolve()
        model = GaussianHMM(
            N_STATES,
            timestep_s=TIMESTEP_S,
            sticky_expected_duration_s=BASELINE_DWELL_S,
            sticky_prior_strength=BASELINE_STRENGTH,
            seed=0,
        )
        metadata = {
            "encoder_seed": 123,
            "embedding_dir": str(embedding),
            "preprocessor": str(pca),
        }
        _validate_model_contract(
            model,
            metadata,
            encoder_seed=123,
            downstream_seed=0,
            strength=BASELINE_STRENGTH,
            dwell_s=BASELINE_DWELL_S,
            embedding_dir=embedding,
            preprocessor_path=pca,
        )
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            _validate_model_contract(
                model,
                metadata,
                encoder_seed=123,
                downstream_seed=0,
                strength=1000.0,
                dwell_s=BASELINE_DWELL_S,
                embedding_dir=embedding,
                preprocessor_path=pca,
            )


if __name__ == "__main__":
    unittest.main()
