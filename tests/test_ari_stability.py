import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from social_gnn.downstream.ari_stability import (
    ACROSS_BOTH,
    MATCHED_DOWNSTREAM,
    WITHIN_ENCODER,
    AssignmentData,
    align_assignments,
    ari_matrix,
    audit_run_grid,
    enumerate_seed_pairs,
    expected_grid,
    require_complete_grid,
    summarize_groups,
)


class AriStabilityTest(unittest.TestCase):
    def test_pair_grouping_has_exact_9_9_18_counts_without_duplicates(self):
        pairs = enumerate_seed_pairs()
        self.assertEqual(len(pairs), 36)
        counts = pd.Series([value["comparison_group"] for value in pairs]).value_counts()
        self.assertEqual(int(counts[WITHIN_ENCODER]), 9)
        self.assertEqual(int(counts[MATCHED_DOWNSTREAM]), 9)
        self.assertEqual(int(counts[ACROSS_BOTH]), 18)
        canonical = set()
        for value in pairs:
            left = (value["run_1_encoder_seed"], value["run_1_downstream_seed"])
            right = (value["run_2_encoder_seed"], value["run_2_downstream_seed"])
            self.assertNotEqual(left, right)
            canonical.add(tuple(sorted((left, right))))
        self.assertEqual(len(canonical), 36)

    def test_ari_matrix_is_symmetric_with_perfect_diagonal(self):
        rows = enumerate_seed_pairs()
        for index, value in enumerate(rows):
            value["ARI"] = index / 100.0
        matrix, labels = ari_matrix(pd.DataFrame(rows))
        self.assertEqual(matrix.shape, (9, 9))
        self.assertEqual(len(labels), 9)
        np.testing.assert_allclose(matrix, matrix.T)
        np.testing.assert_allclose(np.diag(matrix), 1.0)

    def test_ari_is_invariant_to_label_permutation(self):
        left = np.asarray([0, 0, 1, 1, 2, 2])
        right = np.asarray([2, 2, 0, 0, 1, 1])
        self.assertAlmostEqual(adjusted_rand_score(left, right), 1.0)

    def test_alignment_uses_physical_keys_and_valid_intersection(self):
        left = AssignmentData(
            trial_id="trial",
            patch_start_s=np.asarray([0.0, 0.2, 0.4, 0.6]),
            patch_end_s=np.asarray([0.2, 0.4, 0.6, 0.8]),
            valid_time_mask=np.asarray([True, True, False, True]),
            predicted_state=np.asarray([0, 1, -1, 2]),
        )
        right = AssignmentData(
            trial_id="trial",
            patch_start_s=np.asarray([0.6, 0.0, 0.4, 0.8]),
            patch_end_s=np.asarray([0.8, 0.2, 0.6, 1.0]),
            valid_time_mask=np.asarray([True, True, True, True]),
            predicted_state=np.asarray([5, 3, 4, 6]),
        )
        aligned = align_assignments(left, right)
        np.testing.assert_array_equal(aligned["left_labels"], [0, 2])
        np.testing.assert_array_equal(aligned["right_labels"], [3, 5])
        self.assertEqual(aligned["number_of_aligned_patches"], 2)
        self.assertEqual(aligned["left_only_physical_patches"], 1)
        self.assertEqual(aligned["right_only_physical_patches"], 1)
        self.assertEqual(aligned["validity_disagreements_on_common_patches"], 1)
        self.assertFalse(aligned["identical_full_timeline"])

    def test_missing_run_is_reported_and_rejected(self):
        expected = expected_grid(
            encoders=["e"], algorithms=["hmm"], encoder_seeds=[1, 2, 3], downstream_seeds=[0, 1, 2]
        )
        observed = expected.iloc[:-1].copy()
        audit = audit_run_grid(observed, expected)
        missing = audit[audit["grid_status"] == "missing"]
        self.assertEqual(missing.shape[0], 1)
        with self.assertRaisesRegex(ValueError, "incomplete or ambiguous"):
            require_complete_grid(audit)

    def test_summary_statistics_keep_counts_and_metadata(self):
        frame = pd.DataFrame(
            {
                "encoder_condition": ["e"] * 4,
                "downstream_algorithm": ["hmm"] * 4,
                "comparison_group": [WITHIN_ENCODER] * 4,
                "ARI": [0.1, 0.2, 0.3, 0.4],
            }
        )
        summary = summarize_groups(frame)
        self.assertEqual(summary.loc[0, "encoder_condition"], "e")
        self.assertEqual(summary.loc[0, "downstream_algorithm"], "hmm")
        self.assertEqual(int(summary.loc[0, "pair_count"]), 4)
        self.assertAlmostEqual(float(summary.loc[0, "ari_mean"]), 0.25)
        self.assertAlmostEqual(float(summary.loc[0, "ari_median"]), 0.25)
        self.assertAlmostEqual(float(summary.loc[0, "ari_q25"]), 0.175)
        self.assertAlmostEqual(float(summary.loc[0, "ari_q75"]), 0.325)


if __name__ == "__main__":
    unittest.main()
