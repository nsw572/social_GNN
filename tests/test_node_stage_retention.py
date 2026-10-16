import unittest

import numpy as np

from social_gnn.evaluate_node_stage_retention import _metric_rows


class NodeStageRetentionTest(unittest.TestCase):
    def test_metric_groups_follow_nine_dimensional_node_target(self):
        rng = np.random.default_rng(901)
        truth = rng.normal(size=(32, 18))
        rows = _metric_rows(
            model="test",
            representation="test",
            probe="exact",
            truth=truth,
            prediction=truth.copy(),
            feature_names=[f"feature_{index}" for index in range(18)],
        )
        aggregates = {
            row["target_group"]: row
            for row in rows
            if row["target"] == "__aggregate__"
        }
        self.assertEqual(
            set(aggregates), {"node_mean", "node_abs_difference", "combined"}
        )
        self.assertAlmostEqual(aggregates["combined"]["r2"], 1.0)

    def test_metric_groups_reject_odd_summary_width(self):
        truth = np.zeros((4, 17), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "2 \\* node_dim"):
            _metric_rows(
                model="test",
                representation="test",
                probe="test",
                truth=truth,
                prediction=truth,
                feature_names=[f"feature_{index}" for index in range(17)],
            )


if __name__ == "__main__":
    unittest.main()
