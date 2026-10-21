import unittest

import numpy as np
import pandas as pd

from social_gnn.downstream.aggregate import _aggregate_runs


class DownstreamAggregateTest(unittest.TestCase):
    def test_metric_aggregation_keeps_run_counts_and_ignores_missing_values(self):
        runs = pd.DataFrame(
            [
                {
                    "encoder": "encoder",
                    "algorithm": "hmm",
                    "K": 6,
                    "max_duration_patches": np.nan,
                    "encoder_seed": 1,
                    "downstream_seed": 7,
                    "converged": True,
                    "effective_entropy_states": 5.0,
                    "bouts_per_minute": 60.0,
                },
                {
                    "encoder": "encoder",
                    "algorithm": "hmm",
                    "K": 6,
                    "max_duration_patches": np.nan,
                    "encoder_seed": 2,
                    "downstream_seed": 8,
                    "converged": False,
                    "effective_entropy_states": np.nan,
                    "bouts_per_minute": 80.0,
                },
            ]
        )
        summary = _aggregate_runs(runs, pd.DataFrame())
        self.assertEqual(summary.shape[0], 1)
        self.assertEqual(int(summary.loc[0, "runs"]), 2)
        self.assertEqual(int(summary.loc[0, "encoder_seeds"]), 2)
        self.assertEqual(int(summary.loc[0, "downstream_seeds"]), 2)
        self.assertAlmostEqual(summary.loc[0, "converged_fraction"], 0.5)
        self.assertAlmostEqual(summary.loc[0, "effective_entropy_states_mean"], 5.0)
        self.assertAlmostEqual(summary.loc[0, "bouts_per_minute_mean"], 70.0)


if __name__ == "__main__":
    unittest.main()
