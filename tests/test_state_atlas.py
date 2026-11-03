import unittest

import numpy as np
import pandas as pd

from social_gnn.downstream.state_atlas import (
    DYNAMIC_TRAITS,
    POSE_TRAITS,
    compute_dynamic_diagnostics,
    compute_quality_channels,
)
from social_gnn.downstream.state_atlas_video import plan_state_clips


class StateAtlasQualityTests(unittest.TestCase):
    def test_quality_uses_confidence_and_coverage_but_not_value_magnitude(self):
        confidence = np.ones((2, 2, 2, 8), dtype=np.float64)
        coverage = np.ones_like(confidence)
        values = np.zeros_like(confidence)
        values[0, 0, 1, list(DYNAMIC_TRAITS)] = [1.0, 2.0, 3.0]
        values[1, 0, 1, list(DYNAMIC_TRAITS)] = [1000.0, -2000.0, 3000.0]
        support, dynamic, pose = compute_quality_channels(
            confidence, coverage, edge_values=values, valid_time_mask=np.ones(2, bool)
        )
        self.assertAlmostEqual(dynamic[0, 0, 1], 1.0)
        self.assertAlmostEqual(dynamic[1, 0, 1], 1.0)
        self.assertAlmostEqual(pose[0, 0, 1], 1.0)
        self.assertTrue(np.isnan(dynamic[0, 0, 0]))
        self.assertEqual(support[0, 0, 0].sum(), 0.0)

    def test_missing_pose_trait_reduces_pose_quality(self):
        confidence = np.ones((1, 2, 2, 8), dtype=np.float64)
        coverage = np.ones_like(confidence)
        confidence[0, 0, 1, POSE_TRAITS[0]] = 0.0
        _support, _dynamic, pose = compute_quality_channels(confidence, coverage)
        self.assertAlmostEqual(pose[0, 0, 1], 0.75)

    def test_support_is_geometric_combination(self):
        confidence = np.ones((1, 2, 2, 8), dtype=np.float64)
        coverage = np.ones_like(confidence)
        confidence[0, 0, 1, 5] = 0.25
        coverage[0, 0, 1, 5] = 0.36
        support, dynamic, _pose = compute_quality_channels(confidence, coverage)
        self.assertAlmostEqual(support[0, 0, 1, 5], 0.3)
        self.assertAlmostEqual(dynamic[0, 0, 1], (0.3 + 1.0 + 1.0) / 3.0)

    def test_dynamic_diagnostics_do_not_bridge_invalid_gap(self):
        values = np.zeros((4, 2, 2, 8), dtype=np.float64)
        valid = np.ones_like(values, dtype=bool)
        values[:, 0, 1, 5] = [1.0, -1.0, 4.0, -4.0]
        time_valid = np.asarray([True, True, False, True])
        flip, jump = compute_dynamic_diagnostics(
            values,
            valid,
            np.asarray([0.02, 0.06, 0.10, 0.14]),
            time_valid,
        )
        self.assertEqual(flip[1, 0, 1, 0], 1.0)
        self.assertAlmostEqual(jump[1, 0, 1, 0], 2.0)
        self.assertTrue(np.isnan(flip[2, 0, 1, 0]))
        self.assertTrue(np.isnan(flip[3, 0, 1, 0]))


class StateAtlasClipTests(unittest.TestCase):
    def _bouts(self):
        rows = []
        for state in range(2):
            for trial_index, trial in enumerate(("A", "B")):
                for bout in range(4):
                    start = state * 20 + trial_index * 8 + bout * 1.5
                    rows.append(
                        {
                            "trial_id": trial,
                            "bout_id": state * 100 + trial_index * 10 + bout,
                            "state": state,
                            "bout_start_s": start,
                            "bout_end_s": start + 0.2 + 0.2 * bout,
                            "bout_duration_s": 0.2 + 0.2 * bout,
                            "mean_state_confidence": 0.7 + 0.05 * bout,
                            "median_dynamic_quality": 0.6 + 0.05 * bout,
                            "median_pose_quality": 0.65 + 0.05 * bout,
                        }
                    )
        return pd.DataFrame(rows)

    def _boundaries(self):
        return pd.DataFrame(
            {
                "trial_id": ["A", "B"],
                "boundary_index": [0, 1],
                "boundary_time_s": [5.0, 10.0],
                "from_state": [0, 1],
                "to_state": [1, 0],
                "boundary_state_confidence": [0.9, 0.8],
                "boundary_dynamic_quality": [0.8, 0.7],
                "boundary_pose_quality": [0.9, 0.8],
            }
        )

    def _videos(self):
        return pd.DataFrame(
            {
                "trial_id": ["A", "B"],
                "source_video": ["A.mp4", "B.mp4"],
                "video_time_offset_s": [0.0, 0.0],
                "nominal_frame_rate": [29.97, 29.97],
                "stream_start_time_s": [0.0, 0.0],
                "video_duration_s": [100.0, 100.0],
                "source_available": [False, False],
            }
        )

    def test_clip_plan_is_deterministic_and_spreads_trials(self):
        left = plan_state_clips(
            bouts=self._bouts(),
            boundaries=self._boundaries(),
            video_metadata=self._videos(),
            n_states=2,
            clips_per_category=2,
            random_seed=7,
        )
        right = plan_state_clips(
            bouts=self._bouts(),
            boundaries=self._boundaries(),
            video_metadata=self._videos(),
            n_states=2,
            clips_per_category=2,
            random_seed=7,
        )
        pd.testing.assert_frame_equal(left, right)
        for (_state, _category), rows in left.groupby(["state", "category"]):
            self.assertEqual(rows.trial_id.nunique(), min(2, len(rows)))
        self.assertTrue((left.video_frame_end_reference >= left.video_frame_start_reference).all())
        self.assertTrue((left.render_status == "source_unavailable").all())


if __name__ == "__main__":
    unittest.main()
