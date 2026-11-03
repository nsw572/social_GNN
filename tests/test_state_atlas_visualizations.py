import tempfile
import unittest
from pathlib import Path

import numpy as np

from social_gnn.downstream.state_atlas_fingerprint import (
    robust_standardized_medians,
    summarize_state_fingerprints,
)
from social_gnn.downstream.state_atlas_pose_validity import (
    compute_pose_validity_summary,
)
from social_gnn.downstream.state_atlas_timeline import (
    build_timeline_payload,
    write_timeline_html,
)
from social_gnn.downstream.state_atlas_viz_common import AtlasViewTrial


def make_trial(trial_id="T", offset=0.0):
    length = 6
    start = offset + np.arange(length) * 0.04
    values = np.zeros((length, 2, 2, 8), dtype=np.float32)
    confidence = np.ones_like(values)
    coverage = np.ones_like(values)
    edge_valid = np.ones_like(values, dtype=bool)
    diagonal = np.arange(2)
    edge_valid[:, diagonal, diagonal] = False
    values[:, 0, 1, 0] = np.arange(length) + 1
    values[:, 1, 0, 0] = np.arange(length) + 1
    values[:, 0, 1, 2] = np.arange(length) + 10
    values[:, 1, 0, 2] = np.arange(length) + 20
    edge_valid[1, 0, 1, 1] = False
    confidence[1, 0, 1, 1] = 0
    coverage[1, 0, 1, 1] = 0
    valid_time = np.asarray([True, True, True, True, False, True])
    edge_valid[~valid_time] = False
    atlas = np.full((length, 2, 2, 10), np.nan, dtype=np.float32)
    atlas[..., :8] = np.where(edge_valid, values, np.nan)
    dynamic = np.ones((length, 2, 2), dtype=np.float32)
    pose = np.ones((length, 2, 2), dtype=np.float32)
    dynamic[:, diagonal, diagonal] = np.nan
    pose[:, diagonal, diagonal] = np.nan
    pose[1, 0, 1] = 0.5
    atlas[..., 8] = dynamic
    atlas[..., 9] = pose
    sign = np.full((length, 2, 2, 3), np.nan, dtype=np.float32)
    jump = np.full_like(sign, np.nan)
    sign[1:, 0, 1] = 0
    jump[1:, 0, 1] = 1
    return AtlasViewTrial(
        trial_id=trial_id,
        identity=np.asarray([0, 1]),
        patch_start_s=start,
        patch_end_s=start + 0.04,
        valid_time_mask=valid_time,
        predicted_state=np.asarray([0, 0, 1, 1, -1, 0]),
        state_confidence=np.asarray([0.9, 0.8, 0.95, 0.85, 0, 0.7], dtype=np.float32),
        bout_id=np.asarray([0, 0, 1, 1, -1, 2]),
        boundary_adjacent=np.asarray([False, True, True, False, False, False]),
        atlas_edge_tensor=atlas,
        edge_values=values,
        edge_confidence=confidence,
        edge_coverage=coverage,
        edge_valid=edge_valid,
        dynamic_quality=dynamic,
        pose_quality=pose,
        dynamic_sign_flip=sign,
        dynamic_abs_jump=jump,
    )


class TimelineExplorerTests(unittest.TestCase):
    def test_payload_keeps_missingness_and_directions(self):
        payload = build_timeline_payload(make_trial())
        self.assertTrue(np.isnan(payload["numeric"]["nose_to_nose_distance"][1]))
        self.assertTrue(np.isnan(payload["numeric"]["centroid_distance"][4]))
        self.assertEqual(
            payload["numeric"]["source_nose_to_target_centroid_distance_0to1"][0],
            10,
        )
        self.assertEqual(
            payload["numeric"]["source_nose_to_target_centroid_distance_1to0"][0],
            20,
        )
        self.assertEqual(len(payload["boundaries"]), 1)

    def test_html_contains_required_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_timeline_html(
                make_trial(),
                Path(directory) / "T.html",
                trial_files={"T": "T.html"},
                default_window_s=37.5,
                include_plotlyjs="none",
            )
            text = path.read_text(encoding="utf-8")
            for token in (
                "scaleMode",
                "clipEnabled",
                "Pose validity indicators",
                "Dynamic sign-flip / jump diagnostics",
                "Robust z-score",
                '"default_window_s":37.5',
            ):
                self.assertIn(token, text)


class FingerprintTests(unittest.TestCase):
    def test_summary_uses_valid_values_and_quality_support(self):
        summary, distributions = summarize_state_fingerprints(
            [make_trial()], n_states=2
        )
        row = summary[
            summary.state.eq(0) & summary.channel.eq("nose_to_nose_distance")
        ].iloc[0]
        self.assertEqual(row.eligible_count, 3)
        self.assertEqual(row.usable_support_count, 2)
        self.assertAlmostEqual(row.usable_support_fraction, 2 / 3)
        directed = distributions[
            (0, "source_nose_to_target_centroid_distance 1->0")
        ]
        np.testing.assert_array_equal(directed, [20, 21, 25])

    def test_standardization_is_per_channel(self):
        summary, _ = summarize_state_fingerprints([make_trial()], n_states=2)
        standardized = robust_standardized_medians(summary)
        self.assertEqual(standardized.shape[1], 13)
        self.assertTrue(np.isfinite(standardized.loc[0, "centroid_distance"]))


class PoseValidityTests(unittest.TestCase):
    def test_validity_delta_uses_channel_global_baseline(self):
        summary = compute_pose_validity_summary([make_trial()], n_states=2)
        rows = summary[
            summary.trial_scope.eq("ALL")
            & summary.pose_channel.eq("nose_to_nose_distance")
        ].set_index("state")
        self.assertEqual(rows.loc[0, "total_eligible_count"], 3)
        self.assertEqual(rows.loc[0, "valid_count"], 2)
        expected_global = 4 / 5
        self.assertAlmostEqual(rows.loc[0, "global_validity_baseline"], expected_global)
        self.assertAlmostEqual(
            rows.loc[0, "delta_from_global_validity"], 2 / 3 - expected_global
        )
        self.assertTrue(
            {"confidence_on_valid_median", "coverage_on_valid_median"}.issubset(
                summary.columns
            )
        )


if __name__ == "__main__":
    unittest.main()
