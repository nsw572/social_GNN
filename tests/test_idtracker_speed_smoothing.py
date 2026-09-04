import unittest

import numpy as np

from social_gnn.idtracker_identity_smoothing import (
    detect_and_smooth_contact_speed_artifacts,
)


class IdtrackerSpeedSmoothingTest(unittest.TestCase):
    @staticmethod
    def _opposite_motion_with_contact_spike() -> np.ndarray:
        frames = np.arange(200, dtype=float)
        trajectories = np.zeros((200, 2, 2), dtype=float)
        trajectories[:, 0, 0] = frames
        trajectories[:, 1, 0] = 200.0 - frames
        # Both centroids collapse onto one point and then return.
        trajectories[50, :, 0] = 100.0
        return trajectories

    def test_detects_and_linearly_repairs_paired_contact_spike(self):
        trajectories = self._opposite_motion_with_contact_spike()
        result = detect_and_smooth_contact_speed_artifacts(
            trajectories,
            fps=1.0,
            body_length_px=300.0,
            speed_multiplier=5.0,
            contact_distance_body_lengths=0.35,
        )

        self.assertEqual(result.events[0]["start_frame"], 50)
        self.assertEqual(result.events[0]["end_frame"], 51)
        self.assertTrue(result.artifact_mask[50:52].all())
        np.testing.assert_allclose(
            result.smoothed_trajectories[50:52, 0, 0], [50.0, 51.0]
        )
        np.testing.assert_allclose(
            result.smoothed_trajectories[50:52, 1, 0], [150.0, 149.0]
        )
        np.testing.assert_allclose(result.smoothed_speed_px_s[50:52], 1.0)
        np.testing.assert_allclose(result.centroid_confidence[50:52], 0.05)

    def test_rejects_same_direction_global_motion(self):
        frames = np.arange(200, dtype=float)
        trajectories = np.zeros((200, 2, 2), dtype=float)
        trajectories[:, 0, 0] = frames
        trajectories[:, 1, 0] = frames + 100.0
        trajectories[50:, :, 0] += 50.0

        result = detect_and_smooth_contact_speed_artifacts(
            trajectories,
            fps=1.0,
            body_length_px=300.0,
            speed_multiplier=5.0,
            contact_distance_body_lengths=0.35,
        )
        self.assertFalse(result.artifact_seed_mask.any())
        self.assertFalse(result.artifact_mask.any())
        self.assertEqual(result.events, [])

    def test_pairs_two_identity_transitions_and_swaps_the_interval(self):
        frames = np.arange(200, dtype=float)
        physical = np.zeros((200, 2, 2), dtype=float)
        physical[:, 0, 0] = np.where(frames <= 50, frames, np.maximum(0, 100 - frames))
        physical[:, 1, 0] = 100.0 - physical[:, 0, 0]

        raw = physical.copy()
        raw[51:100] = physical[51:100, ::-1]
        result = detect_and_smooth_contact_speed_artifacts(
            raw,
            fps=1.0,
            body_length_px=10.0,
            speed_multiplier=5.0,
            contact_distance_body_lengths=0.35,
            maximum_swap_pair_gap_seconds=120.0,
        )

        self.assertEqual(len(result.swap_pairs), 1)
        pair = result.swap_pairs[0]
        self.assertEqual(pair["entry_contact_minimum_frame"], 50)
        self.assertEqual(pair["exit_speed_start_frame"], 100)
        self.assertTrue(result.identity_swap_mask[52:100].all())
        self.assertFalse(result.identity_swap_mask[:52].any())
        self.assertFalse(result.identity_swap_mask[100:].any())
        np.testing.assert_allclose(result.smoothed_trajectories[52:100], physical[52:100])
        self.assertLessEqual(float(np.nanmax(result.smoothed_speed_px_s)), 1.0)
        np.testing.assert_allclose(result.centroid_confidence[49:52], 0.05)
        np.testing.assert_allclose(result.centroid_confidence[100], 0.05)

    def test_does_not_pair_contact_when_tracks_bounce_without_crossing(self):
        frames = np.arange(200, dtype=float)
        trajectories = np.zeros((200, 2, 2), dtype=float)
        left_track = np.where(frames <= 50, frames, np.maximum(0, 100 - frames))
        trajectories[:, 0, 0] = left_track
        trajectories[:, 1, 0] = 100.0 - left_track
        trajectories[100:, 0, 1] += 50.0

        result = detect_and_smooth_contact_speed_artifacts(
            trajectories,
            fps=1.0,
            body_length_px=10.0,
            speed_multiplier=5.0,
            maximum_swap_pair_gap_seconds=120.0,
        )

        self.assertEqual(result.swap_pairs, [])
        self.assertFalse(result.identity_swap_mask.any())


if __name__ == "__main__":
    unittest.main()
