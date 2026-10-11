import unittest

import numpy as np

from social_gnn.imu_video_identity import (
    detect_impulse_event_times,
    estimate_horizontal_acceleration,
    first_consecutive_valid_time,
    match_entry_order,
    score_motion_identity_permutations,
    trajectory_motion_signature,
)


class ImuVideoIdentityTest(unittest.TestCase):
    def test_horizontal_component_removes_static_gravity(self):
        time = np.arange(200, dtype=float) * 0.01
        acceleration = np.zeros((len(time), 3), dtype=float)
        acceleration[:, 2] = 1.0
        acceleration[80:100, 0] += 0.8
        result = estimate_horizontal_acceleration(
            acceleration, time, gravity_window_s=0.5
        )
        self.assertLess(float(np.nanmedian(result.horizontal_magnitude[:50])), 1e-6)
        self.assertGreater(float(np.nanmax(result.horizontal_magnitude)), 0.1)

    def test_motion_assignment_recovers_swapped_video_order(self):
        time = np.arange(100, dtype=float) * 0.04
        signal_a = np.sin(time * 2.0)
        signal_b = np.cos(time * 1.3)
        imu = np.column_stack((signal_a, signal_b))
        video = np.column_stack((signal_b, signal_a))
        evidence = score_motion_identity_permutations(imu, video)
        self.assertEqual(evidence.imu_to_video, (1, 0))
        self.assertGreater(evidence.correlation_matrix[0, 1], 0.9)

    def test_entry_order_maps_identity_and_reports_weak_separation(self):
        strong = match_entry_order(
            np.array([2.0, 8.0]), np.array([1.0, 7.0]), minimum_separation_s=1.0
        )
        self.assertEqual(strong.imu_to_video, (0, 1))
        self.assertEqual(strong.confidence, 1.0)
        weak = match_entry_order(
            np.array([2.0, 2.2]), np.array([1.0, 1.1]), minimum_separation_s=1.0
        )
        self.assertLess(weak.confidence, 1.0)

    def test_trajectory_signature_shape(self):
        time = np.arange(20, dtype=float) * 0.1
        positions = np.column_stack((time**2, np.zeros_like(time)))
        signature = trajectory_motion_signature(positions, time)
        self.assertEqual(signature.shape, (20,))
        self.assertTrue(np.isfinite(signature).all())

    def test_impulse_and_sustained_entry_helpers(self):
        time = np.arange(100, dtype=float) * 0.1
        signal = np.zeros(100, dtype=float)
        signal[20] = 10.0
        signal[80] = 12.0
        events = detect_impulse_event_times(
            signal, time, threshold_z=5.0, refractory_s=1.0, max_events=2
        )
        self.assertEqual(events.tolist(), [20, 80])
        valid = np.zeros(100, dtype=bool)
        valid[30:35] = True
        self.assertAlmostEqual(
            first_consecutive_valid_time(time, valid, consecutive_samples=3), 3.0
        )


if __name__ == "__main__":
    unittest.main()
