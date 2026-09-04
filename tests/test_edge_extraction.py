import unittest
from pathlib import Path

import numpy as np

from social_gnn.edge_extraction import (
    TRAIT_NAMES,
    EdgeExtractionResult,
    FrameInputs,
    aggregate_fixed_patches,
    build_fixed_social_clock,
    extract_social_edges,
    write_edge_outputs,
)


class EdgeExtractionTest(unittest.TestCase):
    @staticmethod
    def _two_mouse_inputs() -> FrameInputs:
        time = np.arange(4, dtype=float)
        positions = np.zeros((4, 2, 2), dtype=float)
        positions[:, 0, 0] = [0, 1, 2, 3]
        positions[:, 1, 0] = [10, 9, 8, 7]
        velocities = np.full_like(positions, np.nan)
        velocities[1:, 0, 0] = 1.0
        velocities[1:, 0, 1] = 0.0
        velocities[1:, 1, 0] = -1.0
        velocities[1:, 1, 1] = 0.0
        velocity_confidence = np.zeros((4, 2), dtype=float)
        velocity_confidence[1:] = 1.0
        noses = positions.copy()
        noses[:, 0, 0] += 1.0
        noses[:, 1, 0] -= 1.0
        directions = np.zeros_like(positions)
        directions[:, 0, 0] = 1.0
        directions[:, 1, 0] = -1.0
        return FrameInputs(
            frame_ids=np.arange(4, dtype=np.int64),
            frame_time_s=time,
            identities=np.asarray([1, 2], dtype=np.int64),
            position_px=positions,
            position_confidence=np.ones((4, 2), dtype=float),
            velocity_px_s=velocities,
            velocity_confidence=velocity_confidence,
            nose_px=noses,
            nose_confidence=np.ones((4, 2), dtype=float),
            head_direction=directions,
            head_direction_confidence=np.ones((4, 2), dtype=float),
        )

    def test_directed_frame_traits_and_fixed_patch_values(self):
        result = extract_social_edges(
            self._two_mouse_inputs(),
            patch_length_s=2.0,
            patch_count=2,
            minimum_movement_speed_px_s=0.1,
        )
        self.assertEqual(tuple(result.edge_value.shape), (2, 2, 8))
        np.testing.assert_array_equal(result.edge_identity, [[1, 2], [2, 1]])

        edge_1_to_2 = result.edge_value[0, 0]
        np.testing.assert_allclose(
            edge_1_to_2,
            [9.0, 7.0, 8.0, 1.0, -1.0, 1.0, 2.0, -1.0],
        )
        self.assertAlmostEqual(result.edge_value[0, 1, 2], 8.0)
        np.testing.assert_allclose(result.edge_confidence[0, 0, :5], 1.0)
        np.testing.assert_allclose(result.edge_coverage[0, 0, :5], 1.0)
        np.testing.assert_allclose(result.edge_confidence[0, 0, 5:], 0.5)
        np.testing.assert_allclose(result.edge_coverage[0, 0, 5:], 0.5)

    def test_confidence_weighted_mean_is_separate_from_coverage(self):
        times = np.arange(4, dtype=float)
        values = np.zeros((4, 1, 8), dtype=float)
        confidence = np.zeros_like(values)
        values[:2, 0, 0] = [0.0, 10.0]
        confidence[:2, 0, 0] = [1.0, 0.25]
        values[:2, 0, 5] = [100.0, 4.0]
        confidence[:2, 0, 5] = 1.0

        counts, patch_value, patch_conf, coverage, valid_counts = (
            aggregate_fixed_patches(
                frame_time_s=times,
                frame_edge_value=values,
                frame_edge_confidence=confidence,
                patch_start_s=np.asarray([0.0]),
                patch_end_s=np.asarray([2.0]),
            )
        )
        self.assertEqual(int(counts[0]), 2)
        self.assertAlmostEqual(float(patch_value[0, 0, 0]), 2.0)
        self.assertAlmostEqual(float(patch_conf[0, 0, 0]), 0.625)
        self.assertAlmostEqual(float(coverage[0, 0, 0]), 1.0)
        self.assertEqual(int(valid_counts[0, 0, 0]), 2)
        self.assertAlmostEqual(float(patch_value[0, 0, 5]), 4.0)
        self.assertAlmostEqual(float(patch_conf[0, 0, 5]), 0.5)
        self.assertAlmostEqual(float(coverage[0, 0, 5]), 0.5)

    def test_clock_drops_partial_tail_unless_count_is_explicit(self):
        steps, starts, ends = build_fixed_social_clock(
            np.asarray([0.0, 1.0, 2.0, 3.0, 4.0]),
            patch_length_s=2.0,
        )
        np.testing.assert_array_equal(steps, [0, 1])
        np.testing.assert_allclose(starts, [0.0, 2.0])
        np.testing.assert_allclose(ends, [2.0, 4.0])
        explicit, _, explicit_ends = build_fixed_social_clock(
            np.asarray([0.0, 1.0]), patch_length_s=2.0, patch_count=3
        )
        np.testing.assert_array_equal(explicit, [0, 1, 2])
        self.assertEqual(float(explicit_ends[-1]), 6.0)

    def test_npz_keeps_value_confidence_and_coverage_separate(self):
        result = extract_social_edges(
            self._two_mouse_inputs(), patch_length_s=2.0, patch_count=2
        )
        directory = Path.cwd() / "social_gnn" / "_test_edge_outputs"
        directory.mkdir(exist_ok=True)
        try:
            summary = write_edge_outputs(
                result,
                output_dir=directory,
                output_stem="sample",
                idtracker_csv="idtracker.csv",
                matched_mousegpt_csv="mousegpt.csv",
            )
            with np.load(summary["outputs"]["npz"], allow_pickle=False) as payload:
                self.assertEqual(tuple(payload["edge_value"].shape), (2, 2, 8))
                self.assertEqual(
                    tuple(payload["edge_confidence_dense"].shape), (2, 2, 2, 8)
                )
                self.assertIn("edge_coverage", payload.files)
                self.assertNotIn("confidence_weighted_edge_value", payload.files)
                self.assertEqual(payload["trait_names"].tolist(), list(TRAIT_NAMES))
        finally:
            for path in directory.glob("sample_social_edges*"):
                path.unlink(missing_ok=True)
            directory.rmdir()


if __name__ == "__main__":
    unittest.main()
