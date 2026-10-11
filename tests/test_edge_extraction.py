import unittest
from pathlib import Path

import numpy as np

from social_gnn.edge_extraction import (
    DYNAMIC_TRAIT_INDICES,
    STATE_TRAIT_INDICES,
    TRAIT_NAMES,
    FrameInputs,
    aggregate_fixed_patches,
    extract_social_edges_on_clock,
    write_edge_outputs,
)


class EdgeExtractionTest(unittest.TestCase):
    @staticmethod
    def _two_mouse_inputs(*, separating: bool = False) -> FrameInputs:
        time = np.arange(6, dtype=float)
        positions = np.zeros((6, 2, 2), dtype=float)
        direction = -1.0 if separating else 1.0
        positions[:, 0, 0] = direction * time
        positions[:, 1, 0] = 12.0 - direction * time
        velocities = np.full_like(positions, np.nan)
        velocity_confidence = np.zeros((6, 2), dtype=float)
        noses = positions.copy()
        noses[:, 0, 0] += 1.0
        noses[:, 1, 0] -= 1.0
        head_direction = np.zeros_like(positions)
        head_direction[:, 0, 0] = 1.0
        head_direction[:, 1, 0] = -1.0
        return FrameInputs(
            frame_ids=np.arange(6, dtype=np.int64),
            frame_time_s=time,
            identities=np.asarray([1, 2], dtype=np.int64),
            position_px=positions,
            position_confidence=np.ones((6, 2), dtype=float),
            velocity_px_s=velocities,
            velocity_confidence=velocity_confidence,
            nose_px=noses,
            nose_confidence=np.ones((6, 2), dtype=float),
            head_direction=head_direction,
            head_direction_confidence=np.ones((6, 2), dtype=float),
        )

    @staticmethod
    def _extract(inputs: FrameInputs, *, motion_floor: float = 0.1):
        return extract_social_edges_on_clock(
            inputs,
            social_step=np.arange(3, dtype=np.int64),
            patch_start_s=np.asarray([0.0, 2.0, 4.0]),
            patch_end_s=np.asarray([2.0, 4.0, 6.0]),
            minimum_patch_displacement_px=motion_floor,
        )

    def test_state_traits_use_confidence_weighted_patch_means(self):
        times = np.arange(4, dtype=float)
        values = np.zeros((4, 1, 8), dtype=float)
        confidence = np.zeros_like(values)
        values[:2, 0, 0] = [0.0, 10.0]
        confidence[:2, 0, 0] = [1.0, 0.25]

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

    def test_dynamic_traits_use_patch_differences_and_timestamp_dt(self):
        result = self._extract(self._two_mouse_inputs())
        self.assertEqual(tuple(result.edge_value.shape), (3, 2, 8))
        np.testing.assert_array_equal(result.edge_identity, [[1, 2], [2, 1]])
        np.testing.assert_allclose(result.patch_center_s, [1.0, 3.0, 5.0])
        np.testing.assert_allclose(result.patch_delta_t_s, [2.0, 2.0, 0.0])

        # Patch means for distance are 11, 7, 3 pixels; dt is 2 seconds.
        self.assertAlmostEqual(float(result.edge_value[0, 0, 0]), 11.0)
        self.assertAlmostEqual(float(result.edge_value[0, 0, 5]), 1.0)
        self.assertAlmostEqual(float(result.edge_value[0, 0, 6]), 2.0)
        self.assertAlmostEqual(float(result.edge_value[0, 0, 7]), -1.0)
        self.assertAlmostEqual(float(result.edge_value[0, 1, 5]), 1.0)
        np.testing.assert_allclose(result.edge_confidence[0, 0], 1.0)

        np.testing.assert_allclose(
            result.edge_value[-1, :, DYNAMIC_TRAIT_INDICES], 0.0
        )
        np.testing.assert_allclose(
            result.edge_confidence[-1, :, DYNAMIC_TRAIT_INDICES], 0.0
        )

    def test_approach_and_closing_signs_become_negative_when_separating(self):
        result = self._extract(self._two_mouse_inputs(separating=True))
        self.assertAlmostEqual(float(result.edge_value[0, 0, 5]), -1.0)
        self.assertAlmostEqual(float(result.edge_value[0, 1, 5]), -1.0)
        self.assertAlmostEqual(float(result.edge_value[0, 0, 6]), -2.0)
        self.assertAlmostEqual(float(result.edge_value[0, 1, 6]), -2.0)

    def test_radial_speed_is_directional_while_closing_is_symmetric(self):
        inputs = self._two_mouse_inputs()
        positions = np.array(inputs.position_px, copy=True)
        positions[:, 0, 0] = np.arange(6, dtype=float)
        positions[:, 1, 0] = 12.0
        inputs.position_px = positions
        inputs.nose_px = positions.copy()
        result = self._extract(inputs)
        self.assertAlmostEqual(float(result.edge_value[0, 0, 5]), 1.0)
        self.assertAlmostEqual(float(result.edge_value[0, 1, 5]), 0.0)
        self.assertAlmostEqual(float(result.edge_value[0, 0, 6]), 1.0)
        self.assertAlmostEqual(float(result.edge_value[0, 1, 6]), 1.0)
        self.assertEqual(float(result.edge_confidence[0, 0, 7]), 0.0)
        self.assertEqual(float(result.edge_confidence[0, 1, 7]), 0.0)

    def test_low_patch_motion_downweights_alignment_confidence(self):
        inputs = self._two_mouse_inputs()
        positions = np.array(inputs.position_px, copy=True)
        positions[:2, 0, 0] = 0.0
        positions[:2, 1, 0] = 10.0
        positions[2:4, 0, 0] = 0.25
        positions[2:4, 1, 0] = 10.25
        positions[4:, 0, 0] = 0.5
        positions[4:, 1, 0] = 10.5
        inputs.position_px = positions
        inputs.nose_px = positions.copy()
        result = self._extract(inputs, motion_floor=1.0)
        self.assertAlmostEqual(float(result.edge_value[0, 0, 7]), 1.0)
        self.assertAlmostEqual(float(result.edge_confidence[0, 0, 7]), 0.25)

    def test_npz_keeps_one_matrix_per_patch_and_qc_separate(self):
        result = self._extract(self._two_mouse_inputs())
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
                self.assertEqual(tuple(payload["edge_value_dense"].shape), (3, 2, 2, 8))
                self.assertEqual(
                    tuple(payload["edge_confidence_dense"].shape), (3, 2, 2, 8)
                )
                self.assertEqual(tuple(payload["patch_centroid_px"].shape), (3, 2, 2))
                self.assertIn("edge_coverage", payload.files)
                self.assertNotIn("confidence_weighted_edge_value", payload.files)
                self.assertEqual(payload["trait_names"].tolist(), list(TRAIT_NAMES))
                self.assertEqual(
                    [TRAIT_NAMES[index] for index in STATE_TRAIT_INDICES],
                    summary["aggregation"]["state_traits"],
                )
        finally:
            for path in directory.glob("sample_social_edges*"):
                path.unlink(missing_ok=True)
            directory.rmdir()


if __name__ == "__main__":
    unittest.main()
