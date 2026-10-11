import shutil
import unittest
from pathlib import Path

import numpy as np

from social_gnn.data import SocialTrialPackage
from social_gnn.latent_node_converter import (
    LatentAnimalSource,
    LatentNodeConversionError,
    convert_aligned_latents_to_node_npz,
)


class LatentNodeConverterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path("social_gnn") / "_test_latent_node_converter"
        if cls.output_dir.exists():
            shutil.rmtree(cls.output_dir)
        cls.output_dir.mkdir(parents=True)

    @classmethod
    def tearDownClass(cls):
        if cls.output_dir.exists():
            shutil.rmtree(cls.output_dir)

    def _write_latent(
        self,
        name: str,
        values: np.ndarray,
        *,
        time: np.ndarray | None = None,
        interp_mask: np.ndarray | None = None,
    ) -> Path:
        path = self.output_dir / f"{name}.npz"
        sample_count = len(values)
        np.savez_compressed(
            path,
            downstream=np.asarray(values, dtype=np.float32),
            time=(
                np.asarray(time, dtype=np.float64)
                if time is not None
                else 0.96 + np.arange(sample_count, dtype=np.float64) * 0.04
            ),
            interp_mask=(
                np.asarray(interp_mask, dtype=np.int64)
                if interp_mask is not None
                else np.zeros(sample_count, dtype=np.int64)
            ),
        )
        return path

    def test_four_25hz_latents_are_weighted_into_each_point16_patch(self):
        values_a = np.arange(8, dtype=np.float32)[:, None]
        values_b = values_a + 100.0
        path_a = self._write_latent(
            "weighted_A", values_a, interp_mask=np.asarray([0, 1, 0, 0, 0, 0, 0, 0])
        )
        path_b = self._write_latent("weighted_B", values_b)
        output = self.output_dir / "weighted_nodes.npz"

        summary = convert_aligned_latents_to_node_npz(
            [LatentAnimalSource("A", path_a), LatentAnimalSource("B", path_b)],
            output,
            trial_id="weighted",
            output_identity=[1, 2],
            interpolated_weight=0.0,
            minimum_weight_coverage=0.5,
        )

        self.assertEqual(summary["shape"], [2, 2, 1])
        self.assertEqual(summary["aggregation"]["expected_samples_per_patch"], 4)
        with np.load(output, allow_pickle=False) as payload:
            np.testing.assert_allclose(payload["patch_start_s"], [0.96, 1.12])
            np.testing.assert_allclose(payload["patch_end_s"], [1.12, 1.28])
            # A's interpolated value 1 is excluded: mean([0, 2, 3]) = 5/3.
            self.assertAlmostEqual(
                float(payload["node_features"][0, 0, 0]), 5.0 / 3.0, places=6
            )
            self.assertAlmostEqual(
                float(payload["node_features"][0, 1, 0]), 101.5, places=6
            )
            np.testing.assert_allclose(payload["node_weight_coverage"][:, 0], [0.75, 1.0])
            self.assertEqual(payload["node_mask"].tolist(), [[True, True], [True, True]])
            self.assertEqual(payload["identity"].tolist(), [1, 2])

    def test_insufficient_coverage_sets_mask_and_zero_without_dropping_patch(self):
        values = np.arange(8, dtype=np.float32)[:, None]
        path_a = self._write_latent(
            "coverage_A", values, interp_mask=np.asarray([1, 1, 1, 0, 0, 0, 0, 0])
        )
        path_b = self._write_latent("coverage_B", values + 50.0)
        output = self.output_dir / "coverage_nodes.npz"
        convert_aligned_latents_to_node_npz(
            [("A", path_a), ("B", path_b)],
            output,
            trial_id="coverage",
            minimum_weight_coverage=0.5,
        )
        with np.load(output, allow_pickle=False) as payload:
            self.assertFalse(bool(payload["node_mask"][0, 0]))
            self.assertEqual(float(payload["node_features"][0, 0, 0]), 0.0)
            self.assertTrue(bool(payload["node_mask"][1, 0]))
            self.assertEqual(tuple(payload["node_features"].shape), (2, 2, 1))

    def test_fractional_interpolation_weight_is_used_in_mean_and_coverage(self):
        values = np.asarray([[0.0], [10.0], [20.0], [30.0]])
        path_a = self._write_latent(
            "fraction_A", values, interp_mask=np.asarray([0, 1, 0, 0])
        )
        path_b = self._write_latent("fraction_B", values + 100.0)
        output = self.output_dir / "fraction_nodes.npz"
        convert_aligned_latents_to_node_npz(
            [("A", path_a), ("B", path_b)],
            output,
            trial_id="fraction",
            interpolated_weight=0.25,
        )
        with np.load(output, allow_pickle=False) as payload:
            expected = (0.0 + 0.25 * 10.0 + 20.0 + 30.0) / 3.25
            self.assertAlmostEqual(
                float(payload["node_features"][0, 0, 0]), expected, places=5
            )
            self.assertAlmostEqual(
                float(payload["node_weight_coverage"][0, 0]), 3.25 / 4.0
            )

    def test_timestamp_mismatch_fails_instead_of_silent_alignment(self):
        values = np.arange(8, dtype=np.float32)[:, None]
        path_a = self._write_latent("mismatch_A", values)
        wrong_time = 0.961 + np.arange(8, dtype=np.float64) * 0.04
        path_b = self._write_latent("mismatch_B", values + 10.0, time=wrong_time)
        with self.assertRaisesRegex(
            LatentNodeConversionError, "misaligned|same sampled boundary overlap"
        ):
            convert_aligned_latents_to_node_npz(
                [("A", path_a), ("B", path_b)],
                self.output_dir / "mismatch_nodes.npz",
                trial_id="mismatch",
            )

    def test_tiny_float_offset_and_boundary_only_extra_samples_are_aligned(self):
        base_time = 0.96 + np.arange(9, dtype=np.float64) * 0.04
        values_a = np.arange(9, dtype=np.float32)[:, None]
        values_b = np.arange(8, dtype=np.float32)[:, None] + 100.0
        path_a = self._write_latent("trim_A", values_a, time=base_time)
        path_b = self._write_latent(
            "trim_B", values_b, time=base_time[:8] + 2.4e-7
        )
        output = self.output_dir / "trim_nodes.npz"
        summary = convert_aligned_latents_to_node_npz(
            [("A", path_a), ("B", path_b)],
            output,
            trial_id="trim",
        )
        self.assertEqual(summary["shape"], [2, 2, 1])
        self.assertEqual(
            summary["time_alignment"]["boundary_samples_dropped_suffix"], [1, 0]
        )
        self.assertEqual(
            summary["time_alignment"]["samples_after_last_complete_patch"], 0
        )
        self.assertLess(
            summary["time_alignment"]["maximum_timestamp_spread_s"], 1e-6
        )

    def test_output_is_compatible_with_social_trial_package(self):
        values = np.arange(16, dtype=np.float32).reshape(8, 2)
        path_a = self._write_latent("package_A", values)
        path_b = self._write_latent("package_B", values + 100.0)
        node_path = self.output_dir / "package_nodes.npz"
        convert_aligned_latents_to_node_npz(
            [("A", path_a), ("B", path_b)],
            node_path,
            trial_id="package",
            output_identity=[1, 2],
        )
        with np.load(node_path, allow_pickle=False) as node:
            edge_path = self.output_dir / "package_edges.npz"
            np.savez_compressed(
                edge_path,
                identity=node["identity"],
                patch_start_s=node["patch_start_s"],
                patch_end_s=node["patch_end_s"],
                edge_values=np.zeros((2, 2, 2, 8), dtype=np.float32),
                edge_confidence=np.ones((2, 2, 2, 8), dtype=np.float32),
            )
        package = SocialTrialPackage.from_npz(node_path, edge_path)
        self.assertEqual(package.node_features.shape, (2, 2, 2))
        self.assertEqual(package.edge_values.shape, (2, 2, 2, 8))


if __name__ == "__main__":
    unittest.main()
