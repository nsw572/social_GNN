import csv
import io
import json
import shutil
import unittest
import zipfile
from pathlib import Path

import numpy as np

from social_gnn.authoritative_clock import load_node_clock
from social_gnn.data import SocialTrialPackage
from social_gnn.edge_extraction import aggregate_fixed_patches
from social_gnn.h_zip_node_converter import (
    HZipConversionError,
    convert_h_zip_to_node_npz,
)


class HZipConverterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path("social_gnn") / "_test_h_zip_converter"
        if cls.output_dir.exists():
            shutil.rmtree(cls.output_dir)
        cls.output_dir.mkdir(parents=True)

    @classmethod
    def tearDownClass(cls):
        if cls.output_dir.exists():
            shutil.rmtree(cls.output_dir)

    def _write_h_zip(
        self,
        name: str,
        *,
        duplicate_nodes: bool = False,
        mismatched_clock: bool = False,
    ) -> Path:
        path = self.output_dir / f"{name}.zip"
        latent_a = np.arange(20, dtype=np.float32).reshape(5, 4)
        latent_b = latent_a.copy() if duplicate_nodes else latent_a + 100.0
        latent_buffer = io.BytesIO()
        np.savez_compressed(
            latent_buffer,
            mouse_A_latent=latent_a,
            mouse_B_latent=latent_b,
            codes_A=np.arange(5, dtype=np.int32),
            codes_B=np.arange(5, dtype=np.int32) + 10,
        )
        csv_buffer = io.StringIO(newline="")
        fieldnames = [
            "mouse_id",
            "source_file",
            "window_idx",
            "patch_idx",
            "t_start",
            "t_end",
            "t_center",
            "interp_frac",
            "drop_flag",
            "pair_row",
            "code",
            "pair_id",
        ]
        writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
        writer.writeheader()
        for pair_row in range(5):
            start = pair_row * 0.5
            for identity in ("A", "B"):
                identity_start = (
                    start + 0.01
                    if mismatched_clock and identity == "B" and pair_row == 2
                    else start
                )
                writer.writerow(
                    {
                        "mouse_id": identity,
                        "source_file": f"mouse_{identity}",
                        "window_idx": 0,
                        "patch_idx": pair_row,
                        "t_start": identity_start,
                        "t_end": identity_start + 1.0,
                        "t_center": identity_start + 0.5,
                        "interp_frac": 0.1 if pair_row == 3 else 0.0,
                        "drop_flag": pair_row == 3,
                        "pair_row": pair_row,
                        "code": pair_row if identity == "A" else pair_row + 10,
                        "pair_id": name,
                    }
                )
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            root = name
            archive.writestr(
                f"{root}/meta.json",
                json.dumps({"pair_id": name, "time_unit": "seconds"}),
            )
            archive.writestr(f"{root}/patches.csv", csv_buffer.getvalue())
            archive.writestr(f"{root}/latents.npz", latent_buffer.getvalue())
        return path

    def test_conversion_preserves_overlapping_authoritative_clock(self):
        source = self._write_h_zip("valid")
        output = self.output_dir / "valid_node_features.npz"
        summary = convert_h_zip_to_node_npz(source, output)
        self.assertEqual(summary["shape"], [5, 2, 4])
        self.assertTrue(summary["clock"]["overlapping"])
        self.assertTrue(summary["qc"]["training_eligible"])
        with np.load(output, allow_pickle=False) as payload:
            self.assertEqual(tuple(payload["node_features"].shape), (5, 2, 4))
            np.testing.assert_allclose(payload["patch_start_s"], [0, 0.5, 1, 1.5, 2])
            np.testing.assert_allclose(payload["patch_end_s"], [1, 1.5, 2, 2.5, 3])
            self.assertEqual(payload["node_mask"][:, 0].tolist(), [True, True, True, False, True])
            self.assertEqual(payload["identity"].tolist(), ["A", "B"])

        steps, starts, ends, metadata = load_node_clock(output)
        np.testing.assert_array_equal(steps, np.arange(5))
        np.testing.assert_allclose(starts, [0, 0.5, 1, 1.5, 2])
        np.testing.assert_allclose(ends, [1, 1.5, 2, 2.5, 3])
        self.assertTrue(metadata["overlapping"])

        edge_path = self.output_dir / "valid_social_edges.npz"
        np.savez_compressed(
            edge_path,
            identity=np.asarray(["A", "B"]),
            patch_start_s=starts,
            patch_end_s=ends,
            edge_value_dense=np.zeros((5, 2, 2, 8), dtype=np.float32),
            edge_confidence_dense=np.ones((5, 2, 2, 8), dtype=np.float32),
        )
        package = SocialTrialPackage.from_npz(output, edge_path)
        self.assertEqual(package.node_features.shape, (5, 2, 4))
        self.assertEqual(package.edge_values.shape, (5, 2, 2, 8))

    def test_duplicate_nodes_are_blocked_unless_explicitly_allowed(self):
        source = self._write_h_zip("duplicate", duplicate_nodes=True)
        blocked_output = self.output_dir / "blocked_node_features.npz"
        with self.assertRaisesRegex(
            HZipConversionError, "Trial 'duplicate'.*exact duplicate node streams"
        ):
            convert_h_zip_to_node_npz(source, blocked_output)
        self.assertFalse(blocked_output.exists())

        allowed_output = self.output_dir / "allowed_node_features.npz"
        summary = convert_h_zip_to_node_npz(
            source, allowed_output, allow_duplicate_nodes=True
        )
        self.assertFalse(summary["qc"]["training_eligible"])
        self.assertTrue(summary["qc"]["duplicate_override_used"])
        with np.load(allowed_output, allow_pickle=False) as payload:
            self.assertFalse(bool(payload["training_eligible"]))

    def test_mismatched_mouse_clock_is_rejected_with_pair_row(self):
        source = self._write_h_zip("bad_clock", mismatched_clock=True)
        with self.assertRaisesRegex(
            HZipConversionError, "Trial 'bad_clock'.*t_start.*pair_row 2"
        ):
            convert_h_zip_to_node_npz(
                source, self.output_dir / "bad_clock_node_features.npz"
            )

    def test_overlapping_intervals_aggregate_independently(self):
        times = np.asarray([0.0, 0.5, 1.0, 1.5])
        values = np.zeros((4, 1, 8), dtype=float)
        confidence = np.ones_like(values)
        values[:, 0, 0] = [0.0, 10.0, 20.0, 30.0]
        _, patch_values, _, coverage, _ = aggregate_fixed_patches(
            frame_time_s=times,
            frame_edge_value=values,
            frame_edge_confidence=confidence,
            patch_start_s=np.asarray([0.0, 0.5]),
            patch_end_s=np.asarray([1.0, 1.5]),
        )
        self.assertAlmostEqual(float(patch_values[0, 0, 0]), 5.0)
        self.assertAlmostEqual(float(patch_values[1, 0, 0]), 15.0)
        np.testing.assert_allclose(coverage[:, 0, 0], 1.0)


if __name__ == "__main__":
    unittest.main()
