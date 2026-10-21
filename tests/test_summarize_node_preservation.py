from __future__ import annotations

import csv
import shutil
import unittest
import uuid
from pathlib import Path

from social_gnn.evaluation import write_csv
from social_gnn.summarize_node_preservation import summarize


class NodePreservationSummaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("social_gnn") / f"_test_summary_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _probe(self, name: str, offset: float) -> Path:
        path = self.root / name
        rows = [
            {
                "model": "candidate",
                "probe": "linear",
                "input_mode": "native",
                "subset": "confidence_weighted_all",
                "split": "test",
                "target_kind": "node_invariant",
                "target": "node_mean_0",
                "base_trait": "",
                "r2": 0.8 + offset,
            },
            {
                "model": "candidate",
                "probe": "linear",
                "input_mode": "native",
                "subset": "confidence_weighted_all",
                "split": "test",
                "target_kind": "node_invariant",
                "target": "node_abs_difference_0",
                "base_trait": "",
                "r2": 0.6 + offset,
            },
            {
                "model": "candidate",
                "probe": "linear",
                "input_mode": "native",
                "subset": "confidence_weighted_all",
                "split": "test",
                "target_kind": "edge",
                "target": "source_faces_target",
                "base_trait": "source_faces_target",
                "r2": 0.4 + offset,
            },
            {
                "model": "candidate",
                "probe": "linear",
                "input_mode": "native",
                "subset": "confidence_weighted_all",
                "split": "test",
                "target_kind": "edge_invariant",
                "target": "edge_summary",
                "base_trait": "",
                "r2": 0.5 + offset,
            },
        ]
        write_csv(path, rows)
        return path

    def _geometry(self, name: str, variance: float, rank: float, error: float) -> Path:
        path = self.root / name
        write_csv(
            path,
            [
                {
                    "model": "candidate",
                    "trial_id": "A4_C4",
                    "parameter_count": 10,
                    "total_variance": variance,
                    "entropy_effective_rank": rank,
                    "mean_permutation_error": error,
                }
            ],
        )
        return path

    def test_probe_and_geometry_are_aggregated_across_seeds(self) -> None:
        probe_a = self._probe("probe_a.csv", 0.0)
        probe_b = self._probe("probe_b.csv", 0.2)
        geometry_a = self._geometry("geometry_a.csv", 2.0, 3.0, 0.0)
        geometry_b = self._geometry("geometry_b.csv", 4.0, 5.0, 0.2)
        output = self.root / "summary"

        summarize(
            probe_specs=[
                f"candidate=candidate={probe_a}",
                f"candidate=candidate={probe_b}",
            ],
            role_models=[],
            geometry_specs=[
                f"candidate=candidate={geometry_a}",
                f"candidate=candidate={geometry_b}",
            ],
            output_dir=output,
        )

        with (output / "unified_comparison.csv").open(
            newline="", encoding="utf-8-sig"
        ) as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(int(row["seed_count"]), 2)
        self.assertAlmostEqual(float(row["node_mean_r2"]), 0.9)
        self.assertAlmostEqual(float(row["directed_edge_r2"]), 0.5)
        self.assertEqual(int(row["parameter_count"]), 10)
        self.assertAlmostEqual(float(row["total_embedding_variance"]), 3.0)
        self.assertAlmostEqual(float(row["effective_rank"]), 4.0)
        self.assertAlmostEqual(float(row["permutation_error"]), 0.1)
        self.assertAlmostEqual(float(row["permutation_error_sd"]), 2**0.5 / 10)


if __name__ == "__main__":
    unittest.main()
