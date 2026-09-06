import shutil
import unittest
from pathlib import Path

import numpy as np
import torch

from social_gnn.data import (
    SocialTrialDataset,
    SocialTrialPackage,
    SocialTrialSource,
    SocialTrialValidationError,
    build_social_dataloader,
    collate_social_trials,
)
from social_gnn.graph_builder import compose_edge_inputs
from social_gnn.models import SocialGNNWithTCN


def make_package(trial_id: str, timesteps: int, *, node_dim: int = 4):
    patch_start_s = np.arange(timesteps, dtype=np.float64) * 0.25
    package = SocialTrialPackage(
        trial_id=trial_id,
        node_features=np.arange(timesteps * 2 * node_dim, dtype=np.float32).reshape(
            timesteps, 2, node_dim
        ),
        node_mask=np.ones((timesteps, 2), dtype=bool),
        edge_values=np.zeros((timesteps, 2, 2, 8), dtype=np.float32),
        edge_confidence=np.ones((timesteps, 2, 2, 8), dtype=np.float32),
        patch_start_s=patch_start_s,
        patch_end_s=patch_start_s + 0.25,
        identity=np.asarray(["blue", "red"]),
    )
    return package


class SocialTrialDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output_dir = Path("social_gnn") / "_test_social_trial_data"
        if cls.output_dir.exists():
            shutil.rmtree(cls.output_dir)
        cls.output_dir.mkdir(parents=True)

    @classmethod
    def tearDownClass(cls):
        if cls.output_dir.exists():
            shutil.rmtree(cls.output_dir)

    def test_package_rejects_nonfinite_and_out_of_range_confidence(self):
        valid = make_package("base", 3)
        bad_nodes = np.array(valid.node_features, copy=True)
        bad_nodes[1, 0, 0] = np.nan
        with self.assertRaisesRegex(
            SocialTrialValidationError, "Trial 'nan_trial'.*node_features"
        ):
            SocialTrialPackage(
                "nan_trial",
                bad_nodes,
                valid.node_mask,
                valid.edge_values,
                valid.edge_confidence,
                valid.patch_start_s,
                valid.patch_end_s,
                valid.identity,
            )

        bad_confidence = np.array(valid.edge_confidence, copy=True)
        bad_confidence[2, 0, 1, 4] = 1.01
        with self.assertRaisesRegex(
            SocialTrialValidationError, r"Trial 'bad_conf'.*\[0,1\]"
        ):
            SocialTrialPackage(
                "bad_conf",
                valid.node_features,
                valid.node_mask,
                valid.edge_values,
                bad_confidence,
                valid.patch_start_s,
                valid.patch_end_s,
                valid.identity,
            )

    def test_from_npz_strictly_rejects_clock_and_identity_mismatch(self):
        valid = make_package("source", 4)
        node_path = self.output_dir / "source_node_features.npz"
        edge_path = self.output_dir / "source_social_edges.npz"
        np.savez_compressed(
            node_path,
            trial_id=np.asarray("source"),
            node_features=valid.node_features,
            node_mask=valid.node_mask,
            patch_start_s=valid.patch_start_s,
            patch_end_s=valid.patch_end_s,
            identity=valid.identity,
        )
        wrong_start = np.array(valid.patch_start_s, copy=True)
        wrong_start[2] += 0.01
        np.savez_compressed(
            edge_path,
            trial_id=np.asarray("source"),
            edge_value_dense=valid.edge_values,
            edge_confidence_dense=valid.edge_confidence,
            patch_start_s=wrong_start,
            patch_end_s=valid.patch_end_s,
            identity=valid.identity,
        )
        with self.assertRaisesRegex(
            SocialTrialValidationError, "Trial 'source'.*patch_start_s mismatch"
        ):
            SocialTrialPackage.from_npz(node_path, edge_path)

        np.savez_compressed(
            edge_path,
            trial_id=np.asarray("source"),
            edge_value_dense=valid.edge_values,
            edge_confidence_dense=valid.edge_confidence,
            patch_start_s=valid.patch_start_s,
            patch_end_s=valid.patch_end_s,
            identity=np.asarray(["red", "blue"]),
        )
        with self.assertRaisesRegex(
            SocialTrialValidationError, "Trial 'source'.*identity mismatch"
        ):
            SocialTrialPackage.from_npz(node_path, edge_path)

    def test_lazy_source_reads_existing_edge_extractor_keys(self):
        valid = make_package("lazy", 3)
        node_path = self.output_dir / "lazy_node_features.npz"
        edge_path = self.output_dir / "lazy_social_edges.npz"
        np.savez_compressed(
            node_path,
            node_features=valid.node_features,
            node_mask=valid.node_mask,
            patch_start_s=valid.patch_start_s,
            patch_end_s=valid.patch_end_s,
            identity=valid.identity,
        )
        np.savez_compressed(
            edge_path,
            edge_value_dense=valid.edge_values,
            edge_confidence_dense=valid.edge_confidence,
            patch_start_s=valid.patch_start_s,
            patch_end_s=valid.patch_end_s,
            identity=valid.identity,
        )
        dataset = SocialTrialDataset(
            [SocialTrialSource("lazy", node_path, edge_path)]
        )
        sample = dataset[0]
        self.assertEqual(tuple(sample["node_features"].shape), (3, 2, 4))
        self.assertEqual(tuple(sample["edge_values"].shape), (3, 2, 2, 8))

    def test_dataset_samples_are_fresh_for_future_ssl_augmentation(self):
        package = make_package("immutable", 3)
        dataset = SocialTrialDataset([package])
        first = dataset[0]
        original = float(first["node_features"][0, 0, 0])
        first["node_features"][0, 0, 0] = 999.0
        second = dataset[0]
        self.assertEqual(float(second["node_features"][0, 0, 0]), original)
        self.assertFalse(package.node_features.flags.writeable)

    def test_collate_pads_only_time_and_preserves_trial_boundaries(self):
        short = make_package("short", 3)
        long = make_package("long", 5)
        short_mask = np.array(short.node_mask, copy=True)
        short_mask[1, 1] = False
        short = SocialTrialPackage(
            short.trial_id,
            short.node_features,
            short_mask,
            short.edge_values,
            short.edge_confidence,
            short.patch_start_s,
            short.patch_end_s,
            short.identity,
        )
        batch = collate_social_trials(
            [short.to_torch_sample(), long.to_torch_sample()]
        )
        self.assertEqual(tuple(batch["node_features"].shape), (2, 5, 2, 4))
        self.assertEqual(tuple(batch["edge_values"].shape), (2, 5, 2, 2, 8))
        self.assertEqual(
            batch["time_mask"].tolist(), [[True] * 3 + [False] * 2, [True] * 5]
        )
        self.assertFalse(bool(batch["node_mask"][0, 1, 1]))
        self.assertTrue(
            torch.equal(batch["node_features"][0, 3:], torch.zeros(2, 2, 4))
        )
        self.assertEqual(batch["patch_start_s"][1, 0].item(), 0.0)
        self.assertEqual(batch["trial_id"], ["short", "long"])
        self.assertEqual(batch["sequence_length"].tolist(), [3, 5])

    def test_dataloader_shuffle_unit_is_a_full_trial(self):
        dataset = SocialTrialDataset(
            [make_package("a", 2), make_package("b", 4)]
        )
        loader = build_social_dataloader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        self.assertEqual(batch["trial_id"], ["a", "b"])
        self.assertEqual(batch["sequence_length"].tolist(), [2, 4])

    def test_padded_batch_feeds_graph_tcn_without_upstream_files(self):
        dataset = SocialTrialDataset(
            [make_package("short_model", 3), make_package("long_model", 5)]
        )
        batch = next(
            iter(build_social_dataloader(dataset, batch_size=2, shuffle=False))
        )
        edge_features = compose_edge_inputs(
            batch["edge_values"], batch["edge_confidence"]
        )
        model = SocialGNNWithTCN(
            node_dim=4,
            edge_dim=16,
            graph_hidden_dim=8,
            temporal_hidden_dim=6,
            tcn_levels=2,
            tcn_dropout=0.0,
        ).eval()
        with torch.no_grad():
            social_state = model(
                batch["node_features"],
                edge_features,
                node_mask=batch["node_mask"],
                time_mask=batch["time_mask"],
            )
        self.assertEqual(tuple(social_state.shape), (2, 5, 6))
        self.assertTrue(
            torch.equal(social_state[0, 3:], torch.zeros_like(social_state[0, 3:]))
        )


if __name__ == "__main__":
    unittest.main()
