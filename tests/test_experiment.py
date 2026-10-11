import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
import torch

from social_gnn.data import SocialTrialPackage, collate_social_trials
from social_gnn.experiment import (
    apply_edge_mode,
    fit_feature_standardizer,
    run_experiment,
)


def make_package(trial_id: str, timesteps: int, *, offset: float = 0.0):
    time = np.arange(timesteps, dtype=np.float64) * 0.16
    node_features = np.zeros((timesteps, 2, 3), dtype=np.float32)
    for frame in range(timesteps):
        for animal in range(2):
            node_features[frame, animal] = (
                offset + frame,
                offset + 2 * frame + animal,
                offset - frame + 2 * animal,
            )
    edge_values = np.zeros((timesteps, 2, 2, 8), dtype=np.float32)
    edge_confidence = np.zeros_like(edge_values)
    for frame in range(timesteps):
        for source, target in ((0, 1), (1, 0)):
            edge_values[frame, source, target] = (
                offset + frame + source + np.arange(8, dtype=np.float32)
            )
            edge_confidence[frame, source, target] = 1.0
    return SocialTrialPackage(
        trial_id=trial_id,
        node_features=node_features,
        node_mask=np.ones((timesteps, 2), dtype=bool),
        edge_values=edge_values,
        edge_confidence=edge_confidence,
        patch_start_s=time,
        patch_end_s=time + 0.16,
        identity=np.asarray([1, 2]),
    )


def write_package(root: Path, package: SocialTrialPackage):
    node_path = root / f"{package.trial_id}_node_features.npz"
    edge_path = root / f"{package.trial_id}_social_edges.npz"
    np.savez_compressed(
        node_path,
        trial_id=np.asarray(package.trial_id),
        node_features=package.node_features,
        node_mask=package.node_mask,
        patch_start_s=package.patch_start_s,
        patch_end_s=package.patch_end_s,
        identity=package.identity,
    )
    np.savez_compressed(
        edge_path,
        trial_id=np.asarray(package.trial_id),
        # The production extractor writes this sparse directed-edge view in
        # addition to the unambiguous dense tensor below.
        edge_confidence=package.edge_confidence[:, [0, 1], [1, 0], :],
        edge_value_dense=package.edge_values,
        edge_confidence_dense=package.edge_confidence,
        patch_start_s=package.patch_start_s,
        patch_end_s=package.patch_end_s,
        identity=package.identity,
    )
    return node_path, edge_path


class SocialExperimentTest(unittest.TestCase):
    def setUp(self):
        self.root = Path("social_gnn") / f"_test_experiment_{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_train_only_standardizer_preserves_masks_and_confidence(self):
        train = make_package("train", 5)
        held_out = make_package("held_out", 4, offset=100.0)
        standardizer = fit_feature_standardizer([train])
        transformed_train = standardizer.transform(train)
        transformed_held_out = standardizer.transform(held_out)

        valid_nodes = transformed_train.node_features[transformed_train.node_mask]
        np.testing.assert_allclose(valid_nodes.mean(axis=0), 0.0, atol=1e-6)
        np.testing.assert_allclose(valid_nodes.std(axis=0), 1.0, atol=1e-6)
        self.assertTrue(
            np.array_equal(
                transformed_held_out.edge_confidence, held_out.edge_confidence
            )
        )
        self.assertTrue(np.isfinite(transformed_held_out.node_features).all())
        self.assertTrue(np.isfinite(transformed_held_out.edge_values).all())
        self.assertTrue(np.equal(np.diagonal(transformed_train.edge_values, axis1=1, axis2=2), 0).all())
        # Stored source packages remain immutable and unmodified.
        self.assertEqual(float(train.node_features[0, 0, 0]), 0.0)

    def test_edge_ablation_never_shuffles_across_trials_or_padding(self):
        short = make_package("short", 3, offset=0.0)
        long = make_package("long", 5, offset=100.0)
        batch = collate_social_trials(
            [short.to_torch_sample(), long.to_torch_sample()]
        )
        shuffled = apply_edge_mode(
            batch,
            "shuffled",
            generator=torch.Generator().manual_seed(9),
        )
        for index, length in enumerate((3, 5)):
            original = batch["edge_values"][index, :length, 0, 1, 0]
            changed = shuffled["edge_values"][index, :length, 0, 1, 0]
            self.assertEqual(sorted(original.tolist()), sorted(changed.tolist()))
        self.assertTrue(
            torch.equal(
                shuffled["edge_values"][0, 3:], batch["edge_values"][0, 3:]
            )
        )
        node_only = apply_edge_mode(batch, "node_only")
        self.assertFalse(node_only["edge_values"].any())
        self.assertFalse(node_only["edge_confidence"].any())
        self.assertTrue(batch["edge_confidence"].any())

    def test_one_epoch_formal_run_writes_checkpoints_metrics_and_embeddings(self):
        train = make_package("train", 5)
        validation = make_package("validation", 4, offset=10.0)
        train_node, train_edge = write_package(self.root, train)
        val_node, val_edge = write_package(self.root, validation)
        config = {
            "version": "social_gnn_experiment_v0",
            "output_dir": str(self.root / "runs"),
            "seed": 17,
            "device": "cpu",
            "trials": [
                {"id": "train", "node_npz": str(train_node), "edge_npz": str(train_edge)},
                {"id": "validation", "node_npz": str(val_node), "edge_npz": str(val_edge)},
            ],
            "split": {"validation_trials": ["validation"]},
            "model": {"hidden_dim": 8, "layers": 1, "decoder_hidden_dim": 8},
            "ssl": {
                "node_mask_probability": 1.0,
                "edge_mask_probability": 1.0,
                "node_modality_dropout": 0.0,
                "edge_modality_dropout": 0.0,
            },
            "vae": {"latent_dim": 4, "kl_beta": 0.001},
            "training": {
                "epochs": 1,
                "batch_size": 1,
                "num_workers": 0,
                "learning_rate": 0.001,
            },
            "runs": [
                {"name": "masked", "objective": "masked_ssl", "edge_mode": "full"},
                {"name": "vae", "objective": "vae", "edge_mode": "full"},
            ],
        }
        manifest = run_experiment(config)
        self.assertEqual(manifest["status"], "completed")
        session = Path(manifest["output_dir"])
        self.assertTrue((session / "normalization.json").is_file())
        for name in ("masked", "vae"):
            self.assertTrue((session / name / "best.pt").is_file())
            self.assertTrue((session / name / "last.pt").is_file())
            self.assertTrue((session / name / "metrics.jsonl").is_file())
            embedding_path = (
                session
                / name
                / "embeddings"
                / "validation"
                / "validation_g_t.npz"
            )
            with np.load(embedding_path, allow_pickle=False) as payload:
                self.assertEqual(payload["g_t"].shape[0], 4)
                self.assertTrue(np.isfinite(payload["g_t"]).all())


if __name__ == "__main__":
    unittest.main()
