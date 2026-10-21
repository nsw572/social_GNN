import unittest
import numpy as np
import torch

from social_gnn.data import SocialTrialPackage
from social_gnn.evaluation import (
    LoadedEncoder,
    SYMMETRIC_TRAITS,
    encode_package,
    displacement_summary,
    matched_replacement_indices,
    nonidentity_permutation,
    perturb_package,
    unique_edge_targets,
)
from social_gnn.experiment import fit_feature_standardizer
from social_gnn.masked_social_ssl import MaskedSocialSSL


def make_package(timesteps: int = 40) -> SocialTrialPackage:
    time = np.arange(timesteps, dtype=np.float64) * 0.16
    nodes = np.zeros((timesteps, 2, 3), dtype=np.float32)
    nodes[:, 0, 0] = np.linspace(0, 1, timesteps)
    nodes[:, 1, 0] = np.linspace(1, 0, timesteps)
    edges = np.zeros((timesteps, 2, 2, 8), dtype=np.float32)
    confidence = np.zeros_like(edges)
    for trait in range(8):
        forward = np.arange(timesteps, dtype=np.float32) + trait * 100
        reverse = forward if trait in SYMMETRIC_TRAITS else forward + 10_000
        edges[:, 0, 1, trait] = forward
        edges[:, 1, 0, trait] = reverse
        confidence[:, 0, 1, trait] = 0.5 + trait / 100
        confidence[:, 1, 0, trait] = 0.5 + trait / 100
    return SocialTrialPackage(
        trial_id="trial",
        node_features=nodes,
        node_mask=np.ones((timesteps, 2), dtype=bool),
        edge_values=edges,
        edge_confidence=confidence,
        patch_start_s=time,
        patch_end_s=time + 0.16,
        identity=np.asarray([1, 2]),
    )


class EvaluationUtilitiesTest(unittest.TestCase):
    def test_gnn_probe_is_deterministic_and_patch_aligned(self):
        package = make_package(12)
        normalizer = fit_feature_standardizer([package])
        normalized = normalizer.transform(package)
        model = MaskedSocialSSL(
            node_dim=3,
            number_of_nodes=2,
            hidden_dim=8,
            layers=1,
        ).eval()
        common = dict(
            name="gnn",
            model=model,
            normalizer=normalizer,
            native_edge_mode="full",
            checkpoint_path="gnn.pt",
            epoch=1,
            parameter_count=sum(value.numel() for value in model.parameters()),
            objective="masked_ssl",
        )
        graph = encode_package(
            LoadedEncoder(**common),
            normalized,
            device=torch.device("cpu"),
        )
        self.assertEqual(graph.shape, (12, 8))
        np.testing.assert_array_equal(
            graph,
            encode_package(
                LoadedEncoder(**common),
                normalized,
                device=torch.device("cpu"),
            ),
        )

    def test_derangement_has_no_identity_mapping(self):
        permutation = nonidentity_permutation(101, np.random.default_rng(4))
        self.assertEqual(sorted(permutation.tolist()), list(range(101)))
        self.assertFalse(np.any(permutation == np.arange(101)))

    def test_trait_perturbation_couples_both_directions(self):
        package = make_package()
        permutation = np.roll(np.arange(40), 3)
        changed = perturb_package(
            package,
            "trait",
            rng=np.random.default_rng(1),
            permutation=permutation,
            trait=0,
        )
        np.testing.assert_array_equal(
            changed.edge_values[:, 0, 1, 0],
            package.edge_values[permutation, 0, 1, 0],
        )
        np.testing.assert_array_equal(
            changed.edge_values[:, 1, 0, 0],
            package.edge_values[permutation, 1, 0, 0],
        )
        np.testing.assert_array_equal(
            changed.edge_values[..., 1], package.edge_values[..., 1]
        )

    def test_unique_targets_do_not_duplicate_symmetric_traits(self):
        values, confidence, names = unique_edge_targets(make_package())
        self.assertEqual(values.shape, (40, 11))
        self.assertEqual(confidence.shape, (40, 11))
        self.assertEqual(len(names), 11)
        self.assertEqual(sum(name == "centroid_distance" for name in names), 1)
        self.assertIn("source_faces_target_0_to_1", names)
        self.assertIn("source_faces_target_1_to_0", names)

    def test_matched_indices_are_real_nonlocal_patches(self):
        package = make_package(80)
        different, similar, diagnostics = matched_replacement_indices(
            package,
            rng=np.random.default_rng(8),
            candidate_count=24,
            minimum_time_gap=5,
        )
        self.assertEqual(different.shape, (80,))
        self.assertEqual(similar.shape, (80,))
        self.assertFalse(np.any(different == np.arange(80)))
        self.assertTrue(np.isfinite(diagnostics["edge_distance"]).all())
        self.assertTrue(
            np.all(
                diagnostics["edge_distance"]
                >= diagnostics["control_edge_distance"]
            )
        )

    def test_normalized_displacement_is_scale_invariant(self):
        rng = np.random.default_rng(9)
        reference = rng.normal(size=(50, 8))
        alternative = reference + rng.normal(scale=0.1, size=(50, 8))
        scale = float(np.sqrt(np.var(reference, axis=0).sum()))
        first = displacement_summary(reference, alternative, scale=scale)
        second = displacement_summary(
            reference * 20, alternative * 20, scale=scale * 20
        )
        self.assertAlmostEqual(
            first["mean_normalized_displacement"],
            second["mean_normalized_displacement"],
            places=10,
        )


if __name__ == "__main__":
    unittest.main()
