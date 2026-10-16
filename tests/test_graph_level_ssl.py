import unittest

import torch

from social_gnn.graph_level_ssl import (
    GRAPH_EDGE_TARGET_DIM,
    GraphLevelSocialSSL,
    graph_node_target_dim,
    graph_level_reconstruction_loss,
    graph_level_targets,
)
from social_gnn.masked_social_ssl import SocialMaskConfig, apply_social_masks
from social_gnn.models import SocialV0


def _batch(timesteps=5, node_dim=6):
    torch.manual_seed(31)
    values = torch.randn(1, timesteps, 2, 2, 8)
    confidence = torch.rand(1, timesteps, 2, 2, 8)
    diagonal = torch.eye(2, dtype=torch.bool).view(1, 1, 2, 2, 1)
    values = values.masked_fill(diagonal, 0.0)
    confidence = confidence.masked_fill(diagonal, 0.0)
    return {
        "node_features": torch.randn(1, timesteps, 2, node_dim),
        "node_mask": torch.ones(1, timesteps, 2, dtype=torch.bool),
        "edge_values": values,
        "edge_confidence": confidence,
        "time_mask": torch.ones(1, timesteps, dtype=torch.bool),
    }


class GraphLevelSSLTest(unittest.TestCase):
    def test_invariant_targets_have_documented_dimensions(self):
        batch = _batch()
        masked = apply_social_masks(
            batch,
            SocialMaskConfig(
                node_mask_probability=1.0,
                edge_mask_probability=1.0,
                couple_symmetric_traits=True,
            ),
        )
        targets = graph_level_targets(batch, masked)
        self.assertEqual(tuple(targets.node_values.shape), (1, 5, graph_node_target_dim(6)))
        self.assertEqual(tuple(targets.edge_values.shape), (1, 5, GRAPH_EDGE_TARGET_DIM))
        self.assertTrue(targets.node_selected.all())
        self.assertTrue(targets.edge_selected.all())

    def test_nine_dimensional_nodes_expand_summary_and_decoder_dynamically(self):
        batch = _batch(node_dim=9)
        masked = apply_social_masks(
            batch,
            SocialMaskConfig(
                node_mask_probability=1.0,
                edge_mask_probability=1.0,
                couple_symmetric_traits=True,
            ),
        )
        targets = graph_level_targets(batch, masked)
        model = GraphLevelSocialSSL(
            node_dim=9,
            hidden_dim=8,
            graph_dim=4,
            decoder_hidden_dim=7,
            layers=1,
        )
        outputs = model(batch, masked_inputs=masked)
        self.assertEqual(tuple(targets.node_values.shape), (1, 5, 18))
        self.assertEqual(tuple(outputs["node_prediction"].shape), (1, 5, 18))
        loss = graph_level_reconstruction_loss(outputs, batch)["total"]
        self.assertTrue(torch.isfinite(loss))

    def test_graph_decoders_use_only_g_and_gradients_reach_g(self):
        batch = _batch()
        model = GraphLevelSocialSSL(
            node_dim=6,
            hidden_dim=8,
            graph_dim=4,
            decoder_hidden_dim=7,
            layers=1,
            readout="mean",
        )
        outputs = model(
            batch,
            mask_config=SocialMaskConfig(
                node_mask_probability=1.0,
                edge_mask_probability=1.0,
                couple_symmetric_traits=True,
            ),
        )
        outputs["g_t"].retain_grad()
        self.assertEqual(model.node_decoder[0].in_features, 4)
        self.assertEqual(model.edge_decoder[0].in_features, 4)
        loss = graph_level_reconstruction_loss(outputs, batch)["total"]
        loss.backward()
        self.assertIsNotNone(outputs["g_t"].grad)
        self.assertGreater(float(outputs["g_t"].grad.abs().sum()), 0.0)

    def test_soft_role_readout_is_permutation_invariant(self):
        torch.manual_seed(41)
        model = SocialV0(
            node_dim=6,
            edge_dim=16,
            hidden_dim=12,
            graph_dim=9,
            layers=2,
            readout="soft_role",
            number_of_nodes=2,
        ).eval()
        nodes = torch.randn(2, 4, 2, 6)
        edges = torch.randn(2, 4, 2, 2, 16)
        mask = torch.ones(2, 4, 2, dtype=torch.bool)
        original = model(nodes, edges, node_mask=mask)
        permutation = torch.tensor([1, 0])
        permuted = model(
            nodes.index_select(2, permutation),
            edges.index_select(2, permutation).index_select(3, permutation),
            node_mask=mask.index_select(2, permutation),
        )
        torch.testing.assert_close(original, permuted, atol=1e-6, rtol=1e-6)

    def test_ordered_mlp_baseline_is_not_claimed_invariant(self):
        batch = _batch()
        model = GraphLevelSocialSSL(
            node_dim=6,
            graph_dim=8,
            backbone="mlp",
            fusion_hidden_dim=32,
        ).eval()
        original = model.encode_batch(batch)
        permutation = torch.tensor([1, 0])
        permuted_batch = dict(batch)
        permuted_batch["node_features"] = batch["node_features"].index_select(2, permutation)
        permuted_batch["node_mask"] = batch["node_mask"].index_select(2, permutation)
        permuted_batch["edge_values"] = batch["edge_values"].index_select(2, permutation).index_select(3, permutation)
        permuted_batch["edge_confidence"] = batch["edge_confidence"].index_select(2, permutation).index_select(3, permutation)
        permuted = model.encode_batch(permuted_batch)
        self.assertFalse(torch.allclose(original, permuted))


if __name__ == "__main__":
    unittest.main()
