import unittest

import torch

from social_gnn.masked_social_ssl import SocialMaskConfig, apply_social_masks
from social_gnn.node_preservation_ssl import (
    DualBranchSocialEncoder,
    NodePreservingGraphSSL,
    NodeSetEncoder,
    node_preserving_reconstruction_loss,
)


def _batch(timesteps=4, node_dim=6):
    torch.manual_seed(112)
    nodes = torch.randn(2, timesteps, 2, node_dim)
    values = torch.randn(2, timesteps, 2, 2, 8)
    confidence = torch.rand_like(values)
    diagonal = torch.eye(2, dtype=torch.bool).view(1, 1, 2, 2, 1)
    values = values.masked_fill(diagonal, 0.0)
    confidence = confidence.masked_fill(diagonal, 0.0)
    return {
        "node_features": nodes,
        "node_mask": torch.ones(2, timesteps, 2, dtype=torch.bool),
        "edge_values": values,
        "edge_confidence": confidence,
        "time_mask": torch.ones(2, timesteps, dtype=torch.bool),
    }


def _permute(batch):
    permutation = torch.tensor([1, 0])
    result = dict(batch)
    result["node_features"] = batch["node_features"].index_select(2, permutation)
    result["node_mask"] = batch["node_mask"].index_select(2, permutation)
    result["edge_values"] = batch["edge_values"].index_select(2, permutation).index_select(3, permutation)
    result["edge_confidence"] = batch["edge_confidence"].index_select(2, permutation).index_select(3, permutation)
    return result


class NodePreservationSSLTest(unittest.TestCase):
    def test_summary_target_tracks_nine_dimensional_nodes(self):
        batch = _batch(timesteps=3, node_dim=9)
        model = NodePreservingGraphSSL(
            node_dim=9,
            graph_dim=8,
            hidden_dim=8,
            architecture="gnn",
            target_mode="summary",
        )
        output = model(batch, view="full")
        self.assertEqual(tuple(output["node_prediction"].shape), (2, 3, 18))
        loss = node_preserving_reconstruction_loss(output, batch)["total"]
        self.assertTrue(torch.isfinite(loss))

    def test_node_set_and_dual_encoders_are_permutation_invariant(self):
        batch = _batch()
        edge_features = torch.cat((batch["edge_values"], batch["edge_confidence"]), dim=-1)
        permuted = _permute(batch)
        permuted_edges = torch.cat(
            (permuted["edge_values"], permuted["edge_confidence"]), dim=-1
        )
        encoders = (
            NodeSetEncoder(node_dim=6, graph_dim=9),
            DualBranchSocialEncoder(
                node_dim=6,
                edge_dim=16,
                hidden_dim=12,
                graph_dim=9,
                layers=2,
                number_of_nodes=2,
            ),
        )
        for encoder in encoders:
            encoder.eval()
            with torch.no_grad():
                original = encoder(
                    batch["node_features"], edge_features, batch["node_mask"]
                )
                swapped = encoder(
                    permuted["node_features"],
                    permuted_edges,
                    permuted["node_mask"],
                )
            self.assertEqual(tuple(original.shape), (2, 4, 9))
            torch.testing.assert_close(original, swapped, atol=1e-6, rtol=1e-6)

    def test_decoders_only_receive_g_and_gradients_reach_g(self):
        batch = _batch()
        model = NodePreservingGraphSSL(
            node_dim=6,
            graph_dim=10,
            hidden_dim=12,
            architecture="dual",
            target_mode="joint_set",
        )
        output = model(batch, view="full")
        output["g_t"].retain_grad()
        self.assertEqual(model.node_decoder[0].in_features, 10)
        self.assertEqual(model.edge_decoder[0].in_features, 10)
        loss = node_preserving_reconstruction_loss(output, batch)["total"]
        loss.backward()
        self.assertIsNotNone(output["g_t"].grad)
        self.assertGreater(float(output["g_t"].grad.abs().sum()), 0.0)

    def test_masked_branch_never_reads_original_hidden_targets(self):
        batch = _batch()
        mask_config = SocialMaskConfig(
            node_mask_probability=1.0,
            edge_mask_probability=1.0,
            couple_symmetric_traits=True,
        )
        masked = apply_social_masks(batch, mask_config)
        model = NodePreservingGraphSSL(
            node_dim=6,
            graph_dim=8,
            hidden_dim=8,
            architecture="dual",
            target_mode="summary",
        ).eval()
        changed = dict(batch)
        changed["node_features"] = batch["node_features"] + 1000.0
        changed["edge_values"] = batch["edge_values"] - 1000.0
        with torch.no_grad():
            first = model(batch, view="masked", masked_inputs=masked)
            second = model(changed, view="masked", masked_inputs=masked)
        torch.testing.assert_close(first["g_t"], second["g_t"])
        torch.testing.assert_close(first["node_prediction"], second["node_prediction"])
        torch.testing.assert_close(first["edge_prediction"], second["edge_prediction"])

    def test_joint_matching_uses_one_consistent_node_edge_permutation(self):
        batch = _batch(timesteps=2)
        permutation = torch.tensor([1, 0])
        node_prediction = batch["node_features"].index_select(2, permutation).clone()
        edge_prediction = batch["edge_values"].index_select(2, permutation).index_select(3, permutation).clone()
        outputs = {
            "g_t": torch.zeros(2, 2, 5, requires_grad=True),
            "node_prediction": node_prediction.requires_grad_(),
            "edge_prediction": edge_prediction.requires_grad_(),
            "masked_inputs": None,
            "view": "full",
            "target_mode": "joint_set",
        }
        losses = node_preserving_reconstruction_loss(outputs, batch)
        self.assertAlmostEqual(float(losses["node"].detach()), 0.0, places=7)
        self.assertAlmostEqual(float(losses["edge"].detach()), 0.0, places=7)

        inconsistent = dict(outputs)
        inconsistent["edge_prediction"] = batch["edge_values"].clone()
        inconsistent_loss = node_preserving_reconstruction_loss(inconsistent, batch)
        self.assertGreater(float(inconsistent_loss["total"].detach()), 0.0)

    def test_social_v0_stage_shapes_remain_patch_aligned(self):
        batch = _batch(timesteps=3)
        model = NodePreservingGraphSSL(
            node_dim=6,
            graph_dim=8,
            hidden_dim=8,
            architecture="gnn",
            target_mode="summary",
        )
        edge_features = torch.cat((batch["edge_values"], batch["edge_confidence"]), dim=-1)
        graph, stages = model.encoder(
            batch["node_features"],
            edge_features,
            batch["node_mask"],
            return_stages=True,
        )
        self.assertEqual(tuple(graph.shape), (2, 3, 8))
        self.assertEqual(len(stages), 3)
        for stage in stages:
            self.assertEqual(tuple(stage.shape), (2, 3, 2, 8))


if __name__ == "__main__":
    unittest.main()
