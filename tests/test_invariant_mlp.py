import io
import unittest

import torch

from social_gnn.graph_builder import compose_edge_inputs
from social_gnn.graph_level_ssl import (
    SymmetrizedMLPFusionEncoder,
    swap_two_animal_graph,
)
from social_gnn.masked_social_ssl import SocialMaskConfig, apply_social_masks
from social_gnn.node_preservation_ssl import (
    NodePreservingGraphSSL,
    hybrid_reconstruction_loss,
    node_preserving_reconstruction_loss,
)


def _batch(batch_size=2, timesteps=4, node_dim=9):
    generator = torch.Generator().manual_seed(914)
    nodes = torch.randn(batch_size, timesteps, 2, node_dim, generator=generator)
    values = torch.randn(batch_size, timesteps, 2, 2, 8, generator=generator)
    confidence = torch.rand(batch_size, timesteps, 2, 2, 8, generator=generator)
    diagonal = torch.eye(2, dtype=torch.bool).view(1, 1, 2, 2, 1)
    values = values.masked_fill(diagonal, 0.0)
    confidence = confidence.masked_fill(diagonal, 0.0)
    return {
        "node_features": nodes,
        "node_mask": torch.ones(batch_size, timesteps, 2, dtype=torch.bool),
        "edge_values": values,
        "edge_confidence": confidence,
        "time_mask": torch.ones(batch_size, timesteps, dtype=torch.bool),
    }


def _swap_batch(batch):
    index = torch.tensor([1, 0])
    swapped = dict(batch)
    swapped["node_features"] = batch["node_features"].index_select(2, index)
    swapped["node_mask"] = batch["node_mask"].index_select(2, index)
    swapped["edge_values"] = (
        batch["edge_values"].index_select(2, index).index_select(3, index)
    )
    swapped["edge_confidence"] = (
        batch["edge_confidence"].index_select(2, index).index_select(3, index)
    )
    return swapped


class InvariantMLPTest(unittest.TestCase):
    def test_swap_transposes_directed_values_confidence_and_validity(self):
        batch = _batch(batch_size=1, timesteps=1)
        batch["node_mask"][0, 0] = torch.tensor([True, False])
        edges = compose_edge_inputs(batch["edge_values"], batch["edge_confidence"])
        nodes, swapped, mask = swap_two_animal_graph(
            batch["node_features"], edges, batch["node_mask"]
        )
        torch.testing.assert_close(nodes[:, :, 0], batch["node_features"][:, :, 1])
        torch.testing.assert_close(swapped[:, :, 0, 1], edges[:, :, 1, 0])
        torch.testing.assert_close(swapped[:, :, 0, 1, :8], batch["edge_values"][:, :, 1, 0])
        torch.testing.assert_close(
            swapped[:, :, 0, 1, 8:], batch["edge_confidence"][:, :, 1, 0]
        )
        self.assertEqual(mask[0, 0].tolist(), [False, True])

    def test_exact_simultaneous_swap_invariance(self):
        batch = _batch()
        model = SymmetrizedMLPFusionEncoder(
            node_dim=9,
            number_of_nodes=2,
            edge_dim=16,
            graph_dim=64,
            fusion_hidden_dim=160,
        ).eval()
        edge = compose_edge_inputs(batch["edge_values"], batch["edge_confidence"])
        swapped = _swap_batch(batch)
        swapped_edge = compose_edge_inputs(
            swapped["edge_values"], swapped["edge_confidence"]
        )
        with torch.no_grad():
            original_g = model(batch["node_features"], edge, batch["node_mask"])
            swapped_g = model(
                swapped["node_features"], swapped_edge, swapped["node_mask"]
            )
        self.assertEqual(tuple(original_g.shape), (2, 4, 64))
        torch.testing.assert_close(original_g, swapped_g, atol=0.0, rtol=0.0)

    def test_invalid_pair_cannot_leak_edge_values(self):
        batch = _batch(batch_size=1)
        batch["node_mask"][:, :, 1] = False
        changed = dict(batch)
        changed["edge_values"] = batch["edge_values"] + 10000.0
        changed["edge_confidence"] = torch.ones_like(batch["edge_confidence"])
        model = SymmetrizedMLPFusionEncoder(
            node_dim=9,
            number_of_nodes=2,
            edge_dim=16,
            graph_dim=12,
        ).eval()
        with torch.no_grad():
            first = model(
                batch["node_features"],
                compose_edge_inputs(batch["edge_values"], batch["edge_confidence"]),
                batch["node_mask"],
            )
            second = model(
                changed["node_features"],
                compose_edge_inputs(changed["edge_values"], changed["edge_confidence"]),
                changed["node_mask"],
            )
        torch.testing.assert_close(first, second)

    def test_no_cross_batch_or_time_leakage(self):
        batch = _batch(batch_size=2, timesteps=3)
        model = SymmetrizedMLPFusionEncoder(
            node_dim=9,
            number_of_nodes=2,
            edge_dim=16,
            graph_dim=11,
        ).eval()
        edge = compose_edge_inputs(batch["edge_values"], batch["edge_confidence"])
        with torch.no_grad():
            baseline = model(batch["node_features"], edge, batch["node_mask"])
            changed_nodes = batch["node_features"].clone()
            changed_edges = edge.clone()
            changed_nodes[1] += 1000.0
            changed_edges[1] -= 1000.0
            changed_nodes[0, 2] += 500.0
            changed_edges[0, 2] += 500.0
            changed = model(changed_nodes, changed_edges, batch["node_mask"])
        torch.testing.assert_close(baseline[0, :2], changed[0, :2])
        self.assertFalse(torch.allclose(baseline[1], changed[1]))
        self.assertFalse(torch.allclose(baseline[0, 2], changed[0, 2]))

    def test_decoders_use_only_g_and_gradients_cross_bottleneck(self):
        batch = _batch()
        model = NodePreservingGraphSSL(
            node_dim=9,
            graph_dim=64,
            hidden_dim=64,
            architecture="invariant_mlp",
            target_mode="summary",
            invariant_mlp_hidden_dim=160,
        )
        output = model(batch, view="full")
        output["g_t"].retain_grad()
        self.assertEqual(model.node_decoder[0].in_features, 64)
        self.assertEqual(model.edge_decoder[0].in_features, 64)
        loss = node_preserving_reconstruction_loss(output, batch)["total"]
        loss.backward()
        self.assertIsNotNone(output["g_t"].grad)
        self.assertGreater(float(output["g_t"].grad.abs().sum()), 0.0)
        self.assertGreater(
            sum(
                float(parameter.grad.abs().sum())
                for parameter in model.encoder.parameters()
                if parameter.grad is not None
            ),
            0.0,
        )

    def test_masked_input_is_the_only_encoder_input(self):
        batch = _batch()
        masked = apply_social_masks(
            batch,
            SocialMaskConfig(
                node_mask_probability=1.0,
                edge_mask_probability=1.0,
                couple_symmetric_traits=True,
            ),
            generator=torch.Generator().manual_seed(25),
        )
        model = NodePreservingGraphSSL(
            node_dim=9,
            graph_dim=16,
            architecture="invariant_mlp",
            target_mode="summary",
        ).eval()
        changed = dict(batch)
        changed["node_features"] = batch["node_features"] + 1234.0
        changed["edge_values"] = batch["edge_values"] - 1234.0
        with torch.no_grad():
            first = model(batch, view="masked", masked_inputs=masked)
            second = model(changed, view="masked", masked_inputs=masked)
        torch.testing.assert_close(first["g_t"], second["g_t"])
        torch.testing.assert_close(first["node_prediction"], second["node_prediction"])
        torch.testing.assert_close(first["edge_prediction"], second["edge_prediction"])
        hybrid = hybrid_reconstruction_loss(first, model(batch, view="full"), batch)
        self.assertTrue(torch.isfinite(hybrid["total"]))

    def test_eval_is_deterministic_and_checkpoint_roundtrip_is_exact(self):
        batch = _batch()
        model = NodePreservingGraphSSL(
            node_dim=9,
            graph_dim=64,
            architecture="invariant_mlp",
            target_mode="summary",
        ).eval()
        with torch.no_grad():
            first = model(batch, view="full")["g_t"]
            second = model(batch, view="full")["g_t"]
        torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored = NodePreservingGraphSSL(
            node_dim=9,
            graph_dim=64,
            architecture="invariant_mlp",
            target_mode="summary",
        ).eval()
        restored.load_state_dict(torch.load(stream, weights_only=True))
        with torch.no_grad():
            replay = restored(batch, view="full")["g_t"]
        torch.testing.assert_close(first, replay, atol=0.0, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
