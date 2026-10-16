import importlib
import unittest

import torch
from torch import nn

import social_gnn
from social_gnn.graph_builder import complete_directed_edge_index, compose_edge_inputs
from social_gnn.models import EdgeAwareGINELayer, SocialV0


class _SelectAggregate(nn.Module):
    def forward(self, concatenated):
        return concatenated[..., 1:]


class SocialV0Test(unittest.TestCase):
    def test_graph_dim_can_be_smaller_than_message_passing_width(self):
        model = SocialV0(
            node_dim=3,
            edge_dim=16,
            hidden_dim=8,
            graph_dim=4,
            layers=1,
        )
        nodes = torch.randn(2, 5, 2, 3)
        edges = torch.randn(2, 5, 2, 2, 16)
        graph, node_embeddings = model(
            nodes, edges, return_node_embeddings=True
        )
        self.assertEqual(tuple(graph.shape), (2, 5, 4))
        self.assertEqual(tuple(node_embeddings.shape), (2, 5, 2, 8))

    def test_role_aware_readout_preserves_shape_and_order_sensitivity(self):
        torch.manual_seed(4)
        model = SocialV0(
            node_dim=3,
            edge_dim=16,
            hidden_dim=8,
            layers=1,
            readout="role_aware",
            number_of_nodes=2,
        )
        nodes = torch.randn(1, 3, 2, 3)
        edges = torch.randn(1, 3, 2, 2, 16)
        first = model(nodes, edges)
        second = model(nodes.flip(2), edges.flip(2).flip(3))
        self.assertEqual(tuple(first.shape), (1, 3, 8))
        self.assertFalse(torch.allclose(first, second))

    def test_edge_value_and_coverage_aware_confidence_make_16_channels(self):
        values = torch.ones(2, 3, 2, 2, 8)
        confidence = torch.full_like(values, 0.5)
        combined = compose_edge_inputs(values, confidence)
        self.assertEqual(tuple(combined.shape), (2, 3, 2, 2, 16))
        self.assertTrue(torch.equal(combined[..., :8], values))
        self.assertTrue(torch.equal(combined[..., 8:], confidence))

    def test_directed_complete_graph(self):
        edge_index = complete_directed_edge_index(3)
        self.assertEqual(tuple(edge_index.shape), (2, 6))
        self.assertEqual({tuple(x) for x in edge_index.t().tolist()},
                         {(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)})

    def test_dense_message_passing_respects_source_to_target_direction(self):
        layer = EdgeAwareGINELayer(node_dim=1, edge_dim=1, hidden_dim=1)
        layer.edge_encoder = nn.Identity()
        layer.message_mlp = nn.Identity()
        layer.node_update = _SelectAggregate()
        nodes = torch.tensor([[[10.0], [20.0]]])
        edges = torch.zeros(1, 2, 2, 1)
        edges[:, 0, 1, 0] = 5.0
        output = layer(nodes, edges)
        self.assertTrue(torch.equal(output, torch.tensor([[[20.0], [15.0]]])))

    def test_dense_message_passing_excludes_self_loops_and_invalid_senders(self):
        layer = EdgeAwareGINELayer(node_dim=1, edge_dim=1, hidden_dim=1)
        layer.edge_encoder = nn.Identity()
        layer.message_mlp = nn.Identity()
        layer.node_update = _SelectAggregate()
        nodes = torch.tensor([[[10.0], [20.0]]])
        edges = torch.full((1, 2, 2, 1), 100.0)
        output = layer(nodes, edges, node_mask=torch.tensor([[True, False]]))
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_multi_animal_shape_and_gradients(self):
        model = SocialV0(node_dim=5, edge_dim=3, hidden_dim=8)
        nodes = torch.randn(2, 4, 3, 5, requires_grad=True)
        edges = torch.randn(2, 4, 3, 3, 3)
        output = model(nodes, edges)
        self.assertEqual(tuple(output.shape), (2, 4, 8))
        output.mean().backward()
        self.assertIsNotNone(nodes.grad)

    def test_single_animal_bypass(self):
        model = SocialV0(node_dim=5, edge_dim=3, hidden_dim=8)
        nodes = torch.randn(2, 4, 1, 5)
        edges = torch.randn(2, 4, 1, 1, 3)
        output = model(nodes, edges)
        self.assertEqual(tuple(output.shape), (2, 4, 8))

    def test_each_patch_embedding_is_temporally_independent(self):
        torch.manual_seed(23)
        model = SocialV0(node_dim=5, edge_dim=3, hidden_dim=8).eval()
        nodes = torch.randn(1, 4, 2, 5)
        edges = torch.randn(1, 4, 2, 2, 3)
        changed_nodes = nodes.clone()
        changed_edges = edges.clone()
        changed_nodes[:, 2:] += 100.0
        changed_edges[:, 2:] -= 100.0
        with torch.no_grad():
            original = model(nodes, edges)
            changed = model(changed_nodes, changed_edges)
        self.assertTrue(torch.equal(original[:, :2], changed[:, :2]))

    def test_temporal_encoder_is_absent_from_public_model(self):
        self.assertFalse(hasattr(social_gnn, "TemporalConvNet"))
        self.assertFalse(hasattr(social_gnn, "SocialGNNWithTCN"))
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("social_gnn.temporal")


if __name__ == "__main__":
    unittest.main()
