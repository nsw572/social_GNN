import unittest

import torch

from social_gnn.graph_builder import complete_directed_edge_index, compose_edge_inputs
from social_gnn.models import SocialV0


class SocialV0Test(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
