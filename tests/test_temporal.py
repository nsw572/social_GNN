import unittest

import torch

from social_gnn.models import SocialGNNWithTCN
from social_gnn.temporal import TemporalConvNet


class TemporalConvNetTest(unittest.TestCase):
    def test_shape_and_receptive_field(self):
        model = TemporalConvNet(
            input_dim=5,
            hidden_dim=7,
            levels=4,
            kernel_size=3,
            dilation_base=2,
            dropout=0.0,
        )
        output = model(torch.randn(2, 11, 5))
        self.assertEqual(tuple(output.shape), (2, 11, 7))
        self.assertEqual(model.receptive_field, 61)

    def test_future_inputs_cannot_change_past_states(self):
        torch.manual_seed(7)
        model = TemporalConvNet(
            input_dim=4, hidden_dim=6, levels=3, kernel_size=3, dropout=0.0
        ).eval()
        original = torch.randn(2, 12, 4)
        changed = original.clone()
        changed[:, 6:] = changed[:, 6:] + 100.0 * torch.randn_like(changed[:, 6:])
        with torch.no_grad():
            original_state = model(original)
            changed_state = model(changed)
        self.assertTrue(
            torch.allclose(original_state[:, :6], changed_state[:, :6], atol=1e-6)
        )

    def test_masked_values_do_not_affect_states_and_outputs_are_zero(self):
        torch.manual_seed(11)
        model = TemporalConvNet(
            input_dim=3, hidden_dim=5, levels=2, kernel_size=2, dropout=0.0
        ).eval()
        time_mask = torch.tensor(
            [[True, True, True, False, False], [True, True, False, True, True]]
        )
        original = torch.randn(2, 5, 3)
        changed = original.clone()
        changed[~time_mask] = 999.0
        with torch.no_grad():
            original_state = model(original, time_mask)
            changed_state = model(changed, time_mask)
        self.assertTrue(
            torch.allclose(
                original_state[time_mask], changed_state[time_mask], atol=1e-6
            )
        )
        self.assertTrue(torch.equal(original_state[~time_mask], torch.zeros_like(original_state[~time_mask])))

    def test_graph_tcn_composition_has_gradients(self):
        torch.manual_seed(17)
        model = SocialGNNWithTCN(
            node_dim=5,
            edge_dim=3,
            graph_hidden_dim=8,
            temporal_hidden_dim=10,
            tcn_levels=3,
            tcn_dropout=0.0,
        )
        nodes = torch.randn(2, 9, 2, 5, requires_grad=True)
        edges = torch.randn(2, 9, 2, 2, 3, requires_grad=True)
        node_mask = torch.ones(2, 9, 2, dtype=torch.bool)
        node_mask[1, 7:] = False
        social_state, graph_state = model(
            nodes,
            edges,
            node_mask=node_mask,
            return_graph_embeddings=True,
        )
        self.assertEqual(tuple(graph_state.shape), (2, 9, 8))
        self.assertEqual(tuple(social_state.shape), (2, 9, 10))
        self.assertTrue(torch.equal(social_state[1, 7:], torch.zeros_like(social_state[1, 7:])))
        social_state.square().sum().backward()
        self.assertIsNotNone(nodes.grad)
        self.assertIsNotNone(edges.grad)


if __name__ == "__main__":
    unittest.main()
