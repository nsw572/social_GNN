import unittest

import torch

from social_gnn.masked_social_ssl import (
    MaskedSocialSSL,
    SocialMaskConfig,
    apply_social_masks,
    masked_social_reconstruction_loss,
)
from social_gnn.vae import SocialPatchVAE, social_patch_vae_loss


def _batch(batch_size=1, timesteps=3, nodes=2, node_dim=6):
    torch.manual_seed(7)
    return {
        "node_features": torch.randn(batch_size, timesteps, nodes, node_dim),
        "node_mask": torch.ones(batch_size, timesteps, nodes, dtype=torch.bool),
        "edge_values": torch.randn(batch_size, timesteps, nodes, nodes, 8),
        "edge_confidence": torch.ones(batch_size, timesteps, nodes, nodes, 8),
        "time_mask": torch.ones(batch_size, timesteps, dtype=torch.bool),
    }


class MaskedSocialSSLTest(unittest.TestCase):
    def test_masked_ssl_and_vae_accept_nine_dimensional_nodes(self):
        batch = _batch(node_dim=9)
        masked_model = MaskedSocialSSL(node_dim=9, hidden_dim=8, layers=1)
        masked_output = masked_model(
            batch,
            mask_config=SocialMaskConfig(
                node_mask_probability=1.0,
                edge_mask_probability=1.0,
            ),
        )
        self.assertEqual(tuple(masked_output["node_prediction"].shape), (1, 3, 2, 9))
        masked_social_reconstruction_loss(masked_output, batch)["total"].backward()

        vae_model = SocialPatchVAE(
            node_dim=9,
            number_of_nodes=2,
            hidden_dim=8,
            latent_dim=4,
            layers=1,
        )
        vae_output = vae_model(batch)
        self.assertEqual(tuple(vae_output["node_prediction"].shape), (1, 3, 2, 9))
        social_patch_vae_loss(vae_output, batch, kl_beta=1e-3)["total"].backward()

    def test_graph_bottleneck_updates_decoder_dimensions(self):
        batch = _batch()
        model = MaskedSocialSSL(
            node_dim=6,
            hidden_dim=8,
            graph_dim=4,
            layers=1,
        )
        outputs = model(
            batch,
            mask_config=SocialMaskConfig(
                node_mask_probability=0.5,
                edge_mask_probability=0.5,
            ),
        )
        self.assertEqual(tuple(outputs["g_t"].shape), (1, 3, 4))
        self.assertEqual(tuple(outputs["node_prediction"].shape), (1, 3, 2, 6))
        self.assertEqual(tuple(outputs["edge_prediction"].shape), (1, 3, 2, 2, 8))
        masked_social_reconstruction_loss(outputs, batch)["total"].backward()

    def test_symmetric_trait_masks_are_coupled_when_enabled(self):
        batch = _batch()
        masked = apply_social_masks(
            batch,
            SocialMaskConfig(
                node_mask_probability=0.0,
                edge_mask_probability=0.5,
                couple_symmetric_traits=True,
            ),
            generator=torch.Generator().manual_seed(23),
        )
        for trait in (0, 1, 4, 6, 7):
            self.assertTrue(
                torch.equal(
                    masked.edge_target_mask[:, :, 0, 1, trait],
                    masked.edge_target_mask[:, :, 1, 0, trait],
                )
            )

    def test_masking_does_not_mutate_targets_and_hides_confidence(self):
        batch = _batch()
        original_nodes = batch["node_features"].clone()
        original_values = batch["edge_values"].clone()
        original_confidence = batch["edge_confidence"].clone()
        masked = apply_social_masks(
            batch,
            SocialMaskConfig(
                node_mask_probability=1.0,
                edge_mask_probability=1.0,
            ),
        )
        self.assertTrue(torch.equal(batch["node_features"], original_nodes))
        self.assertTrue(torch.equal(batch["edge_values"], original_values))
        self.assertTrue(torch.equal(batch["edge_confidence"], original_confidence))
        self.assertTrue(masked.node_target_mask.all())
        self.assertTrue(masked.node_mask.all())
        self.assertTrue(masked.edge_target_mask[..., 0, 1, :].all())
        self.assertTrue(masked.edge_target_mask[..., 1, 0, :].all())
        hidden_confidence = masked.edge_confidence[masked.edge_target_mask]
        self.assertTrue(torch.equal(hidden_confidence, torch.zeros_like(hidden_confidence)))

    def test_padding_is_never_masked(self):
        batch = _batch(timesteps=4)
        batch["time_mask"][:, -1] = False
        batch["node_mask"][:, -1] = False
        masked = apply_social_masks(
            batch,
            SocialMaskConfig(
                node_mask_probability=1.0,
                edge_mask_probability=1.0,
                node_modality_dropout=1.0,
                edge_modality_dropout=1.0,
            ),
        )
        self.assertFalse(masked.node_target_mask[:, -1].any())
        self.assertFalse(masked.edge_target_mask[:, -1].any())
        self.assertFalse(masked.node_modality_dropped[:, -1].any())
        self.assertFalse(masked.edge_modality_dropped[:, -1].any())

    def test_whole_modalities_are_not_both_dropped_for_one_patch(self):
        batch = _batch()
        masked = apply_social_masks(
            batch,
            SocialMaskConfig(
                node_modality_dropout=1.0,
                edge_modality_dropout=1.0,
            ),
        )
        self.assertTrue(masked.node_modality_dropped.all())
        self.assertFalse(masked.edge_modality_dropped.any())

    def test_masked_model_and_loss_shapes(self):
        batch = _batch()
        model = MaskedSocialSSL(node_dim=6, hidden_dim=8, layers=1)
        outputs = model(
            batch,
            mask_config=SocialMaskConfig(
                node_mask_probability=0.5,
                edge_mask_probability=0.5,
                node_modality_dropout=0.25,
                edge_modality_dropout=0.25,
            ),
        )
        self.assertEqual(tuple(outputs["g_t"].shape), (1, 3, 8))
        self.assertEqual(tuple(outputs["node_prediction"].shape), (1, 3, 2, 6))
        self.assertEqual(tuple(outputs["edge_prediction"].shape), (1, 3, 2, 2, 8))
        losses = masked_social_reconstruction_loss(outputs, batch)
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()

    def test_vae_parallel_baseline_is_patch_local(self):
        batch = _batch()
        model = SocialPatchVAE(
            node_dim=6, number_of_nodes=2, hidden_dim=8, latent_dim=4, layers=1
        )
        outputs = model(batch)
        self.assertEqual(tuple(outputs["mu"].shape), (1, 3, 4))
        self.assertEqual(tuple(outputs["node_prediction"].shape), (1, 3, 2, 6))
        self.assertEqual(tuple(outputs["edge_prediction"].shape), (1, 3, 2, 2, 8))
        losses = social_patch_vae_loss(outputs, batch, kl_beta=1e-3)
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()


if __name__ == "__main__":
    unittest.main()
