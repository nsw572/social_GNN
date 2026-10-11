"""Parallel patch-level VAE baseline for Social-GNN.

This is an independent self-reconstruction experiment. It uses the same
patch-local SocialV0 encoder and has no TCN/RNN/Transformer stage.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from .graph_builder import compose_edge_inputs
from .masked_social_ssl import _weighted_mse, _mlp
from .models import SocialV0


class SocialPatchVAE(nn.Module):
    """VAE that reconstructs all node features and edge values per patch."""

    def __init__(
        self,
        *,
        node_dim: int,
        number_of_nodes: int = 2,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        layers: int = 2,
        decoder_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if number_of_nodes < 1:
            raise ValueError("number_of_nodes must be >= 1")
        self.number_of_nodes = number_of_nodes
        self.node_dim = node_dim
        self.latent_dim = latent_dim
        decoder_hidden_dim = decoder_hidden_dim or hidden_dim
        self.encoder = SocialV0(node_dim=node_dim, edge_dim=16, hidden_dim=hidden_dim, layers=layers)
        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)
        self.node_decoder = _mlp(
            latent_dim, decoder_hidden_dim, number_of_nodes * node_dim
        )
        self.edge_decoder = _mlp(latent_dim, decoder_hidden_dim, number_of_nodes * number_of_nodes * 8)

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        sample: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        node_features = batch["node_features"]
        if node_features.ndim != 4 or node_features.shape[2] != self.number_of_nodes:
            raise ValueError(
                "SocialPatchVAE expects node_features [B,T,N,D] with "
                f"N={self.number_of_nodes}"
            )
        edge_features = compose_edge_inputs(
            batch["edge_values"], batch["edge_confidence"]
        )
        graph_embedding = self.encoder(
            node_features, edge_features, node_mask=batch.get("node_mask")
        )
        mu = self.to_mu(graph_embedding)
        logvar = self.to_logvar(graph_embedding).clamp(min=-20.0, max=20.0)
        if sample is None:
            sample = self.training
        if sample:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        b, t, _ = z.shape
        node_prediction = self.node_decoder(z).reshape(
            b, t, self.number_of_nodes, self.node_dim
        )
        edge_prediction = self.edge_decoder(z).reshape(
            b, t, self.number_of_nodes, self.number_of_nodes, 8
        )
        return {
            "g_t": graph_embedding,
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "node_prediction": node_prediction,
            "edge_prediction": edge_prediction,
        }


def social_patch_vae_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    kl_beta: float = 1e-3,
) -> dict[str, torch.Tensor]:
    """Return reconstruction, KL, and total losses for the VAE baseline."""

    time_mask = batch.get("time_mask")
    if time_mask is None:
        time_mask = torch.ones(
            batch["node_mask"].shape[:2], dtype=torch.bool, device=batch["node_mask"].device
        )
    node_weights = batch["node_mask"].bool() & time_mask.unsqueeze(-1)
    node_loss = _weighted_mse(
        outputs["node_prediction"], batch["node_features"], node_weights
    )
    edge_weights = (
        time_mask.view(time_mask.shape[0], time_mask.shape[1], 1, 1, 1)
        & batch["node_mask"].bool().unsqueeze(-1).unsqueeze(-1)
        & batch["node_mask"].bool().unsqueeze(-2).unsqueeze(-1)
    ).to(dtype=batch["edge_confidence"].dtype)
    edge_weights = edge_weights * batch["edge_confidence"]
    edge_squared = (outputs["edge_prediction"] - batch["edge_values"]).pow(2)
    edge_denominator = edge_weights.sum()
    if float(edge_denominator.detach()) == 0.0:
        edge_loss = outputs["edge_prediction"].sum() * 0.0
    else:
        edge_loss = (edge_squared * edge_weights).sum() / edge_denominator

    valid_time = time_mask.to(dtype=outputs["mu"].dtype).unsqueeze(-1)
    kl_per_element = -0.5 * (
        1.0 + outputs["logvar"] - outputs["mu"].pow(2) - outputs["logvar"].exp()
    )
    kl_denominator = valid_time.sum() * outputs["mu"].shape[-1]
    if float(kl_denominator.detach()) == 0.0:
        kl_loss = outputs["mu"].sum() * 0.0
    else:
        kl_loss = (kl_per_element * valid_time).sum() / kl_denominator
    reconstruction = node_loss + edge_loss
    return {
        "total": reconstruction + float(kl_beta) * kl_loss,
        "reconstruction": reconstruction,
        "node": node_loss,
        "edge": edge_loss,
        "kl": kl_loss,
    }
