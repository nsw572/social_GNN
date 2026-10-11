"""Patch-level masked multimodal SSL for Social-GNN.

The masking functions never mutate a stored ``SocialTrialPackage`` or a
DataLoader batch. They return corrupted inputs plus explicit target masks;
the original batch remains the reconstruction target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from .graph_builder import compose_edge_inputs
from .models import SocialV0


@dataclass(frozen=True)
class SocialMaskConfig:
    """Probabilities for patch-local node/edge masking and modality dropout."""

    node_mask_probability: float = 0.15
    edge_mask_probability: float = 0.15
    node_modality_dropout: float = 0.0
    edge_modality_dropout: float = 0.0
    fill_value: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "node_mask_probability",
            "edge_mask_probability",
            "node_modality_dropout",
            "edge_modality_dropout",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1], got {value}")


@dataclass
class MaskedSocialInputs:
    """Corrupted model inputs and masks describing what must be reconstructed."""

    node_features: torch.Tensor
    node_mask: torch.Tensor
    edge_values: torch.Tensor
    edge_confidence: torch.Tensor
    node_target_mask: torch.Tensor
    edge_target_mask: torch.Tensor
    node_modality_dropped: torch.Tensor
    edge_modality_dropped: torch.Tensor


def _random_like(
    reference: torch.Tensor,
    shape: tuple[int, ...],
    generator: torch.Generator | None,
) -> torch.Tensor:
    return torch.rand(shape, device=reference.device, generator=generator)


def _require_batch_shapes(batch: Mapping[str, torch.Tensor]) -> tuple[int, int, int, int]:
    node_features = batch["node_features"]
    node_mask = batch["node_mask"]
    edge_values = batch["edge_values"]
    edge_confidence = batch["edge_confidence"]
    if node_features.ndim != 4:
        raise ValueError("node_features must have shape [B,T,N,D]")
    b, t, n, d = node_features.shape
    if node_mask.shape != (b, t, n):
        raise ValueError("node_mask must have shape [B,T,N]")
    expected_edges = (b, t, n, n, 8)
    if tuple(edge_values.shape) != expected_edges:
        raise ValueError(f"edge_values must have shape {expected_edges}")
    if tuple(edge_confidence.shape) != expected_edges:
        raise ValueError(f"edge_confidence must have shape {expected_edges}")
    time_mask = batch.get("time_mask")
    if time_mask is not None and time_mask.shape != (b, t):
        raise ValueError("time_mask must have shape [B,T]")
    return b, t, n, d


def apply_social_masks(
    batch: Mapping[str, torch.Tensor],
    config: SocialMaskConfig | None = None,
    *,
    generator: torch.Generator | None = None,
) -> MaskedSocialInputs:
    """Create masked inputs while preserving the original batch as target.

    Node masking is per node and patch. Edge masking is per directed edge and
    trait. Modality dropout removes a whole node or edge modality for a patch.
    Padded time and invalid nodes/edges are never selected as targets.
    """

    config = config or SocialMaskConfig()
    b, t, n, _d = _require_batch_shapes(batch)
    nodes = batch["node_features"]
    original_node_mask = batch["node_mask"].bool()
    values = batch["edge_values"]
    confidence = batch["edge_confidence"]
    time_mask = batch.get("time_mask")
    if time_mask is None:
        time_mask = torch.ones((b, t), dtype=torch.bool, device=nodes.device)
    else:
        time_mask = time_mask.bool()

    valid_nodes = original_node_mask & time_mask.unsqueeze(-1)
    node_target_mask = (
        _random_like(nodes, (b, t, n), generator) < config.node_mask_probability
    ) & valid_nodes
    node_modality_dropped = (
        _random_like(nodes, (b, t), generator) < config.node_modality_dropout
    ) & time_mask
    node_target_mask |= node_modality_dropped.unsqueeze(-1) & valid_nodes

    masked_nodes = nodes.clone()
    masked_nodes[node_target_mask] = config.fill_value
    # This mask describes whether the animal exists in the stored data, not
    # whether its features were hidden for SSL. Hidden valid nodes must remain
    # in message passing so incoming social edges can reconstruct them.
    input_node_mask = original_node_mask.clone()

    pair_valid = valid_nodes.unsqueeze(-1) & valid_nodes.unsqueeze(-2)
    diagonal = torch.eye(n, dtype=torch.bool, device=nodes.device).view(1, 1, n, n, 1)
    pair_valid = pair_valid.unsqueeze(-1) & ~diagonal
    edge_target_mask = (
        _random_like(values, tuple(values.shape), generator)
        < config.edge_mask_probability
    ) & pair_valid
    edge_modality_dropped = (
        _random_like(nodes, (b, t), generator) < config.edge_modality_dropout
    ) & time_mask & ~node_modality_dropped
    edge_target_mask |= edge_modality_dropped.view(b, t, 1, 1, 1) & pair_valid

    masked_values = values.clone()
    masked_confidence = confidence.clone()
    masked_values[edge_target_mask] = config.fill_value
    # A masked edge must not advertise the confidence of its hidden target.
    masked_confidence[edge_target_mask] = 0.0

    return MaskedSocialInputs(
        node_features=masked_nodes,
        node_mask=input_node_mask,
        edge_values=masked_values,
        edge_confidence=masked_confidence,
        node_target_mask=node_target_mask,
        edge_target_mask=edge_target_mask,
        node_modality_dropped=node_modality_dropped,
        edge_modality_dropped=edge_modality_dropped,
    )


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))


class MaskedSocialSSL(nn.Module):
    """Patch-local GNN plus node/edge reconstruction heads.

    The encoder has no temporal state. ``forward`` can generate fresh masks or
    accept a precomputed ``MaskedSocialInputs`` object for deterministic tests.
    """

    def __init__(
        self,
        *,
        node_dim: int,
        hidden_dim: int = 64,
        layers: int = 2,
        decoder_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        decoder_hidden_dim = decoder_hidden_dim or hidden_dim
        self.encoder = SocialV0(node_dim=node_dim, edge_dim=16, hidden_dim=hidden_dim, layers=layers)
        self.node_mask_token = nn.Parameter(torch.zeros(node_dim))
        nn.init.normal_(self.node_mask_token, std=0.02)
        self.node_decoder = _mlp(hidden_dim * 2, decoder_hidden_dim, node_dim)
        self.edge_decoder = _mlp(hidden_dim * 3, decoder_hidden_dim, 8)

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        masked_inputs: MaskedSocialInputs | None = None,
        mask_config: SocialMaskConfig | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
        _require_batch_shapes(batch)
        masked_inputs = masked_inputs or apply_social_masks(
            batch, mask_config, generator=generator
        )
        masked_nodes = masked_inputs.node_features.clone()
        masked_nodes = torch.where(
            masked_inputs.node_target_mask.unsqueeze(-1),
            self.node_mask_token.view(1, 1, 1, -1),
            masked_nodes,
        )
        edge_features = compose_edge_inputs(
            masked_inputs.edge_values, masked_inputs.edge_confidence
        )
        graph_embedding, node_embedding = self.encoder(
            masked_nodes,
            edge_features,
            node_mask=masked_inputs.node_mask,
            return_node_embeddings=True,
        )
        b, t, n, h = node_embedding.shape
        graph_for_nodes = graph_embedding.unsqueeze(2).expand(-1, -1, n, -1)
        node_prediction = self.node_decoder(
            torch.cat((node_embedding, graph_for_nodes), dim=-1)
        )
        source = node_embedding.unsqueeze(3).expand(-1, -1, -1, n, -1)
        target = node_embedding.unsqueeze(2).expand(-1, -1, n, -1, -1)
        graph_for_edges = graph_embedding.unsqueeze(2).unsqueeze(2).expand(-1, -1, n, n, -1)
        edge_prediction = self.edge_decoder(
            torch.cat((source, target, graph_for_edges), dim=-1)
        )
        return {
            "g_t": graph_embedding,
            "node_prediction": node_prediction,
            "edge_prediction": edge_prediction,
            "masked_inputs": masked_inputs,
        }


def _weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    squared = (prediction - target).pow(2)
    expanded = weights.to(dtype=squared.dtype)
    denominator = expanded.sum() * prediction.shape[-1]
    if float(denominator.detach()) == 0.0:
        return prediction.sum() * 0.0
    return (squared * expanded.unsqueeze(-1)).sum() / denominator


def masked_social_reconstruction_loss(
    outputs: Mapping[str, Any],
    batch: Mapping[str, torch.Tensor],
    *,
    node_weight: float = 1.0,
    edge_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute confidence-aware loss only on dynamically hidden targets."""

    masked = outputs["masked_inputs"]
    time_mask = batch.get("time_mask")
    if time_mask is None:
        time_mask = torch.ones(
            batch["node_mask"].shape[:2], dtype=torch.bool, device=batch["node_mask"].device
        )
    node_weights = (
        masked.node_target_mask
        & batch["node_mask"].bool()
        & time_mask.unsqueeze(-1)
    )
    node_loss = _weighted_mse(
        outputs["node_prediction"], batch["node_features"], node_weights
    )

    edge_weights = masked.edge_target_mask & time_mask.view(
        time_mask.shape[0], time_mask.shape[1], 1, 1, 1
    )
    edge_weights = edge_weights.to(dtype=batch["edge_confidence"].dtype) * batch["edge_confidence"]
    edge_squared = (outputs["edge_prediction"] - batch["edge_values"]).pow(2)
    edge_denominator = edge_weights.sum()
    if float(edge_denominator.detach()) == 0.0:
        edge_loss = outputs["edge_prediction"].sum() * 0.0
    else:
        edge_loss = (edge_squared * edge_weights).sum() / edge_denominator
    total = node_weight * node_loss + edge_weight * edge_loss
    return {"total": total, "node": node_loss, "edge": edge_loss}
