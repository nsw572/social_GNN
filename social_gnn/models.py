"""Patch-level Social-GNN encoder with a dependency-free GINE-style layer."""

from __future__ import annotations

import torch
from torch import nn

from .graph_builder import validate_v0_inputs


class EdgeAwareGINELayer(nn.Module):
    """Small GINE-style message passing layer for dense, directed graphs."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.edge_encoder = nn.Sequential(nn.Linear(edge_dim, hidden_dim), nn.ReLU())
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.node_update = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: [B, N, D], edge_attr: [B, source, target, F].
        batch_size, number_of_nodes, _ = x.shape
        edge = self.edge_encoder(edge_attr)
        sender = x[:, :, None, :].expand(-1, -1, number_of_nodes, -1)
        messages = self.message_mlp(sender + edge)
        edge_valid = ~torch.eye(
            number_of_nodes, dtype=torch.bool, device=x.device
        ).view(1, number_of_nodes, number_of_nodes)
        edge_valid = edge_valid.expand(batch_size, -1, -1)
        if node_mask is not None:
            if node_mask.shape != (batch_size, number_of_nodes):
                raise ValueError("node_mask must have shape [B,N]")
            edge_valid = edge_valid & (
                node_mask[:, :, None] & node_mask[:, None, :]
            )
        messages = messages * edge_valid.to(dtype=messages.dtype).unsqueeze(-1)
        # Sum over sources so output index j receives all i -> j messages.
        aggregate = messages.sum(dim=1)
        output = self.node_update(torch.cat((x, aggregate), dim=-1))
        if node_mask is not None:
            output = output * node_mask.to(dtype=output.dtype).unsqueeze(-1)
        return output


class SocialV0(nn.Module):
    """GINE-style animal graph encoder producing one independent ``g_t`` per patch.

    Inputs: ``node_features [B,T,N,D]`` and ``edge_features [B,T,N,N,F]``.
    Output: ``graph_embedding [B,T,H]``. For N=1, message passing is bypassed
    and the individual feature is projected directly.
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 64, layers: int = 2) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")
        self.node_input = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.ReLU())
        self.layers = nn.ModuleList(
            EdgeAwareGINELayer(hidden_dim, edge_dim, hidden_dim) for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(layers))

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        node_mask: torch.Tensor | None = None,
        *,
        return_node_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        b, t, n, _d, _f = validate_v0_inputs(node_features, edge_features)
        x = self.node_input(node_features)
        if node_mask is not None:
            if node_mask.shape != (b, t, n):
                raise ValueError("node_mask must have shape [B,T,N]")
            # A masked node must not contribute its placeholder activation as a
            # sender during message passing. Edge inputs remain available so a
            # decoder can reconstruct the missing node from social context.
            x = x * node_mask.to(dtype=x.dtype).unsqueeze(-1)
        if n >= 2:
            for layer, norm in zip(self.layers, self.norms):
                flat_x = x.reshape(b * t, n, -1)
                flat_e = edge_features.reshape(b * t, n, n, -1)
                flat_mask = (
                    node_mask.reshape(b * t, n) if node_mask is not None else None
                )
                flat_x = norm(layer(flat_x, flat_e, node_mask=flat_mask) + flat_x)
                x = flat_x.reshape(b, t, n, -1)
                if node_mask is not None:
                    x = x * node_mask.to(dtype=x.dtype).unsqueeze(-1)
        if node_mask is None:
            graph_embedding = x.mean(dim=2)
        else:
            weights = node_mask.to(dtype=x.dtype).unsqueeze(-1)
            graph_embedding = (x * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
        if return_node_embeddings:
            return graph_embedding, x
        return graph_embedding
