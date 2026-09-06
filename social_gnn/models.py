"""Minimal Social-V0 encoder with a dependency-free GINE-style layer."""

from __future__ import annotations

import torch
from torch import nn

from .graph_builder import validate_v0_inputs
from .temporal import TemporalConvNet


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

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D], edge_attr: [B, N, N, F], edge_attr[i,j] means i -> j.
        edge = self.edge_encoder(edge_attr)
        sender = x[:, None, :, :].expand(-1, x.shape[1], -1, -1)
        messages = self.message_mlp(sender + edge)
        aggregate = messages.sum(dim=2)
        return self.node_update(torch.cat((x, aggregate), dim=-1))


class SocialV0(nn.Module):
    """GINE-style animal graph encoder producing one embedding per time window.

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
    ) -> torch.Tensor:
        b, t, n, _d, _f = validate_v0_inputs(node_features, edge_features)
        x = self.node_input(node_features)
        if n >= 2:
            for layer, norm in zip(self.layers, self.norms):
                flat_x = x.reshape(b * t, n, -1)
                flat_e = edge_features.reshape(b * t, n, n, -1)
                flat_x = norm(layer(flat_x, flat_e) + flat_x)
                x = flat_x.reshape(b, t, n, -1)
        if node_mask is None:
            return x.mean(dim=2)
        if node_mask.shape != (b, t, n):
            raise ValueError("node_mask must have shape [B,T,N]")
        weights = node_mask.to(dtype=x.dtype).unsqueeze(-1)
        return (x * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)


class SocialGNNWithTCN(nn.Module):
    """Compose patch-wise graph encoding with causal temporal modelling.

    The returned ``social_state [B,T,D_s]`` is the ``s_t`` interface expected
    by a future SSL decoder. This module intentionally does not define a
    decoder target or loss before the upstream SSL objective is specified.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        *,
        graph_hidden_dim: int = 64,
        graph_layers: int = 2,
        temporal_hidden_dim: int = 64,
        tcn_levels: int = 4,
        tcn_kernel_size: int = 3,
        tcn_dilation_base: int = 2,
        tcn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.graph_encoder = SocialV0(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=graph_hidden_dim,
            layers=graph_layers,
        )
        self.temporal_encoder = TemporalConvNet(
            input_dim=graph_hidden_dim,
            hidden_dim=temporal_hidden_dim,
            levels=tcn_levels,
            kernel_size=tcn_kernel_size,
            dilation_base=tcn_dilation_base,
            dropout=tcn_dropout,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        node_mask: torch.Tensor | None = None,
        time_mask: torch.Tensor | None = None,
        *,
        return_graph_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        graph_embeddings = self.graph_encoder(
            node_features, edge_features, node_mask=node_mask
        )
        if time_mask is None and node_mask is not None:
            time_mask = node_mask.to(dtype=torch.bool).any(dim=2)
        social_state = self.temporal_encoder(graph_embeddings, time_mask=time_mask)
        if return_graph_embeddings:
            return social_state, graph_embeddings
        return social_state
