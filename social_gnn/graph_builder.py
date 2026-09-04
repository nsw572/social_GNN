"""Utilities for constructing animal-level directed graphs."""

from __future__ import annotations

import torch


def complete_directed_edge_index(
    n_animals: int, *, self_loops: bool = False, device: torch.device | None = None
) -> torch.Tensor:
    """Return ``[2, E]`` source/target indices for a complete directed graph."""
    if n_animals < 1:
        raise ValueError("n_animals must be >= 1")
    nodes = torch.arange(n_animals, device=device)
    source = nodes[:, None].expand(n_animals, n_animals).reshape(-1)
    target = nodes[None, :].expand(n_animals, n_animals).reshape(-1)
    if not self_loops:
        keep = source != target
        source, target = source[keep], target[keep]
    return torch.stack((source, target))


def compose_edge_inputs(
    edge_value: torch.Tensor,
    edge_confidence: torch.Tensor,
    edge_coverage: torch.Tensor,
) -> torch.Tensor:
    """Concatenate separate edge channels without multiplying values by confidence.

    Inputs share shape ``[..., 8]``. The returned ``[..., 24]`` channel order is
    value, confidence, then coverage.
    """
    if not edge_value.shape == edge_confidence.shape == edge_coverage.shape:
        raise ValueError("edge value, confidence, and coverage shapes must match")
    if edge_value.ndim < 2:
        raise ValueError("edge inputs must include edge and trait dimensions")
    if edge_value.shape[-1] != 8:
        raise ValueError("the V0 social edge contract contains exactly 8 traits")
    return torch.cat((edge_value, edge_confidence, edge_coverage), dim=-1)


def validate_v0_inputs(
    node_features: torch.Tensor, edge_features: torch.Tensor
) -> tuple[int, int, int, int, int]:
    """Validate and return ``B, T, N, D_node, D_edge``.

    ``node_features`` is ``[B, T, N, D_node]`` and ``edge_features`` is
    ``[B, T, N, N, D_edge]``. The last two dimensions retain directed
    relational information: ``edge_features[..., i, j, :]`` is i -> j.
    """
    if node_features.ndim != 4:
        raise ValueError("node_features must have shape [B, T, N, D]")
    if edge_features.ndim != 5:
        raise ValueError("edge_features must have shape [B, T, N, N, F]")
    b, t, n, d = node_features.shape
    if edge_features.shape[:4] != (b, t, n, n):
        raise ValueError("edge_features must align with [B, T, N, N]")
    return b, t, n, d, edge_features.shape[-1]
