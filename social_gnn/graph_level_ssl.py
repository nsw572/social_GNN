"""Graph-bottleneck SSL and fusion baselines for Social-GNN.

Unlike :mod:`social_gnn.masked_social_ssl`, every decoder in this module sees
only the final graph representation ``g_t``.  The reconstruction targets are
explicitly permutation-invariant for the two-animal graph: node means and
absolute differences, symmetric edge traits once, and mean/absolute
directional asymmetry for directed traits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from .graph_builder import compose_edge_inputs
from .masked_social_ssl import (
    MaskedSocialInputs,
    SocialMaskConfig,
    _require_batch_shapes,
    apply_social_masks,
)
from .models import SocialV0


SYMMETRIC_TRAITS = (0, 1, 4, 6, 7)
DIRECTED_TRAITS = (2, 3, 5)
GRAPH_EDGE_TARGET_DIM = 11


def graph_node_target_dim(node_dim: int) -> int:
    """Return the invariant node-summary width (mean + absolute difference)."""

    node_dim = int(node_dim)
    if node_dim < 1:
        raise ValueError("node_dim must be positive")
    return 2 * node_dim


@dataclass(frozen=True)
class GraphLevelTargets:
    node_values: torch.Tensor
    node_weights: torch.Tensor
    node_selected: torch.Tensor
    edge_values: torch.Tensor
    edge_weights: torch.Tensor
    edge_selected: torch.Tensor


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


def graph_level_targets(
    batch: Mapping[str, torch.Tensor], masked: MaskedSocialInputs
) -> GraphLevelTargets:
    """Build invariant targets and confidence/mask weights for N=2."""

    _b, _t, n, node_dim = _require_batch_shapes(batch)
    if n != 2:
        raise ValueError("graph-level target v1 requires exactly two animals")
    node_target_dim = graph_node_target_dim(node_dim)
    nodes = batch["node_features"]
    node_values = torch.cat(
        (
            nodes.mean(dim=2),
            torch.abs(nodes[:, :, 0] - nodes[:, :, 1]),
        ),
        dim=-1,
    )
    time_mask = batch.get("time_mask")
    if time_mask is None:
        time_mask = torch.ones(
            nodes.shape[:2], dtype=torch.bool, device=nodes.device
        )
    pair_valid = batch["node_mask"].bool().all(dim=2) & time_mask.bool()
    node_selected_patch = masked.node_target_mask.any(dim=2) & pair_valid
    node_selected = node_selected_patch.unsqueeze(-1).expand(
        -1, -1, node_target_dim
    )
    node_weights = pair_valid.to(dtype=nodes.dtype).unsqueeze(-1).expand_as(
        node_values
    )

    values = batch["edge_values"]
    confidence = batch["edge_confidence"]
    target_mask = masked.edge_target_mask
    edge_columns: list[torch.Tensor] = []
    confidence_columns: list[torch.Tensor] = []
    selected_columns: list[torch.Tensor] = []
    for trait in SYMMETRIC_TRAITS:
        edge_columns.append(0.5 * (values[:, :, 0, 1, trait] + values[:, :, 1, 0, trait]))
        confidence_columns.append(
            0.5
            * (
                confidence[:, :, 0, 1, trait]
                + confidence[:, :, 1, 0, trait]
            )
        )
        selected_columns.append(
            target_mask[:, :, 0, 1, trait] | target_mask[:, :, 1, 0, trait]
        )
    for trait in DIRECTED_TRAITS:
        forward = values[:, :, 0, 1, trait]
        reverse = values[:, :, 1, 0, trait]
        forward_conf = confidence[:, :, 0, 1, trait]
        reverse_conf = confidence[:, :, 1, 0, trait]
        selected = (
            target_mask[:, :, 0, 1, trait]
            | target_mask[:, :, 1, 0, trait]
        )
        edge_columns.extend((0.5 * (forward + reverse), torch.abs(forward - reverse)))
        confidence_columns.extend(
            (0.5 * (forward_conf + reverse_conf), torch.minimum(forward_conf, reverse_conf))
        )
        selected_columns.extend((selected, selected))
    edge_values = torch.stack(edge_columns, dim=-1)
    edge_weights = torch.stack(confidence_columns, dim=-1)
    edge_selected = torch.stack(selected_columns, dim=-1) & pair_valid.unsqueeze(-1)
    edge_weights = edge_weights * pair_valid.to(dtype=edge_weights.dtype).unsqueeze(-1)
    return GraphLevelTargets(
        node_values=node_values,
        node_weights=node_weights,
        node_selected=node_selected,
        edge_values=edge_values,
        edge_weights=edge_weights,
        edge_selected=edge_selected,
    )


class MLPFusionEncoder(nn.Module):
    """Non-graph baseline receiving the same ordered node and edge inputs."""

    def __init__(
        self,
        *,
        node_dim: int,
        number_of_nodes: int,
        edge_dim: int,
        graph_dim: int,
        fusion_hidden_dim: int = 192,
    ) -> None:
        super().__init__()
        if number_of_nodes != 2:
            raise ValueError("MLP fusion baseline v1 requires N=2")
        self.number_of_nodes = number_of_nodes
        self.graph_dim = graph_dim
        input_dim = number_of_nodes * node_dim + 2 * edge_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, graph_dim),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        node_mask: torch.Tensor | None = None,
        *,
        return_node_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, None]:
        if node_features.ndim != 4 or edge_features.ndim != 5:
            raise ValueError("MLP fusion expects [B,T,N,D] nodes and [B,T,N,N,F] edges")
        if node_features.shape[2] != 2 or edge_features.shape[2:4] != (2, 2):
            raise ValueError("MLP fusion baseline v1 requires N=2")
        nodes = node_features
        if node_mask is not None:
            nodes = nodes * node_mask.to(dtype=nodes.dtype).unsqueeze(-1)
        directed_edges = torch.cat(
            (edge_features[:, :, 0, 1], edge_features[:, :, 1, 0]), dim=-1
        )
        inputs = torch.cat((nodes.flatten(2), directed_edges), dim=-1)
        graph = self.network(inputs)
        if return_node_embeddings:
            return graph, None
        return graph


def swap_two_animal_graph(
    node_features: torch.Tensor,
    edge_features: torch.Tensor,
    node_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Apply one consistent two-animal swap to nodes, edges, and validity."""

    if node_features.ndim != 4 or node_features.shape[2] != 2:
        raise ValueError("node_features must have shape [B,T,2,D]")
    if edge_features.ndim != 5 or edge_features.shape[2:4] != (2, 2):
        raise ValueError("edge_features must have shape [B,T,2,2,F]")
    if node_mask is not None and node_mask.shape != node_features.shape[:3]:
        raise ValueError("node_mask must have shape [B,T,2]")
    index = torch.as_tensor((1, 0), dtype=torch.long, device=node_features.device)
    swapped_nodes = node_features.index_select(2, index)
    swapped_edges = edge_features.index_select(2, index).index_select(3, index)
    swapped_mask = None if node_mask is None else node_mask.index_select(2, index)
    return swapped_nodes, swapped_edges, swapped_mask


class SymmetrizedMLPFusionEncoder(nn.Module):
    """Exact permutation-invariant, non-message-passing two-animal baseline.

    One shared ordered MLP evaluates the original graph and a consistently
    swapped graph. Their average is invariant to exchanging both animal axes.
    Only off-diagonal social edges are flattened; self edges are excluded.
    Node and pair-validity masks are explicit inputs, while per-trait edge
    confidence remains in ``edge_features``.
    """

    def __init__(
        self,
        *,
        node_dim: int,
        number_of_nodes: int,
        edge_dim: int,
        graph_dim: int,
        fusion_hidden_dim: int = 160,
    ) -> None:
        super().__init__()
        if number_of_nodes != 2:
            raise ValueError("symmetrized MLP baseline requires N=2")
        self.number_of_nodes = int(number_of_nodes)
        self.graph_dim = int(graph_dim)
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        input_dim = 2 * self.node_dim + 2 + 2 * self.edge_dim + 2
        self.network = nn.Sequential(
            nn.Linear(input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, self.graph_dim),
        )

    def _ordered_inputs(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        node_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if node_features.ndim != 4 or node_features.shape[2] != 2:
            raise ValueError("symmetrized MLP expects nodes [B,T,2,D]")
        if edge_features.ndim != 5 or edge_features.shape[2:4] != (2, 2):
            raise ValueError("symmetrized MLP expects edges [B,T,2,2,F]")
        if node_features.shape[-1] != self.node_dim:
            raise ValueError(
                f"expected node dimension {self.node_dim}, got {node_features.shape[-1]}"
            )
        if edge_features.shape[-1] != self.edge_dim:
            raise ValueError(
                f"expected edge dimension {self.edge_dim}, got {edge_features.shape[-1]}"
            )
        if node_mask is None:
            mask = torch.ones(
                node_features.shape[:3],
                dtype=torch.bool,
                device=node_features.device,
            )
        else:
            if node_mask.shape != node_features.shape[:3]:
                raise ValueError("node_mask must have shape [B,T,2]")
            mask = node_mask.bool()
        nodes = node_features * mask.to(dtype=node_features.dtype).unsqueeze(-1)
        directed_edges = torch.cat(
            (edge_features[:, :, 0, 1], edge_features[:, :, 1, 0]), dim=-1
        )
        pair_valid = mask.all(dim=2, keepdim=True).to(dtype=node_features.dtype)
        directed_valid = pair_valid.expand(-1, -1, 2)
        directed_edges = directed_edges * pair_valid.repeat(
            1, 1, 2 * self.edge_dim
        )
        return torch.cat(
            (
                nodes.flatten(2),
                mask.to(dtype=node_features.dtype),
                directed_edges,
                directed_valid,
            ),
            dim=-1,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        node_mask: torch.Tensor | None = None,
        *,
        return_node_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, None]:
        original = self._ordered_inputs(node_features, edge_features, node_mask)
        swapped_nodes, swapped_edges, swapped_mask = swap_two_animal_graph(
            node_features, edge_features, node_mask
        )
        swapped = self._ordered_inputs(swapped_nodes, swapped_edges, swapped_mask)
        graph = 0.5 * (self.network(original) + self.network(swapped))
        if return_node_embeddings:
            return graph, None
        return graph


class GraphLevelSocialSSL(nn.Module):
    """Masked SSL whose decoders receive only the final graph bottleneck."""

    def __init__(
        self,
        *,
        node_dim: int,
        number_of_nodes: int = 2,
        hidden_dim: int = 64,
        graph_dim: int = 64,
        layers: int = 2,
        decoder_hidden_dim: int = 64,
        readout: str = "mean",
        backbone: str = "gnn",
        fusion_hidden_dim: int = 192,
    ) -> None:
        super().__init__()
        if backbone == "gnn":
            self.encoder: nn.Module = SocialV0(
                node_dim=node_dim,
                edge_dim=16,
                hidden_dim=hidden_dim,
                graph_dim=graph_dim,
                layers=layers,
                readout=readout,
                number_of_nodes=number_of_nodes,
            )
        elif backbone == "mlp":
            self.encoder = MLPFusionEncoder(
                node_dim=node_dim,
                number_of_nodes=number_of_nodes,
                edge_dim=16,
                graph_dim=graph_dim,
                fusion_hidden_dim=fusion_hidden_dim,
            )
        else:
            raise ValueError("backbone must be 'gnn' or 'mlp'")
        self.backbone = backbone
        self.node_dim = int(node_dim)
        self.node_target_dim = graph_node_target_dim(self.node_dim)
        self.node_mask_token = nn.Parameter(torch.zeros(node_dim))
        nn.init.normal_(self.node_mask_token, std=0.02)
        self.node_decoder = _mlp(
            graph_dim, decoder_hidden_dim, self.node_target_dim
        )
        self.edge_decoder = _mlp(
            graph_dim, decoder_hidden_dim, GRAPH_EDGE_TARGET_DIM
        )

    def encode_batch(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.encoder(
            batch["node_features"],
            compose_edge_inputs(batch["edge_values"], batch["edge_confidence"]),
            node_mask=batch["node_mask"],
        )

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        masked_inputs: MaskedSocialInputs | None = None,
        mask_config: SocialMaskConfig | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
        _require_batch_shapes(batch)
        masked = masked_inputs or apply_social_masks(
            batch, mask_config, generator=generator
        )
        nodes = torch.where(
            masked.node_target_mask.unsqueeze(-1),
            self.node_mask_token.view(1, 1, 1, -1),
            masked.node_features,
        )
        graph = self.encoder(
            nodes,
            compose_edge_inputs(masked.edge_values, masked.edge_confidence),
            node_mask=masked.node_mask,
        )
        return {
            "g_t": graph,
            "node_prediction": self.node_decoder(graph),
            "edge_prediction": self.edge_decoder(graph),
            "masked_inputs": masked,
        }


def _weighted_selected_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    selected: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    effective = selected.to(dtype=prediction.dtype) * weight.to(dtype=prediction.dtype)
    denominator = effective.sum()
    if float(denominator.detach()) == 0.0:
        return prediction.sum() * 0.0
    return (torch.square(prediction - target) * effective).sum() / denominator


def graph_level_reconstruction_loss(
    outputs: Mapping[str, Any],
    batch: Mapping[str, torch.Tensor],
    *,
    node_weight: float = 1.0,
    edge_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Loss on invariant graph targets, reached exclusively through ``g_t``."""

    targets = graph_level_targets(batch, outputs["masked_inputs"])
    node_loss = _weighted_selected_mse(
        outputs["node_prediction"],
        targets.node_values,
        targets.node_selected,
        targets.node_weights,
    )
    edge_loss = _weighted_selected_mse(
        outputs["edge_prediction"],
        targets.edge_values,
        targets.edge_selected,
        targets.edge_weights,
    )
    return {
        "total": node_weight * node_loss + edge_weight * edge_loss,
        "node": node_loss,
        "edge": edge_loss,
    }
