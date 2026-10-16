"""Node-preserving graph bottlenecks and permutation-consistent SSL losses.

All decoders in this module receive only the final graph embedding ``g_t``.
Masked views pass the corrupted node tensor to both the social and node-set
branches, so the preservation branch cannot see a hidden reconstruction target.
"""

from __future__ import annotations

from itertools import permutations
from typing import Any, Mapping

import torch
from torch import nn

from .graph_builder import compose_edge_inputs
from .graph_level_ssl import (
    GRAPH_EDGE_TARGET_DIM,
    _mlp,
    _weighted_selected_mse,
    graph_node_target_dim,
    graph_level_targets,
)
from .masked_social_ssl import (
    MaskedSocialInputs,
    SocialMaskConfig,
    _require_batch_shapes,
    apply_social_masks,
)
from .models import SocialV0


NEW_GRAPH_OBJECTIVES = frozenset(
    {
        "graph_full_ssl",
        "graph_hybrid_ssl",
        "node_branch_hybrid_ssl",
        "dual_branch_hybrid_ssl",
        "graph_set_full_ssl",
        "graph_set_hybrid_ssl",
        "dual_set_hybrid_ssl",
        "dual_joint_hybrid_ssl",
    }
)
HYBRID_OBJECTIVES = frozenset(
    {
        "graph_hybrid_ssl",
        "node_branch_hybrid_ssl",
        "dual_branch_hybrid_ssl",
        "graph_set_hybrid_ssl",
        "dual_set_hybrid_ssl",
        "dual_joint_hybrid_ssl",
    }
)
FULL_ONLY_OBJECTIVES = NEW_GRAPH_OBJECTIVES - HYBRID_OBJECTIVES


class NodeSetEncoder(nn.Module):
    """Lightweight DeepSets-style encoder for an unordered two-node set."""

    def __init__(
        self,
        *,
        node_dim: int,
        graph_dim: int,
        branch_hidden_dim: int = 32,
        fusion_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.graph_dim = int(graph_dim)
        self.phi = nn.Sequential(
            nn.Linear(node_dim, branch_hidden_dim),
            nn.ReLU(),
            nn.Linear(branch_hidden_dim, branch_hidden_dim),
            nn.ReLU(),
        )
        # Mean plus absolute pair difference is exactly invariant for N=2 and
        # retains nonlinear within-animal feature combinations learned by phi.
        self.rho = nn.Sequential(
            nn.Linear(branch_hidden_dim * 2, fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, self.graph_dim),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
        *,
        return_node_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if node_features.ndim != 4 or node_features.shape[2] != 2:
            raise ValueError("NodeSetEncoder requires node_features [B,T,2,D]")
        encoded = self.phi(node_features)
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
        weight = mask.to(dtype=encoded.dtype).unsqueeze(-1)
        encoded = encoded * weight
        mean = encoded.sum(dim=2) / weight.sum(dim=2).clamp_min(1.0)
        pair_valid = mask.all(dim=2).to(dtype=encoded.dtype).unsqueeze(-1)
        difference = torch.abs(encoded[:, :, 0] - encoded[:, :, 1]) * pair_valid
        graph = self.rho(torch.cat((mean, difference), dim=-1))
        if return_node_embeddings:
            return graph, encoded
        return graph


class DualBranchSocialEncoder(nn.Module):
    """Fuse a social GINE summary with an invariant node-set summary."""

    def __init__(
        self,
        *,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        graph_dim: int,
        layers: int,
        number_of_nodes: int,
        readout: str = "mean",
        branch_hidden_dim: int = 32,
        fusion_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if readout not in {"mean", "soft_role"}:
            raise ValueError("dual branch requires a permutation-invariant readout")
        self.graph_dim = int(graph_dim)
        self.social = SocialV0(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            graph_dim=graph_dim,
            layers=layers,
            readout=readout,
            number_of_nodes=number_of_nodes,
        )
        self.node = NodeSetEncoder(
            node_dim=node_dim,
            graph_dim=graph_dim,
            branch_hidden_dim=branch_hidden_dim,
            fusion_hidden_dim=fusion_hidden_dim,
        )
        self.fusion = nn.Sequential(
            nn.Linear(graph_dim * 2, fusion_hidden_dim),
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
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        social_graph, social_nodes = self.social(
            node_features,
            edge_features,
            node_mask=node_mask,
            return_node_embeddings=True,
        )
        node_graph = self.node(
            node_features,
            edge_features,
            node_mask=node_mask,
        )
        graph = self.fusion(torch.cat((social_graph, node_graph), dim=-1))
        if return_node_embeddings:
            return graph, social_nodes
        return graph


def objective_architecture(objective: str) -> tuple[str, str]:
    """Return ``(architecture, target_mode)`` for one new objective."""

    if objective not in NEW_GRAPH_OBJECTIVES:
        raise ValueError(f"unsupported node-preservation objective {objective!r}")
    architecture = (
        "node_set"
        if objective == "node_branch_hybrid_ssl"
        else "dual"
        if objective.startswith("dual_")
        else "gnn"
    )
    target_mode = (
        "joint_set"
        if objective == "dual_joint_hybrid_ssl"
        else "node_set"
        if "set_" in objective
        else "summary"
    )
    return architecture, target_mode


class NodePreservingGraphSSL(nn.Module):
    """Graph-only decoders with selectable social/node encoder and set target."""

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
        architecture: str = "gnn",
        target_mode: str = "summary",
        branch_hidden_dim: int = 32,
        fusion_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if number_of_nodes != 2:
            raise ValueError("node-preservation SSL currently requires N=2")
        if architecture == "gnn":
            self.encoder: nn.Module = SocialV0(
                node_dim=node_dim,
                edge_dim=16,
                hidden_dim=hidden_dim,
                graph_dim=graph_dim,
                layers=layers,
                readout=readout,
                number_of_nodes=number_of_nodes,
            )
        elif architecture == "node_set":
            self.encoder = NodeSetEncoder(
                node_dim=node_dim,
                graph_dim=graph_dim,
                branch_hidden_dim=branch_hidden_dim,
                fusion_hidden_dim=fusion_hidden_dim,
            )
        elif architecture == "dual":
            self.encoder = DualBranchSocialEncoder(
                node_dim=node_dim,
                edge_dim=16,
                hidden_dim=hidden_dim,
                graph_dim=graph_dim,
                layers=layers,
                number_of_nodes=number_of_nodes,
                readout=readout,
                branch_hidden_dim=branch_hidden_dim,
                fusion_hidden_dim=fusion_hidden_dim,
            )
        else:
            raise ValueError("architecture must be 'gnn', 'node_set', or 'dual'")
        if target_mode not in {"summary", "node_set", "joint_set"}:
            raise ValueError("target_mode must be summary, node_set, or joint_set")
        self.architecture = architecture
        self.target_mode = target_mode
        self.node_dim = int(node_dim)
        self.node_target_dim = graph_node_target_dim(self.node_dim)
        self.node_mask_token = nn.Parameter(torch.zeros(node_dim))
        nn.init.normal_(self.node_mask_token, std=0.02)
        node_output = self.node_target_dim if target_mode == "summary" else 2 * node_dim
        edge_output = 2 * 2 * 8 if target_mode == "joint_set" else GRAPH_EDGE_TARGET_DIM
        self.node_decoder = _mlp(graph_dim, decoder_hidden_dim, node_output)
        self.edge_decoder = _mlp(graph_dim, decoder_hidden_dim, edge_output)

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
        view: str = "masked",
        masked_inputs: MaskedSocialInputs | None = None,
        mask_config: SocialMaskConfig | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
        _require_batch_shapes(batch)
        if view not in {"masked", "full"}:
            raise ValueError("view must be 'masked' or 'full'")
        if view == "masked":
            masked = masked_inputs or apply_social_masks(
                batch, mask_config, generator=generator
            )
            nodes = torch.where(
                masked.node_target_mask.unsqueeze(-1),
                self.node_mask_token.view(1, 1, 1, -1),
                masked.node_features,
            )
            values = masked.edge_values
            confidence = masked.edge_confidence
            node_mask = masked.node_mask
        else:
            masked = None
            nodes = batch["node_features"]
            values = batch["edge_values"]
            confidence = batch["edge_confidence"]
            node_mask = batch["node_mask"]
        graph = self.encoder(
            nodes,
            compose_edge_inputs(values, confidence),
            node_mask=node_mask,
        )
        node_prediction = self.node_decoder(graph)
        edge_prediction = self.edge_decoder(graph)
        if self.target_mode != "summary":
            node_prediction = node_prediction.reshape(*graph.shape[:2], 2, self.node_dim)
        if self.target_mode == "joint_set":
            edge_prediction = edge_prediction.reshape(*graph.shape[:2], 2, 2, 8)
        return {
            "g_t": graph,
            "node_prediction": node_prediction,
            "edge_prediction": edge_prediction,
            "masked_inputs": masked,
            "view": view,
            "target_mode": self.target_mode,
        }


def _valid_masks(
    batch: Mapping[str, torch.Tensor],
    masked: MaskedSocialInputs | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nodes = batch["node_features"]
    time_mask = batch.get("time_mask")
    if time_mask is None:
        time_mask = torch.ones(nodes.shape[:2], dtype=torch.bool, device=nodes.device)
    node_valid = batch["node_mask"].bool() & time_mask.bool().unsqueeze(-1)
    n = nodes.shape[2]
    diagonal = torch.eye(n, dtype=torch.bool, device=nodes.device).view(1, 1, n, n, 1)
    edge_valid = (
        node_valid.unsqueeze(3).unsqueeze(-1)
        & node_valid.unsqueeze(2).unsqueeze(-1)
        & ~diagonal
    )
    if masked is None:
        node_selected = node_valid
        edge_selected = edge_valid.expand_as(batch["edge_values"])
    else:
        node_selected = masked.node_target_mask.bool() & node_valid
        edge_selected = masked.edge_target_mask.bool() & edge_valid
    return node_valid, node_selected, edge_selected


def _summary_loss(
    outputs: Mapping[str, Any],
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    masked = outputs["masked_inputs"]
    if masked is None:
        node_valid, node_selected, edge_selected = _valid_masks(batch, None)
        masked = MaskedSocialInputs(
            node_features=batch["node_features"],
            node_mask=batch["node_mask"],
            edge_values=batch["edge_values"],
            edge_confidence=batch["edge_confidence"],
            node_target_mask=node_selected,
            edge_target_mask=edge_selected,
            node_modality_dropped=torch.zeros_like(node_valid[..., 0]),
            edge_modality_dropped=torch.zeros_like(node_valid[..., 0]),
        )
    targets = graph_level_targets(batch, masked)
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
    return node_loss, edge_loss


def _candidate_set_errors(
    outputs: Mapping[str, Any],
    batch: Mapping[str, torch.Tensor],
    permutation: tuple[int, int],
    *,
    include_directed_edges: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    masked = outputs["masked_inputs"]
    _node_valid, node_selected, edge_selected = _valid_masks(batch, masked)
    index = torch.as_tensor(permutation, dtype=torch.long, device=batch["node_features"].device)
    target_nodes = batch["node_features"].index_select(2, index)
    selected_nodes = node_selected.index_select(2, index).unsqueeze(-1)
    node_numerator = (
        torch.square(outputs["node_prediction"] - target_nodes)
        * selected_nodes.to(dtype=target_nodes.dtype)
    ).sum(dim=(2, 3))
    node_denominator = selected_nodes.to(dtype=target_nodes.dtype).sum(dim=(2, 3))
    node_denominator = node_denominator * target_nodes.shape[-1]
    node_error = node_numerator / node_denominator.clamp_min(1.0)

    if include_directed_edges:
        target_edges = batch["edge_values"].index_select(2, index).index_select(3, index)
        confidence = batch["edge_confidence"].index_select(2, index).index_select(3, index)
        selected_edges = edge_selected.index_select(2, index).index_select(3, index)
        effective = selected_edges.to(dtype=confidence.dtype) * confidence
        edge_numerator = (
            torch.square(outputs["edge_prediction"] - target_edges) * effective
        ).sum(dim=(2, 3, 4))
        edge_denominator = effective.sum(dim=(2, 3, 4))
        edge_error = edge_numerator / edge_denominator.clamp_min(1.0)
    else:
        edge_error = torch.zeros_like(node_error)
        edge_denominator = torch.zeros_like(node_denominator)
    return node_error, node_denominator, edge_error, edge_denominator


def node_preserving_reconstruction_loss(
    outputs: Mapping[str, Any],
    batch: Mapping[str, torch.Tensor],
    *,
    node_weight: float = 1.0,
    edge_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Loss for summary, unordered-set, or jointly matched node-edge targets."""

    mode = str(outputs["target_mode"])
    if mode == "summary":
        node_loss, edge_loss = _summary_loss(outputs, batch)
    elif mode == "node_set":
        candidates = [
            _candidate_set_errors(outputs, batch, value, include_directed_edges=False)
            for value in permutations(range(2))
        ]
        first, second = candidates
        choose_first = first[0] <= second[0]
        selected_error = torch.where(choose_first, first[0], second[0])
        selected_count = torch.where(choose_first, first[1], second[1])
        active = selected_count > 0
        node_loss = (
            selected_error[active].mean()
            if bool(active.any())
            else outputs["node_prediction"].sum() * 0.0
        )
        # The edge summary remains permutation invariant and is trained
        # independently; role correspondence is tested by joint_set below.
        summary_outputs = dict(outputs)
        summary_outputs["node_prediction"] = torch.zeros(
            *outputs["g_t"].shape[:2],
            graph_node_target_dim(batch["node_features"].shape[-1]),
            dtype=outputs["g_t"].dtype,
            device=outputs["g_t"].device,
        )
        _unused, edge_loss = _summary_loss(summary_outputs, batch)
    else:
        candidates = [
            _candidate_set_errors(outputs, batch, value, include_directed_edges=True)
            for value in permutations(range(2))
        ]
        combined = []
        for node_error, node_count, edge_error, edge_count in candidates:
            combined.append(
                node_weight * node_error * (node_count > 0).to(node_error.dtype)
                + edge_weight * edge_error * (edge_count > 0).to(edge_error.dtype)
            )
        choose_first = combined[0] <= combined[1]
        node_error = torch.where(choose_first, candidates[0][0], candidates[1][0])
        node_count = torch.where(choose_first, candidates[0][1], candidates[1][1])
        edge_error = torch.where(choose_first, candidates[0][2], candidates[1][2])
        edge_count = torch.where(choose_first, candidates[0][3], candidates[1][3])
        node_active = node_count > 0
        edge_active = edge_count > 0
        node_loss = (
            node_error[node_active].mean()
            if bool(node_active.any())
            else outputs["node_prediction"].sum() * 0.0
        )
        edge_loss = (
            edge_error[edge_active].mean()
            if bool(edge_active.any())
            else outputs["edge_prediction"].sum() * 0.0
        )
    return {
        "total": node_weight * node_loss + edge_weight * edge_loss,
        "node": node_loss,
        "edge": edge_loss,
    }


def hybrid_reconstruction_loss(
    masked_outputs: Mapping[str, Any],
    full_outputs: Mapping[str, Any],
    batch: Mapping[str, torch.Tensor],
    *,
    preserve_weight: float = 1.0,
    masked_weight: float = 1.0,
    node_weight: float = 1.0,
    edge_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Combine normalized full-input preservation and masked reconstruction."""

    masked = node_preserving_reconstruction_loss(
        masked_outputs, batch, node_weight=node_weight, edge_weight=edge_weight
    )
    full = node_preserving_reconstruction_loss(
        full_outputs, batch, node_weight=node_weight, edge_weight=edge_weight
    )
    total = preserve_weight * full["total"] + masked_weight * masked["total"]
    return {
        "total": total,
        "preserve": full["total"],
        "masked": masked["total"],
        "node_full": full["node"],
        "edge_full": full["edge"],
        "node_masked": masked["node"],
        "edge_masked": masked["edge"],
    }
