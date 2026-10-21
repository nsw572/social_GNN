"""Reproducible patch-level Social-GNN SSL experiments.

This module is deliberately downstream of ``SocialTrialPackage``. It never
rewrites node or edge NPZ files: normalization and edge ablations are applied
to in-memory copies after strict node/edge alignment validation.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml

from .data import (
    SocialTrialDataset,
    SocialTrialPackage,
    build_social_dataloader,
)
from .graph_builder import compose_edge_inputs
from .graph_level_ssl import (
    GraphLevelSocialSSL,
    graph_level_reconstruction_loss,
)
from .masked_social_ssl import (
    MaskedSocialSSL,
    SocialMaskConfig,
    masked_social_reconstruction_loss,
)
from .node_preservation_ssl import (
    FULL_ONLY_OBJECTIVES,
    HYBRID_OBJECTIVES,
    NEW_GRAPH_OBJECTIVES,
    NodePreservingGraphSSL,
    hybrid_reconstruction_loss,
    node_preserving_reconstruction_loss,
    objective_architecture,
)
EXPERIMENT_VERSION = "social_gnn_experiment_v0"
EDGE_MODES = frozenset({"full", "node_only", "shuffled"})
OBJECTIVES = frozenset({"masked_ssl", "graph_ssl", "mlp_graph_ssl"}) | NEW_GRAPH_OBJECTIVES


@dataclass(frozen=True)
class TrialSpec:
    trial_id: str
    node_path: Path
    edge_path: Path


@dataclass(frozen=True)
class FeatureStandardizer:
    """Per-channel train-only statistics for node values and edge traits."""

    node_mean: np.ndarray
    node_std: np.ndarray
    edge_mean: np.ndarray
    edge_std: np.ndarray
    node_count: np.ndarray
    edge_weight: np.ndarray
    epsilon: float = 1e-6

    def transform(self, package: SocialTrialPackage) -> SocialTrialPackage:
        if package.node_features.shape[-1] != self.node_mean.shape[0]:
            raise ValueError(
                f"Trial {package.trial_id!r}: node dimension "
                f"{package.node_features.shape[-1]} does not match fitted "
                f"dimension {self.node_mean.shape[0]}"
            )
        nodes = (
            package.node_features.astype(np.float64) - self.node_mean
        ) / self.node_std
        nodes[~package.node_mask] = 0.0

        edges = (
            package.edge_values.astype(np.float64) - self.edge_mean
        ) / self.edge_std
        number_of_nodes = package.node_features.shape[1]
        pair_valid = package.node_mask[:, :, None] & package.node_mask[:, None, :]
        pair_valid &= ~np.eye(number_of_nodes, dtype=bool)[None, :, :]
        trait_valid = pair_valid[..., None] & (package.edge_confidence > 0.0)
        edges[~trait_valid] = 0.0
        return SocialTrialPackage(
            trial_id=package.trial_id,
            node_features=nodes.astype(np.float32),
            node_mask=package.node_mask,
            edge_values=edges.astype(np.float32),
            edge_confidence=package.edge_confidence,
            patch_start_s=package.patch_start_s,
            patch_end_s=package.patch_end_s,
            identity=package.identity,
        )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "method": "train_only_per_channel_standardization_v0",
            "node_mean": self.node_mean.tolist(),
            "node_std": self.node_std.tolist(),
            "edge_mean": self.edge_mean.tolist(),
            "edge_std": self.edge_std.tolist(),
            "node_count": self.node_count.tolist(),
            "edge_confidence_weight": self.edge_weight.tolist(),
            "epsilon": self.epsilon,
            "edge_statistics_note": (
                "Per-trait means and variances are weighted by stored edge "
                "confidence over valid off-diagonal animal pairs."
            ),
        }


def _safe_std(
    total: np.ndarray,
    squared_total: np.ndarray,
    weight: np.ndarray,
    *,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    if np.any(weight <= 0):
        missing = np.flatnonzero(weight <= 0).tolist()
        raise ValueError(f"no valid training observations for feature indices {missing}")
    mean = total / weight
    variance = np.maximum(squared_total / weight - mean * mean, 0.0)
    std = np.sqrt(variance)
    std = np.where(std < epsilon, 1.0, std)
    return mean, std


def fit_feature_standardizer(
    packages: Sequence[SocialTrialPackage], *, epsilon: float = 1e-6
) -> FeatureStandardizer:
    """Fit node and confidence-weighted edge statistics on training trials only."""

    if not packages:
        raise ValueError("at least one training trial is required")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    node_dim = packages[0].node_features.shape[-1]
    node_total = np.zeros(node_dim, dtype=np.float64)
    node_squared = np.zeros(node_dim, dtype=np.float64)
    node_count = np.zeros(node_dim, dtype=np.float64)
    edge_total = np.zeros(8, dtype=np.float64)
    edge_squared = np.zeros(8, dtype=np.float64)
    edge_weight = np.zeros(8, dtype=np.float64)

    for package in packages:
        if package.node_features.shape[-1] != node_dim:
            raise ValueError("all trials must share one node feature dimension")
        node_values = package.node_features[package.node_mask].astype(np.float64)
        node_total += node_values.sum(axis=0)
        node_squared += np.square(node_values).sum(axis=0)
        node_count += node_values.shape[0]

        number_of_nodes = package.node_features.shape[1]
        pair_valid = package.node_mask[:, :, None] & package.node_mask[:, None, :]
        pair_valid &= ~np.eye(number_of_nodes, dtype=bool)[None, :, :]
        weights = package.edge_confidence.astype(np.float64) * pair_valid[..., None]
        values = package.edge_values.astype(np.float64)
        edge_total += (weights * values).sum(axis=(0, 1, 2))
        edge_squared += (weights * np.square(values)).sum(axis=(0, 1, 2))
        edge_weight += weights.sum(axis=(0, 1, 2))

    node_mean, node_std = _safe_std(
        node_total, node_squared, node_count, epsilon=epsilon
    )
    edge_mean, edge_std = _safe_std(
        edge_total, edge_squared, edge_weight, epsilon=epsilon
    )
    return FeatureStandardizer(
        node_mean=node_mean,
        node_std=node_std,
        edge_mean=edge_mean,
        edge_std=edge_std,
        node_count=node_count,
        edge_weight=edge_weight,
        epsilon=float(epsilon),
    )


def apply_edge_mode(
    batch: Mapping[str, Any],
    mode: str,
    *,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Apply an in-memory edge ablation without crossing trial boundaries."""

    if mode not in EDGE_MODES:
        raise ValueError(f"edge mode must be one of {sorted(EDGE_MODES)}, got {mode!r}")
    result = dict(batch)
    if mode == "full":
        return result
    values = batch["edge_values"].clone()
    confidence = batch["edge_confidence"].clone()
    if mode == "node_only":
        values.zero_()
        confidence.zero_()
    else:
        lengths = batch["sequence_length"].tolist()
        for batch_index, raw_length in enumerate(lengths):
            length = int(raw_length)
            permutation = torch.randperm(length, generator=generator)
            values[batch_index, :length] = values[
                batch_index, :length
            ].index_select(0, permutation)
            confidence[batch_index, :length] = confidence[
                batch_index, :length
            ].index_select(0, permutation)
    result["edge_values"] = values
    result["edge_confidence"] = confidence
    return result


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _accumulate_metrics(
    totals: dict[str, float], losses: Mapping[str, torch.Tensor], weight: int
) -> None:
    for key in totals:
        value = float(losses[key].detach().cpu())
        if not math.isfinite(value):
            raise FloatingPointError(f"non-finite {key} loss")
        totals[key] += value * weight


def _epoch(
    *,
    model: torch.nn.Module,
    dataloader: Iterable[Mapping[str, Any]],
    objective: str,
    edge_mode: str,
    device: torch.device,
    seed: int,
    mask_config: SocialMaskConfig,
    node_loss_weight: float,
    edge_loss_weight: float,
    preserve_loss_weight: float,
    masked_loss_weight: float,
    relational_match_weight: float,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    graph_objective = objective in {"graph_ssl", "mlp_graph_ssl"}
    node_preservation_objective = objective in NEW_GRAPH_OBJECTIVES
    metric_names: tuple[str, ...] = (
        ("total", "node", "edge", "relation")
        if objective == "masked_ssl" and relational_match_weight > 0
        else ("total", "node", "edge")
    )
    if objective in HYBRID_OBJECTIVES:
        metric_names = (
            "total",
            "preserve",
            "masked",
            "node_full",
            "edge_full",
            "node_masked",
            "edge_masked",
        )
    totals = {name: 0.0 for name in metric_names}
    total_weight = 0
    edge_generator = torch.Generator().manual_seed(seed)
    mask_generator = torch.Generator(device=device).manual_seed(seed + 1)

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for raw_batch in dataloader:
            batch = apply_edge_mode(
                raw_batch, edge_mode, generator=edge_generator
            )
            batch = _to_device(batch, device)
            if objective == "masked_ssl":
                output = model(
                    batch,
                    mask_config=mask_config,
                    generator=mask_generator,
                )
                losses = masked_social_reconstruction_loss(
                    output,
                    batch,
                    node_weight=node_loss_weight,
                    edge_weight=edge_loss_weight,
                )
                if relational_match_weight > 0:
                    if model.relation_classifier is None:
                        raise ValueError(
                            "relational_match_weight > 0 requires "
                            "model.relational_match_head=true"
                        )
                    negative_values = batch["edge_values"].clone()
                    negative_confidence = batch["edge_confidence"].clone()
                    for batch_index, raw_length in enumerate(
                        batch["sequence_length"].tolist()
                    ):
                        length = int(raw_length)
                        permutation = torch.randperm(
                            length, device=device, generator=mask_generator
                        )
                        negative_values[batch_index, :length] = batch[
                            "edge_values"
                        ][batch_index, :length].index_select(0, permutation)
                        negative_confidence[batch_index, :length] = batch[
                            "edge_confidence"
                        ][batch_index, :length].index_select(0, permutation)
                    real_g = model.encoder(
                        batch["node_features"],
                        compose_edge_inputs(
                            batch["edge_values"], batch["edge_confidence"]
                        ),
                        node_mask=batch["node_mask"],
                    )
                    negative_g = model.encoder(
                        batch["node_features"],
                        compose_edge_inputs(
                            negative_values, negative_confidence
                        ),
                        node_mask=batch["node_mask"],
                    )
                    real_logits = model.relation_classifier(real_g).squeeze(-1)
                    negative_logits = model.relation_classifier(negative_g).squeeze(-1)
                    valid_time = batch["time_mask"].bool()
                    relation_loss = 0.5 * (
                        torch.nn.functional.binary_cross_entropy_with_logits(
                            real_logits[valid_time],
                            torch.ones_like(real_logits[valid_time]),
                        )
                        + torch.nn.functional.binary_cross_entropy_with_logits(
                            negative_logits[valid_time],
                            torch.zeros_like(negative_logits[valid_time]),
                        )
                    )
                    losses["relation"] = relation_loss
                    losses["total"] = (
                        losses["total"]
                        + relational_match_weight * relation_loss
                    )
            elif graph_objective:
                output = model(
                    batch,
                    mask_config=mask_config,
                    generator=mask_generator,
                )
                losses = graph_level_reconstruction_loss(
                    output,
                    batch,
                    node_weight=node_loss_weight,
                    edge_weight=edge_loss_weight,
                )
            elif node_preservation_objective:
                full_output = model(batch, view="full")
                if objective in HYBRID_OBJECTIVES:
                    masked_output = model(
                        batch,
                        view="masked",
                        mask_config=mask_config,
                        generator=mask_generator,
                    )
                    losses = hybrid_reconstruction_loss(
                        masked_output,
                        full_output,
                        batch,
                        preserve_weight=preserve_loss_weight,
                        masked_weight=masked_loss_weight,
                        node_weight=node_loss_weight,
                        edge_weight=edge_loss_weight,
                    )
                else:
                    losses = node_preserving_reconstruction_loss(
                        full_output,
                        batch,
                        node_weight=node_loss_weight,
                        edge_weight=edge_loss_weight,
                    )
            else:  # pragma: no cover - guarded by objective validation
                raise RuntimeError(f"unsupported GNN objective {objective!r}")
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip_norm
                    )
                optimizer.step()
            weight = max(1, int(batch["time_mask"].sum().item()))
            _accumulate_metrics(totals, losses, weight)
            total_weight += weight
    if total_weight == 0:
        raise ValueError("dataloader produced no valid timesteps")
    return {key: value / total_weight for key, value in totals.items()}


def _resolve_device(value: str) -> torch.device:
    requested = value.strip().lower()
    if requested == "auto":
        if not torch.cuda.is_available():
            return torch.device("cpu")
        free_memory = []
        for index in range(torch.cuda.device_count()):
            with torch.cuda.device(index):
                free_bytes, _total_bytes = torch.cuda.mem_get_info()
            free_memory.append(free_bytes)
        return torch.device(f"cuda:{int(np.argmax(free_memory))}")
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {value!r} requested but CUDA is unavailable")
        index = device.index if device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {index} is unavailable; found "
                f"{torch.cuda.device_count()} device(s)"
            )
    return device


def _trial_specs(config: Mapping[str, Any]) -> list[TrialSpec]:
    specs: list[TrialSpec] = []
    seen: set[str] = set()
    for raw in config.get("trials", []):
        trial_id = str(raw["id"]).strip()
        if not trial_id or trial_id in seen:
            raise ValueError(f"trial IDs must be non-empty and unique: {trial_id!r}")
        seen.add(trial_id)
        specs.append(
            TrialSpec(
                trial_id=trial_id,
                node_path=Path(raw["node_npz"]).expanduser(),
                edge_path=Path(raw["edge_npz"]).expanduser(),
            )
        )
    if not specs:
        raise ValueError("configuration contains no trials")
    return specs


def _load_packages(
    specs: Sequence[TrialSpec], *, allow_missing: bool
) -> tuple[list[SocialTrialPackage], list[dict[str, str]]]:
    packages: list[SocialTrialPackage] = []
    missing: list[dict[str, str]] = []
    for spec in specs:
        absent = [
            str(path.resolve())
            for path in (spec.node_path, spec.edge_path)
            if not path.is_file()
        ]
        if absent:
            missing.append({"trial_id": spec.trial_id, "missing": "; ".join(absent)})
            if allow_missing:
                continue
            raise FileNotFoundError(
                f"Trial {spec.trial_id!r} is incomplete; missing {absent}"
            )
        with np.load(spec.node_path, allow_pickle=False) as node_pack:
            if "training_eligible" in node_pack.files:
                eligible = np.asarray(node_pack["training_eligible"])
                if eligible.size != 1 or not bool(eligible.reshape(-1)[0]):
                    raise ValueError(
                        f"Trial {spec.trial_id!r}: node NPZ is marked "
                        "training_eligible=false"
                    )
        packages.append(
            SocialTrialPackage.from_npz(
                spec.node_path, spec.edge_path, trial_id=spec.trial_id
            )
        )
    if not packages:
        raise ValueError("no complete node/edge trial pair is available")
    node_dims = {package.node_features.shape[-1] for package in packages}
    node_counts = {package.node_features.shape[1] for package in packages}
    if len(node_dims) != 1 or len(node_counts) != 1:
        raise ValueError(
            "all trials must share node feature dimension and number of identities"
        )
    return packages, missing


def _split_packages(
    packages: Sequence[SocialTrialPackage], validation_ids: Sequence[str]
) -> tuple[list[SocialTrialPackage], list[SocialTrialPackage]]:
    available = {package.trial_id for package in packages}
    validation_set = {str(value) for value in validation_ids}
    unknown = sorted(validation_set - available)
    if unknown:
        raise ValueError(f"validation trials are missing or unknown: {unknown}")
    training = [p for p in packages if p.trial_id not in validation_set]
    validation = [p for p in packages if p.trial_id in validation_set]
    if not training or not validation:
        raise ValueError("formal runs require non-empty disjoint train and validation trials")
    return training, validation


def _mask_config(
    config: Mapping[str, Any], override: Mapping[str, Any] | None = None
) -> SocialMaskConfig:
    ssl = {**config.get("ssl", {}), **(override or {})}
    return SocialMaskConfig(
        node_mask_probability=float(ssl.get("node_mask_probability", 0.15)),
        edge_mask_probability=float(ssl.get("edge_mask_probability", 0.15)),
        node_modality_dropout=float(ssl.get("node_modality_dropout", 0.05)),
        edge_modality_dropout=float(ssl.get("edge_modality_dropout", 0.05)),
        couple_symmetric_traits=bool(ssl.get("couple_symmetric_traits", False)),
    )


def _build_model(
    objective: str,
    *,
    node_dim: int,
    number_of_nodes: int,
    model_config: Mapping[str, Any],
) -> torch.nn.Module:
    common = {
        "node_dim": node_dim,
        "hidden_dim": int(model_config.get("hidden_dim", 64)),
        "layers": int(model_config.get("layers", 2)),
        "decoder_hidden_dim": int(
            model_config.get(
                "decoder_hidden_dim", model_config.get("hidden_dim", 64)
            )
        ),
    }
    if objective == "masked_ssl":
        return MaskedSocialSSL(
            **common,
            readout=str(model_config.get("readout", "mean")),
            number_of_nodes=number_of_nodes,
            graph_dim=int(model_config.get("graph_dim", common["hidden_dim"])),
            relational_match_head=bool(
                model_config.get("relational_match_head", False)
            ),
        )
    if objective in {"graph_ssl", "mlp_graph_ssl"}:
        return GraphLevelSocialSSL(
            node_dim=node_dim,
            number_of_nodes=number_of_nodes,
            hidden_dim=common["hidden_dim"],
            graph_dim=int(model_config.get("graph_dim", common["hidden_dim"])),
            layers=common["layers"],
            decoder_hidden_dim=common["decoder_hidden_dim"],
            readout=str(model_config.get("readout", "mean")),
            backbone="mlp" if objective == "mlp_graph_ssl" else "gnn",
            fusion_hidden_dim=int(model_config.get("fusion_hidden_dim", 192)),
        )
    if objective in NEW_GRAPH_OBJECTIVES:
        architecture, target_mode = objective_architecture(objective)
        return NodePreservingGraphSSL(
            node_dim=node_dim,
            number_of_nodes=number_of_nodes,
            hidden_dim=common["hidden_dim"],
            graph_dim=int(model_config.get("graph_dim", common["hidden_dim"])),
            layers=common["layers"],
            decoder_hidden_dim=common["decoder_hidden_dim"],
            readout=str(model_config.get("readout", "mean")),
            architecture=architecture,
            target_mode=target_mode,
            branch_hidden_dim=int(model_config.get("branch_hidden_dim", 32)),
            fusion_hidden_dim=int(model_config.get("branch_fusion_hidden_dim", 64)),
            invariant_mlp_hidden_dim=int(
                model_config.get("invariant_mlp_hidden_dim", 160)
            ),
        )
    raise ValueError(f"unsupported GNN objective {objective!r}")


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    run_config: Mapping[str, Any],
    normalizer: FeatureStandardizer,
    validation_metrics: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "contract_version": EXPERIMENT_VERSION,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "run_config": dict(run_config),
        "normalization": normalizer.to_manifest(),
        "validation_metrics": dict(validation_metrics),
    }


def _export_embeddings(
    *,
    model: torch.nn.Module,
    packages_by_split: Mapping[str, Sequence[SocialTrialPackage]],
    edge_mode: str,
    objective: str,
    device: torch.device,
    output_dir: Path,
    seed: int,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for split, packages in packages_by_split.items():
            split_dir = output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            loader = build_social_dataloader(
                SocialTrialDataset(list(packages)), batch_size=1, shuffle=False
            )
            generator = torch.Generator().manual_seed(seed)
            for raw_batch in loader:
                batch = apply_edge_mode(
                    raw_batch, edge_mode, generator=generator
                )
                batch = _to_device(batch, device)
                if objective == "masked_ssl":
                    edge_features = compose_edge_inputs(
                        batch["edge_values"], batch["edge_confidence"]
                    )
                    embedding = model.encoder(
                        batch["node_features"],
                        edge_features,
                        node_mask=batch["node_mask"],
                    )
                elif objective in {"graph_ssl", "mlp_graph_ssl"} | NEW_GRAPH_OBJECTIVES:
                    embedding = model.encode_batch(batch)
                else:  # pragma: no cover - guarded by objective validation
                    raise RuntimeError(f"unsupported GNN objective {objective!r}")
                length = int(batch["sequence_length"][0].item())
                trial_id = str(batch["trial_id"][0])
                path = split_dir / f"{trial_id}_g_t.npz"
                np.savez_compressed(
                    path,
                    contract_version=np.asarray(EXPERIMENT_VERSION),
                    trial_id=np.asarray(trial_id),
                    split=np.asarray(split),
                    objective=np.asarray(objective),
                    representation=np.asarray("g_t"),
                    edge_mode=np.asarray(edge_mode),
                    g_t=embedding[0, :length].detach().cpu().numpy().astype(np.float32),
                    patch_start_s=batch["patch_start_s"][0, :length].cpu().numpy(),
                    patch_end_s=batch["patch_end_s"][0, :length].cpu().numpy(),
                    identity=np.asarray(batch["identity"][0]),
                )
                records.append(
                    {
                        "trial_id": trial_id,
                        "split": split,
                        "path": str(path.resolve()),
                        "shape": [length, int(embedding.shape[-1])],
                    }
                )
    return records


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _timestamped_directory(root: Path, prefix: str) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    candidate = root / f"{prefix}_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = root / f"{prefix}_{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _run_definitions(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    names: set[str] = set()
    for raw in config.get("runs", []):
        name = str(raw["name"]).strip()
        objective = str(raw["objective"]).strip()
        edge_mode = str(raw.get("edge_mode", "full")).strip()
        if not name or name in names:
            raise ValueError(f"run names must be non-empty and unique: {name!r}")
        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {sorted(OBJECTIVES)}")
        if edge_mode not in EDGE_MODES:
            raise ValueError(f"edge_mode must be one of {sorted(EDGE_MODES)}")
        names.add(name)
        record: dict[str, Any] = {
            "name": name,
            "objective": objective,
            "edge_mode": edge_mode,
        }
        if "selection_metric" in raw:
            record["selection_metric"] = str(raw["selection_metric"])
        for key in ("model", "ssl"):
            if key in raw:
                if not isinstance(raw[key], Mapping):
                    raise ValueError(f"run {key} override must be a mapping")
                record[key] = dict(raw[key])
        for key in (
            "node_loss_weight",
            "edge_loss_weight",
            "preserve_loss_weight",
            "masked_loss_weight",
            "relational_match_weight",
        ):
            if key in raw:
                value = float(raw[key])
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"{key} must be finite and nonnegative")
                record[key] = value
        runs.append(record)
    if not runs:
        raise ValueError("configuration contains no experiment runs")
    return runs


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("experiment config must be a YAML mapping")
    if config.get("version") != EXPERIMENT_VERSION:
        raise ValueError(
            f"expected config version {EXPERIMENT_VERSION!r}, got "
            f"{config.get('version')!r}"
        )
    config["_config_path"] = str(config_path)
    _trial_specs(config)
    _run_definitions(config)
    return config


def run_experiment(
    config: Mapping[str, Any],
    *,
    device_override: str | None = None,
    seed_override: int | None = None,
    smoke_test: bool = False,
) -> dict[str, Any]:
    """Validate inputs, run configured experiments, and export best ``g_t``."""

    seed = int(config.get("seed", 0) if seed_override is None else seed_override)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _resolve_device(device_override or str(config.get("device", "auto")))
    specs = _trial_specs(config)
    packages, missing = _load_packages(specs, allow_missing=smoke_test)
    output_root = Path(config["output_dir"]).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    session_dir = _timestamped_directory(
        output_root, "smoke" if smoke_test else "experiment"
    )

    if smoke_test:
        training = [packages[0]]
        validation = [packages[0]]
        test: list[SocialTrialPackage] = []
    else:
        split_config = config.get("split", {})
        validation_ids = [str(value) for value in split_config.get("validation_trials", [])]
        test_ids = {str(value) for value in split_config.get("test_trials", [])}
        excluded_ids = {str(value) for value in split_config.get("excluded_trials", [])}
        overlap = (set(validation_ids) & test_ids) | (
            (set(validation_ids) | test_ids) & excluded_ids
        )
        if overlap:
            raise ValueError(
                "validation, test, and excluded trial IDs overlap: "
                f"{sorted(overlap)}"
            )
        available = {package.trial_id for package in packages}
        unknown_held_out = sorted((test_ids | excluded_ids) - available)
        if unknown_held_out:
            raise ValueError(
                f"test/excluded trials are missing or unknown: {unknown_held_out}"
            )
        test = [package for package in packages if package.trial_id in test_ids]
        selectable = [
            package
            for package in packages
            if package.trial_id not in test_ids | excluded_ids
        ]
        training, validation = _split_packages(selectable, validation_ids)
    normalizer = fit_feature_standardizer(
        training, epsilon=float(config.get("normalization", {}).get("epsilon", 1e-6))
    )
    normalized_training = [normalizer.transform(package) for package in training]
    normalized_validation = [normalizer.transform(package) for package in validation]
    normalized_test = [normalizer.transform(package) for package in test]
    _write_json(session_dir / "normalization.json", normalizer.to_manifest())

    training_config = config.get("training", {})
    model_config = config.get("model", {})
    ssl_config = config.get("ssl", {})
    epochs = 1 if smoke_test else int(training_config.get("epochs", 100))
    if epochs < 1:
        raise ValueError("training.epochs must be >= 1")
    batch_size = int(training_config.get("batch_size", 1))
    num_workers = int(training_config.get("num_workers", 0))
    node_dim = normalized_training[0].node_features.shape[-1]
    number_of_nodes = normalized_training[0].node_features.shape[1]

    manifest: dict[str, Any] = {
        "contract_version": EXPERIMENT_VERSION,
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config_path": config.get("_config_path"),
        "output_dir": str(session_dir.resolve()),
        "device": str(device),
        "seed": seed,
        "smoke_test": smoke_test,
        "missing_trials_skipped_for_smoke": missing,
        "training_trials": [p.trial_id for p in training],
        "validation_trials": [p.trial_id for p in validation],
        "test_trials_held_out_from_training_and_selection": [p.trial_id for p in test],
        "excluded_trials": (
            []
            if smoke_test
            else sorted(
                str(value)
                for value in config.get("split", {}).get("excluded_trials", [])
            )
        ),
        "runs": [],
    }
    _write_json(session_dir / "manifest.json", manifest)

    for run in _run_definitions(config):
        run_dir = session_dir / run["name"]
        run_dir.mkdir()
        run_seed = seed
        # All masked edge ablations start from identical weights and use the
        # same trial order and target masks. Their only intended difference is
        # the edge input condition.
        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)
        train_loader = build_social_dataloader(
            SocialTrialDataset(normalized_training),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        validation_loader = build_social_dataloader(
            SocialTrialDataset(normalized_validation),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        effective_model_config = {**model_config, **run.get("model", {})}
        effective_ssl_config = {**ssl_config, **run.get("ssl", {})}
        run_mask_config = _mask_config(config, run.get("ssl", {}))
        model = _build_model(
            run["objective"],
            node_dim=node_dim,
            number_of_nodes=number_of_nodes,
            model_config=effective_model_config,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training_config.get("learning_rate", 1e-3)),
            weight_decay=float(training_config.get("weight_decay", 1e-4)),
        )
        best_metric = math.inf
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        history: list[dict[str, Any]] = []
        selection_metric = str(
            run.get(
                "selection_metric",
                "node" if run["objective"] == "masked_ssl" else "total",
            )
        )
        for epoch in range(1, epochs + 1):
            train_metrics = _epoch(
                model=model,
                dataloader=train_loader,
                objective=run["objective"],
                edge_mode=run["edge_mode"],
                device=device,
                seed=run_seed + epoch * 10,
                mask_config=run_mask_config,
                node_loss_weight=float(run.get("node_loss_weight", effective_ssl_config.get("node_loss_weight", 1.0))),
                edge_loss_weight=float(run.get("edge_loss_weight", effective_ssl_config.get("edge_loss_weight", 1.0))),
                preserve_loss_weight=float(run.get("preserve_loss_weight", effective_ssl_config.get("preserve_loss_weight", 1.0))),
                masked_loss_weight=float(run.get("masked_loss_weight", effective_ssl_config.get("masked_loss_weight", 1.0))),
                relational_match_weight=float(
                    run.get("relational_match_weight", effective_ssl_config.get("relational_match_weight", 0.0))
                ),
                optimizer=optimizer,
                gradient_clip_norm=(
                    None
                    if training_config.get("gradient_clip_norm") is None
                    else float(training_config["gradient_clip_norm"])
                ),
            )
            validation_metrics = _epoch(
                model=model,
                dataloader=validation_loader,
                objective=run["objective"],
                edge_mode=run["edge_mode"],
                device=device,
                seed=run_seed + 9_000_000,
                mask_config=run_mask_config,
                node_loss_weight=float(run.get("node_loss_weight", effective_ssl_config.get("node_loss_weight", 1.0))),
                edge_loss_weight=float(run.get("edge_loss_weight", effective_ssl_config.get("edge_loss_weight", 1.0))),
                preserve_loss_weight=float(run.get("preserve_loss_weight", effective_ssl_config.get("preserve_loss_weight", 1.0))),
                masked_loss_weight=float(run.get("masked_loss_weight", effective_ssl_config.get("masked_loss_weight", 1.0))),
                relational_match_weight=float(
                    run.get("relational_match_weight", effective_ssl_config.get("relational_match_weight", 0.0))
                ),
                optimizer=None,
                gradient_clip_norm=None,
            )
            row = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
            }
            history.append(row)
            with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            metric = float(validation_metrics[selection_metric])
            if metric < best_metric:
                best_metric = metric
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                torch.save(
                    _checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        run_config=run,
                        normalizer=normalizer,
                        validation_metrics=validation_metrics,
                    ),
                    run_dir / "best.pt",
                )

        torch.save(
            _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                epoch=epochs,
                run_config=run,
                normalizer=normalizer,
                validation_metrics=history[-1]["validation"],
            ),
            run_dir / "last.pt",
        )
        if best_state is None:
            raise AssertionError("a finite best checkpoint was not selected")
        model.load_state_dict(best_state)
        embeddings = _export_embeddings(
            model=model,
            packages_by_split={
                "train": normalized_training,
                "validation": normalized_validation,
                **({"test": normalized_test} if normalized_test else {}),
            },
            edge_mode=run["edge_mode"],
            objective=run["objective"],
            device=device,
            output_dir=run_dir / "embeddings",
            seed=run_seed + 8_000_000,
        )
        run_summary = {
            **run,
            "status": "completed",
            "epochs": epochs,
            "selection_metric": selection_metric,
            "best_epoch": best_epoch,
            "best_validation_metric": best_metric,
            "best_validation": history[best_epoch - 1]["validation"],
            "last_train": history[-1]["train"],
            "last_validation": history[-1]["validation"],
            "embeddings": embeddings,
        }
        _write_json(run_dir / "summary.json", run_summary)
        manifest["runs"].append(run_summary)
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    manifest["status"] = "completed"
    manifest["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    _write_json(session_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train patch-level Social-GNN SSL objectives from "
            "strictly aligned node/edge trial packages."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", help="Override config device, e.g. cpu or cuda:1")
    parser.add_argument("--seed", type=int, help="Override the configured random seed")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Use the first complete trial for one train/eval step per run; "
            "missing later trials are reported rather than accepted formally."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_experiment_config(args.config)
    manifest = run_experiment(
        config,
        device_override=args.device,
        seed_override=args.seed,
        smoke_test=args.smoke_test,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
