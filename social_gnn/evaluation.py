"""Shared, leakage-resistant evaluation utilities for Social-GNN embeddings.

The functions in this module never modify stored trial files.  They load the
strict :class:`SocialTrialPackage` contract, apply the training normalizer in
memory, and evaluate each patch independently (the encoder has no temporal
state).  Trial IDs remain attached to every result so callers can aggregate
uncertainty at the trial level instead of pretending patches are independent
experimental replicates.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from .data import SocialTrialPackage
from .experiment import (
    FeatureStandardizer,
    _build_model,
    _load_packages,
    _resolve_device,
    _trial_specs,
    load_experiment_config,
)
from .graph_builder import compose_edge_inputs
from .node_preservation_ssl import NEW_GRAPH_OBJECTIVES


TRAIT_NAMES = (
    "centroid_distance",
    "nose_to_nose_distance",
    "source_nose_to_target_centroid_distance",
    "source_faces_target",
    "head_direction_alignment",
    "source_radial_approach_speed",
    "pair_closing_speed",
    "movement_direction_alignment",
)
SYMMETRIC_TRAITS = frozenset({0, 1, 4, 6, 7})
DIRECTED_TRAITS = frozenset(set(range(8)) - SYMMETRIC_TRAITS)


@dataclass(frozen=True)
class LoadedEncoder:
    """A frozen encoder plus the exact train-only normalizer it was fit with."""

    name: str
    model: torch.nn.Module
    normalizer: FeatureStandardizer
    native_edge_mode: str
    checkpoint_path: str
    epoch: int | None
    parameter_count: int
    objective: str = "masked_ssl"
    representation: str = "g_t"


def standardizer_from_manifest(manifest: Mapping[str, Any]) -> FeatureStandardizer:
    """Reconstruct a ``FeatureStandardizer`` from a checkpoint manifest."""

    required = (
        "node_mean",
        "node_std",
        "edge_mean",
        "edge_std",
        "node_count",
        "edge_confidence_weight",
    )
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError(f"normalization manifest is missing {missing}")
    return FeatureStandardizer(
        node_mean=np.asarray(manifest["node_mean"], dtype=np.float64),
        node_std=np.asarray(manifest["node_std"], dtype=np.float64),
        edge_mean=np.asarray(manifest["edge_mean"], dtype=np.float64),
        edge_std=np.asarray(manifest["edge_std"], dtype=np.float64),
        node_count=np.asarray(manifest["node_count"], dtype=np.float64),
        edge_weight=np.asarray(
            manifest["edge_confidence_weight"], dtype=np.float64
        ),
        epsilon=float(manifest.get("epsilon", 1e-6)),
    )


def load_trials(config_path: str | Path) -> tuple[dict[str, Any], list[SocialTrialPackage]]:
    """Load every complete trial named by an experiment configuration."""

    config = load_experiment_config(config_path)
    packages, missing = _load_packages(_trial_specs(config), allow_missing=False)
    if missing:
        raise AssertionError("allow_missing=False returned missing trials")
    return config, packages


def discover_masked_checkpoints(experiment_dir: str | Path) -> dict[str, Path]:
    """Find Masked-SSL ``best.pt`` files without including VAE checkpoints."""

    root = Path(experiment_dir)
    checkpoints: dict[str, Path] = {}
    for path in sorted(root.glob("*/best.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        run = payload.get("run_config", {})
        if run.get("objective") == "masked_ssl":
            checkpoints[str(run.get("name", path.parent.name))] = path
    if not checkpoints:
        raise FileNotFoundError(f"no masked SSL checkpoints found below {root}")
    return checkpoints


def discover_encoder_checkpoints(experiment_dir: str | Path) -> dict[str, Path]:
    """Find checkpoints exposing a deterministic patch-level representation."""

    root = Path(experiment_dir)
    checkpoints: dict[str, Path] = {}
    accepted = {"masked_ssl", "graph_ssl", "mlp_graph_ssl", "vae"} | NEW_GRAPH_OBJECTIVES
    for path in sorted(root.glob("*/best.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        run = payload.get("run_config", {})
        if run.get("objective") in accepted:
            checkpoints[str(run.get("name", path.parent.name))] = path
    if not checkpoints:
        raise FileNotFoundError(f"no encoder checkpoints found below {root}")
    return checkpoints


def load_encoder(
    checkpoint_path: str | Path,
    *,
    model_config: Mapping[str, Any],
    node_dim: int,
    number_of_nodes: int,
    device: torch.device,
    name: str | None = None,
    vae_config: Mapping[str, Any] | None = None,
    representation: str | None = None,
) -> LoadedEncoder:
    """Load one frozen patch encoder and its normalization contract."""

    path = Path(checkpoint_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    run = payload.get("run_config", {})
    objective = str(run.get("objective", ""))
    if objective not in {"masked_ssl", "graph_ssl", "mlp_graph_ssl", "vae"} | NEW_GRAPH_OBJECTIVES:
        raise ValueError(f"{path} is not a supported encoder checkpoint")
    selected_representation = representation or ("mu" if objective == "vae" else "g_t")
    if objective == "vae" and selected_representation not in {"mu", "g_t"}:
        raise ValueError("VAE representation must be 'mu' or 'g_t'")
    if objective != "vae" and selected_representation != "g_t":
        raise ValueError("non-VAE encoders expose only the 'g_t' representation")
    model = _build_model(
        objective,
        node_dim=node_dim,
        number_of_nodes=number_of_nodes,
        model_config=model_config,
        vae_config=vae_config or {},
    )
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return LoadedEncoder(
        name=name or str(run.get("name", path.parent.name)),
        model=model,
        normalizer=standardizer_from_manifest(payload["normalization"]),
        native_edge_mode=str(run.get("edge_mode", "full")),
        checkpoint_path=str(path.resolve()),
        epoch=int(payload["epoch"]) if "epoch" in payload else None,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        objective=objective,
        representation=selected_representation,
    )


def make_random_encoder(
    *,
    reference: LoadedEncoder,
    model_config: Mapping[str, Any],
    node_dim: int,
    number_of_nodes: int,
    device: torch.device,
    seed: int,
    name: str = "random_untrained",
    vae_config: Mapping[str, Any] | None = None,
) -> LoadedEncoder:
    """Create a deterministic untrained architecture control."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = _build_model(
            reference.objective,
            node_dim=node_dim,
            number_of_nodes=number_of_nodes,
            model_config=model_config,
            vae_config=vae_config or {},
        )
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return LoadedEncoder(
        name=name,
        model=model,
        normalizer=reference.normalizer,
        native_edge_mode="full",
        checkpoint_path="<deterministic random initialization>",
        epoch=None,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        objective=reference.objective,
        representation=reference.representation,
    )


def resolve_evaluation_device(value: str) -> torch.device:
    return _resolve_device(value)


def _copy_package(
    package: SocialTrialPackage,
    *,
    node_features: np.ndarray | None = None,
    edge_values: np.ndarray | None = None,
    edge_confidence: np.ndarray | None = None,
) -> SocialTrialPackage:
    return SocialTrialPackage(
        trial_id=package.trial_id,
        node_features=package.node_features if node_features is None else node_features,
        node_mask=package.node_mask,
        edge_values=package.edge_values if edge_values is None else edge_values,
        edge_confidence=(
            package.edge_confidence
            if edge_confidence is None
            else edge_confidence
        ),
        patch_start_s=package.patch_start_s,
        patch_end_s=package.patch_end_s,
        identity=package.identity,
    )


def nonidentity_permutation(length: int, rng: np.random.Generator) -> np.ndarray:
    """Return a random derangement, preserving the source sample distribution."""

    if length < 2:
        raise ValueError("a replacement permutation requires at least two patches")
    original = np.arange(length)
    permutation = rng.permutation(length)
    fixed = np.flatnonzero(permutation == original)
    if fixed.size == 1:
        swap = (int(fixed[0]) + 1) % length
        permutation[fixed[0]], permutation[swap] = (
            permutation[swap],
            permutation[fixed[0]],
        )
    elif fixed.size > 1:
        permutation[fixed] = np.roll(permutation[fixed], 1)
    if np.any(permutation == original):
        raise AssertionError("failed to construct a derangement")
    return permutation


def perturb_package(
    package: SocialTrialPackage,
    mode: str,
    *,
    rng: np.random.Generator,
    permutation: np.ndarray | None = None,
    trait: int | None = None,
) -> SocialTrialPackage:
    """Apply one explicit in-memory perturbation to a normalized package."""

    t = package.node_features.shape[0]
    if mode == "real":
        return package
    if mode == "zero":
        return _copy_package(
            package,
            edge_values=np.zeros_like(package.edge_values),
            edge_confidence=np.zeros_like(package.edge_confidence),
        )
    if permutation is None:
        permutation = nonidentity_permutation(t, rng)
    permutation = np.asarray(permutation, dtype=np.int64)
    if permutation.shape != (t,) or np.any((permutation < 0) | (permutation >= t)):
        raise ValueError("permutation must contain one valid source index per patch")

    if mode in {"shuffled", "matched"}:
        return _copy_package(
            package,
            edge_values=package.edge_values[permutation],
            edge_confidence=package.edge_confidence[permutation],
        )
    if mode == "value_only":
        return _copy_package(package, edge_values=package.edge_values[permutation])
    if mode == "confidence_only":
        return _copy_package(
            package, edge_confidence=package.edge_confidence[permutation]
        )
    if mode == "node_shuffled":
        return _copy_package(package, node_features=package.node_features[permutation])
    if mode == "trait":
        if trait is None or trait not in range(8):
            raise ValueError("trait perturbation requires trait in [0,7]")
        values = np.array(package.edge_values, copy=True)
        confidence = np.array(package.edge_confidence, copy=True)
        # Both directed entries are replaced together.  This is essential for
        # symmetric traits because their reverse edge is an exact duplicate.
        values[..., trait] = package.edge_values[permutation, ..., trait]
        confidence[..., trait] = package.edge_confidence[permutation, ..., trait]
        return _copy_package(
            package, edge_values=values, edge_confidence=confidence
        )
    raise ValueError(f"unknown perturbation mode {mode!r}")


def _flat_nodes(package: SocialTrialPackage) -> np.ndarray:
    values = np.array(package.node_features, copy=True)
    values[~package.node_mask] = 0.0
    return values.reshape(values.shape[0], -1).astype(np.float64)


def _flat_edges(package: SocialTrialPackage) -> np.ndarray:
    if package.edge_values.shape[1] != 2:
        raise ValueError("current matched-pair evaluator expects exactly two animals")
    values = package.edge_values[:, [0, 1], [1, 0], :]
    confidence = package.edge_confidence[:, [0, 1], [1, 0], :]
    return np.concatenate((values, confidence), axis=-1).reshape(values.shape[0], -1).astype(np.float64)


def matched_replacement_indices(
    package: SocialTrialPackage,
    *,
    rng: np.random.Generator,
    candidate_count: int = 96,
    node_keep_fraction: float = 0.2,
    minimum_time_gap: int = 16,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Find node-similar edge-different and edge-similar real patch matches.

    Candidate selection is trial-constrained.  It first retains the closest
    node candidates, then selects the most or least edge-different member.
    This avoids replacing an edge with an arbitrary out-of-distribution vector.
    """

    nodes = _flat_nodes(package)
    edges = _flat_edges(package)
    length = nodes.shape[0]
    if length < 3:
        raise ValueError("matched replacement requires at least three patches")
    count = min(max(8, candidate_count), length - 1)
    keep = max(2, int(math.ceil(count * node_keep_fraction)))
    edge_different = np.empty(length, dtype=np.int64)
    edge_similar = np.empty(length, dtype=np.int64)
    selected_node_distance = np.empty(length, dtype=np.float64)
    selected_edge_distance = np.empty(length, dtype=np.float64)
    control_edge_distance = np.empty(length, dtype=np.float64)

    all_indices = np.arange(length)
    for index in range(length):
        valid = np.abs(all_indices - index) >= minimum_time_gap
        candidates = all_indices[valid]
        if candidates.size < 2:
            candidates = all_indices[all_indices != index]
        if candidates.size > count:
            candidates = rng.choice(candidates, size=count, replace=False)
        node_distance = np.mean(np.square(nodes[candidates] - nodes[index]), axis=1)
        closest = candidates[np.argsort(node_distance)[: min(keep, candidates.size)]]
        edge_distance = np.mean(np.square(edges[closest] - edges[index]), axis=1)
        far_position = int(np.argmax(edge_distance))
        near_position = int(np.argmin(edge_distance))
        edge_different[index] = closest[far_position]
        edge_similar[index] = closest[near_position]
        selected_node_distance[index] = np.mean(
            np.square(nodes[edge_different[index]] - nodes[index])
        )
        selected_edge_distance[index] = edge_distance[far_position]
        control_edge_distance[index] = edge_distance[near_position]
    diagnostics = {
        "node_distance": selected_node_distance,
        "edge_distance": selected_edge_distance,
        "control_edge_distance": control_edge_distance,
    }
    return edge_different, edge_similar, diagnostics


def apply_native_edge_mode(
    package: SocialTrialPackage,
    mode: str,
    *,
    rng: np.random.Generator,
) -> SocialTrialPackage:
    if mode == "full":
        return package
    if mode == "node_only":
        return perturb_package(package, "zero", rng=rng)
    if mode == "shuffled":
        return perturb_package(package, "shuffled", rng=rng)
    raise ValueError(f"unsupported native edge mode {mode!r}")


@torch.inference_mode()
def encode_package(
    encoder: LoadedEncoder,
    package: SocialTrialPackage,
    *,
    device: torch.device,
    chunk_size: int = 4096,
) -> np.ndarray:
    """Encode one normalized full trial in patch-only chunks.

    Ordinary SSL models expose ``g_t``. VAE models expose either deterministic
    posterior ``mu`` (the canonical VAE representation) or pre-VAE ``g_t`` for
    a diagnostic showing where information is lost.
    """

    outputs: list[np.ndarray] = []
    for start in range(0, package.node_features.shape[0], chunk_size):
        stop = min(package.node_features.shape[0], start + chunk_size)
        # SocialTrialPackage arrays are deliberately read-only.  Copy before
        # ``from_numpy`` so PyTorch never receives a non-writable buffer.
        nodes = torch.from_numpy(
            np.array(package.node_features[start:stop], copy=True)
        ).unsqueeze(0).to(device)
        node_mask = torch.from_numpy(
            np.array(package.node_mask[start:stop], copy=True)
        ).unsqueeze(0).to(device)
        values = torch.from_numpy(
            np.array(package.edge_values[start:stop], copy=True)
        ).unsqueeze(0).to(device)
        confidence = torch.from_numpy(
            np.array(package.edge_confidence[start:stop], copy=True)
        ).unsqueeze(0).to(device)
        if encoder.objective == "vae":
            result = encoder.model(
                {
                    "node_features": nodes,
                    "node_mask": node_mask,
                    "edge_values": values,
                    "edge_confidence": confidence,
                },
                sample=False,
            )
            embedding = result[encoder.representation]
        else:
            embedding = encoder.model.encoder(
                nodes,
                compose_edge_inputs(values, confidence),
                node_mask=node_mask,
            )
        outputs.append(embedding[0].cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def latent_scale(embeddings: Iterable[np.ndarray], epsilon: float = 1e-8) -> float:
    """RMS radius around the global mean; invariant to patch count."""

    arrays = [np.asarray(value, dtype=np.float64) for value in embeddings]
    joined = np.concatenate(arrays, axis=0)
    scale = float(np.sqrt(np.var(joined, axis=0).sum()))
    return max(scale, epsilon)


def displacement_summary(
    reference: np.ndarray,
    alternative: np.ndarray,
    *,
    scale: float,
) -> dict[str, float]:
    displacement = np.linalg.norm(
        np.asarray(alternative, dtype=np.float64)
        - np.asarray(reference, dtype=np.float64),
        axis=1,
    ) / scale
    ref_norm = np.linalg.norm(reference, axis=1)
    alt_norm = np.linalg.norm(alternative, axis=1)
    cosine = np.sum(reference * alternative, axis=1) / np.maximum(
        ref_norm * alt_norm, 1e-12
    )
    return {
        "n_patches": int(displacement.size),
        "mean_normalized_displacement": float(np.mean(displacement)),
        "median_normalized_displacement": float(np.median(displacement)),
        "p90_normalized_displacement": float(np.quantile(displacement, 0.9)),
        "mean_cosine_change": float(np.mean(1.0 - cosine)),
    }


def unique_edge_targets(package: SocialTrialPackage) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return 11 nonduplicated target channels for a two-animal graph.

    Symmetric traits appear once.  Directed traits appear once for 0->1 and
    once for 1->0, retaining scientifically meaningful directionality.
    """

    if package.edge_values.shape[1:3] != (2, 2):
        raise ValueError("unique edge target extraction currently requires N=2")
    values: list[np.ndarray] = []
    confidence: list[np.ndarray] = []
    names: list[str] = []
    for trait, name in enumerate(TRAIT_NAMES):
        if trait in SYMMETRIC_TRAITS:
            values.append(package.edge_values[:, 0, 1, trait])
            confidence.append(package.edge_confidence[:, 0, 1, trait])
            names.append(name)
        else:
            for source, target in ((0, 1), (1, 0)):
                values.append(package.edge_values[:, source, target, trait])
                confidence.append(package.edge_confidence[:, source, target, trait])
                names.append(f"{name}_{source}_to_{target}")
    return (
        np.stack(values, axis=1).astype(np.float64),
        np.stack(confidence, axis=1).astype(np.float64),
        names,
    )


def invariant_edge_targets(
    package: SocialTrialPackage,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return permutation-invariant symmetric and directional summaries.

    Directed traits are represented by their two-direction mean and absolute
    asymmetry.  This retains mutual-versus-unilateral structure without
    assigning biological meaning to arbitrary animal indices.
    """

    if package.edge_values.shape[1:3] != (2, 2):
        raise ValueError("invariant edge targets currently require N=2")
    values: list[np.ndarray] = []
    confidence: list[np.ndarray] = []
    names: list[str] = []
    for trait, name in enumerate(TRAIT_NAMES):
        forward = package.edge_values[:, 0, 1, trait]
        reverse = package.edge_values[:, 1, 0, trait]
        forward_conf = package.edge_confidence[:, 0, 1, trait]
        reverse_conf = package.edge_confidence[:, 1, 0, trait]
        if trait in SYMMETRIC_TRAITS:
            values.append(0.5 * (forward + reverse))
            confidence.append(0.5 * (forward_conf + reverse_conf))
            names.append(name)
        else:
            values.extend((0.5 * (forward + reverse), np.abs(forward - reverse)))
            confidence.extend(
                (
                    0.5 * (forward_conf + reverse_conf),
                    np.minimum(forward_conf, reverse_conf),
                )
            )
            names.extend((f"{name}_direction_mean", f"{name}_direction_asymmetry"))
    return (
        np.stack(values, axis=1).astype(np.float64),
        np.stack(confidence, axis=1).astype(np.float64),
        names,
    )


def weighted_regression_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
) -> dict[str, float]:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    valid = np.isfinite(truth) & np.isfinite(prediction) & (weight > 0)
    if not np.any(valid):
        return {"r2": float("nan"), "mae": float("nan"), "rmse": float("nan"), "weight": 0.0}
    y = truth[valid]
    y_hat = prediction[valid]
    w = weight[valid]
    w_sum = float(w.sum())
    mean = float(np.sum(w * y) / w_sum)
    residual = float(np.sum(w * np.square(y - y_hat)))
    total = float(np.sum(w * np.square(y - mean)))
    return {
        "r2": float(1.0 - residual / total) if total > 1e-12 else float("nan"),
        "mae": float(np.sum(w * np.abs(y - y_hat)) / w_sum),
        "rmse": float(np.sqrt(residual / w_sum)),
        "weight": w_sum,
    }


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def finite_mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array[np.isfinite(array)])) if np.isfinite(array).any() else float("nan")
