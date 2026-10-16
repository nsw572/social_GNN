"""Embedding variance, effective dimension, and permutation-consistency audit."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .data import SocialTrialPackage
from .evaluation import (
    discover_encoder_checkpoints,
    encode_package,
    invariant_edge_targets,
    latent_scale,
    load_encoder,
    load_trials,
    make_random_encoder,
    resolve_evaluation_device,
    unique_edge_targets,
    write_csv,
    write_json,
)


def _permuted(package: SocialTrialPackage) -> SocialTrialPackage:
    permutation = np.asarray([1, 0])
    return SocialTrialPackage(
        trial_id=package.trial_id,
        node_features=package.node_features[:, permutation],
        node_mask=package.node_mask[:, permutation],
        edge_values=package.edge_values[:, permutation][:, :, permutation],
        edge_confidence=package.edge_confidence[:, permutation][:, :, permutation],
        patch_start_s=package.patch_start_s,
        patch_end_s=package.patch_end_s,
        identity=package.identity[permutation],
    )


def _geometry(embedding: np.ndarray) -> dict[str, float]:
    centered = embedding.astype(np.float64) - embedding.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(centered.shape[0] - 1, 1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    total = float(eigenvalues.sum())
    squared = float(np.square(eigenvalues).sum())
    participation = total * total / squared if squared > 1e-16 else 0.0
    probability = eigenvalues / total if total > 1e-16 else eigenvalues
    positive = probability > 0
    entropy_rank = (
        float(np.exp(-(probability[positive] * np.log(probability[positive])).sum()))
        if np.any(positive)
        else 0.0
    )
    return {
        "embedding_dim": int(embedding.shape[1]),
        "total_variance": total,
        "mean_dimension_variance": float(np.var(embedding, axis=0).mean()),
        "participation_ratio": participation,
        "entropy_effective_rank": entropy_rank,
        "numerical_rank": int(np.linalg.matrix_rank(centered, tol=1e-6)),
    }


def evaluate_representation_geometry(
    *,
    experiment_config: str | Path,
    experiment_dir: str | Path,
    output_dir: str | Path,
    device_name: str = "auto",
    random_control_count: int = 0,
    seed: int = 20261013,
) -> dict[str, Any]:
    config, raw_packages = load_trials(experiment_config)
    device = resolve_evaluation_device(device_name)
    model_config = config.get("model", {})
    checkpoints = discover_encoder_checkpoints(experiment_dir)
    encoders = [
        load_encoder(
            path,
            model_config=model_config,
            node_dim=raw_packages[0].node_features.shape[-1],
            number_of_nodes=raw_packages[0].node_features.shape[1],
            device=device,
            name=name,
        )
        for name, path in checkpoints.items()
    ]
    if random_control_count > 0:
        reference = encoders[0]
        for index in range(random_control_count):
            encoders.append(
                make_random_encoder(
                    reference=reference,
                    model_config=model_config,
                    node_dim=raw_packages[0].node_features.shape[-1],
                    number_of_nodes=raw_packages[0].node_features.shape[1],
                    device=device,
                    seed=seed + index * 10_007,
                    name=f"random_untrained_seed_{index}",
                )
            )
    rows: list[dict[str, Any]] = []
    for encoder in encoders:
        normalized = [encoder.normalizer.transform(package) for package in raw_packages]
        real_embeddings = {
            package.trial_id: encode_package(encoder, package, device=device)
            for package in normalized
        }
        scale = latent_scale(real_embeddings.values())
        for package in normalized:
            embedding = real_embeddings[package.trial_id]
            permuted = encode_package(encoder, _permuted(package), device=device)
            error = np.linalg.norm(embedding - permuted, axis=1) / scale
            rows.append(
                {
                    "model": encoder.name,
                    "objective": encoder.objective,
                    "trial_id": package.trial_id,
                    "parameter_count": encoder.parameter_count,
                    **_geometry(embedding),
                    "mean_permutation_error": float(error.mean()),
                    "max_permutation_error": float(error.max()),
                }
            )
        combined = np.concatenate(list(real_embeddings.values()), axis=0)
        rows.append(
            {
                "model": encoder.name,
                "objective": encoder.objective,
                "trial_id": "__all__",
                "parameter_count": encoder.parameter_count,
                **_geometry(combined),
                "mean_permutation_error": float(
                    np.mean(
                        [
                            row["mean_permutation_error"]
                            for row in rows
                            if row["model"] == encoder.name
                            and row["trial_id"] != "__all__"
                        ]
                    )
                ),
                "max_permutation_error": float(
                    max(
                        row["max_permutation_error"]
                        for row in rows
                        if row["model"] == encoder.name
                        and row["trial_id"] != "__all__"
                    )
                ),
            }
        )

    # Uncompressed references make their index dependence explicit.
    for package in raw_packages:
        ordered_edge, ordered_conf, _ = unique_edge_targets(package)
        invariant_edge, invariant_conf, _ = invariant_edge_targets(package)
        ordered = np.concatenate(
            (
                package.node_features.reshape(package.node_features.shape[0], -1),
                ordered_edge,
                ordered_conf,
            ),
            axis=1,
        )
        invariant = np.concatenate(
            (
                package.node_features.mean(axis=1),
                np.abs(package.node_features[:, 0] - package.node_features[:, 1]),
                invariant_edge,
                invariant_conf,
            ),
            axis=1,
        )
        rows.extend(
            (
                {
                    "model": "raw_ordered_fusion",
                    "objective": "none",
                    "trial_id": package.trial_id,
                    "parameter_count": 0,
                    **_geometry(ordered),
                    "mean_permutation_error": float("nan"),
                    "max_permutation_error": float("nan"),
                },
                {
                    "model": "raw_invariant_fusion",
                    "objective": "none",
                    "trial_id": package.trial_id,
                    "parameter_count": 0,
                    **_geometry(invariant),
                    "mean_permutation_error": 0.0,
                    "max_permutation_error": 0.0,
                },
            )
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "representation_geometry.csv", rows)
    manifest = {
        "status": "completed",
        "protocol": "social_gnn_representation_geometry_v1",
        "experiment_config": str(Path(experiment_config).resolve()),
        "experiment_dir": str(Path(experiment_dir).resolve()),
        "output_dir": str(output.resolve()),
        "device": str(device),
        "notes": [
            "Permutation error swaps node order, node masks, both edge axes, confidence, and identity order.",
            "Participation ratio and entropy effective rank are computed from the embedding covariance spectrum.",
            "Raw ordered fusion is intentionally not assigned a permutation error because its coordinate ordering changes semantically.",
        ],
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--random-control-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261013)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir or str(
        Path("evaluation_runs")
        / f"representation_geometry_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    manifest = evaluate_representation_geometry(
        experiment_config=args.experiment_config,
        experiment_dir=args.experiment_dir,
        output_dir=output,
        device_name=args.device,
        random_control_count=args.random_control_count,
        seed=args.seed,
    )
    print(manifest["output_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
