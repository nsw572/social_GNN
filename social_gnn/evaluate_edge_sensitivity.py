r"""Evaluate whether frozen Social-GNN embeddings respond to relational edges.

Example
-------
python -m social_gnn.evaluate_edge_sensitivity ^
  --experiment-config configs/social_gnn_experiment_6dim_v0.yaml ^
  --experiment-dir D:\...\experiment_20261013_134352
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .evaluation import (
    TRAIT_NAMES,
    apply_native_edge_mode,
    discover_encoder_checkpoints,
    displacement_summary,
    encode_package,
    finite_mean,
    latent_scale,
    load_encoder,
    load_trials,
    make_random_encoder,
    matched_replacement_indices,
    nonidentity_permutation,
    perturb_package,
    resolve_evaluation_device,
    write_csv,
    write_json,
)


def _plot_summaries(output_dir: Path, rows: list[dict[str, Any]]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[str] = []
    primary = [
        row
        for row in rows
        if row["perturbation"] in {"zero", "shuffled", "matched", "value_only", "confidence_only", "node_shuffled"}
    ]
    models = sorted({row["model"] for row in primary})
    perturbations = ["zero", "shuffled", "matched", "value_only", "confidence_only", "node_shuffled"]
    figure, axis = plt.subplots(figsize=(11, 5.5))
    width = 0.8 / max(len(models), 1)
    x = np.arange(len(perturbations))
    for model_index, model in enumerate(models):
        means = [
            finite_mean(
                [
                    row["mean_normalized_displacement"]
                    for row in primary
                    if row["model"] == model and row["perturbation"] == perturbation
                ]
            )
            for perturbation in perturbations
        ]
        axis.bar(x + (model_index - (len(models) - 1) / 2) * width, means, width, label=model)
    axis.set_xticks(x, perturbations, rotation=20, ha="right")
    axis.set_ylabel("Mean embedding displacement / latent RMS radius")
    axis.set_title("Frozen-encoder perturbation sensitivity (trial means)")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = output_dir / "edge_sensitivity_summary.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(str(path.resolve()))

    traits = [row for row in rows if str(row["perturbation"]).startswith("trait:")]
    figure, axis = plt.subplots(figsize=(12, 5.5))
    x = np.arange(8)
    for model in models:
        means = [
            finite_mean(
                [
                    row["mean_normalized_displacement"]
                    for row in traits
                    if row["model"] == model and row.get("trait_index") == trait
                ]
            )
            for trait in range(8)
        ]
        axis.plot(x, means, marker="o", label=model)
    axis.set_xticks(x, TRAIT_NAMES, rotation=35, ha="right")
    axis.set_ylabel("Normalized displacement")
    axis.set_title("Coupled per-trait replacement sensitivity")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    path = output_dir / "per_trait_sensitivity.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(str(path.resolve()))
    return paths


def evaluate_edge_sensitivity(
    *,
    experiment_config: str | Path,
    experiment_dir: str | Path,
    output_dir: str | Path,
    device_name: str = "auto",
    seed: int = 20261013,
    include_random: bool = True,
    candidate_count: int = 96,
    exclude_trials: tuple[str, ...] = (),
) -> dict[str, Any]:
    config, raw_packages = load_trials(experiment_config)
    excluded = set(exclude_trials)
    unknown = excluded - {package.trial_id for package in raw_packages}
    if unknown:
        raise ValueError(f"unknown excluded trial IDs: {sorted(unknown)}")
    raw_packages = [
        package for package in raw_packages if package.trial_id not in excluded
    ]
    if not raw_packages:
        raise ValueError("edge sensitivity evaluation has no trials after exclusions")
    device = resolve_evaluation_device(device_name)
    model_config = config.get("model", {})
    node_dim = raw_packages[0].node_features.shape[-1]
    number_of_nodes = raw_packages[0].node_features.shape[1]
    checkpoints = discover_encoder_checkpoints(experiment_dir)
    encoders = [
        load_encoder(
            path,
            model_config=model_config,
            node_dim=node_dim,
            number_of_nodes=number_of_nodes,
            device=device,
            name=name,
        )
        for name, path in checkpoints.items()
    ]
    if include_random:
        encoders.append(
            make_random_encoder(
                reference=encoders[0],
                model_config=model_config,
                node_dim=node_dim,
                number_of_nodes=number_of_nodes,
                device=device,
                seed=seed,
            )
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []

    # Select matched pairs once with the reference train-only normalization so
    # every encoder is judged on exactly the same real patch pairs.
    reference_packages = [encoders[0].normalizer.transform(p) for p in raw_packages]
    match_maps: dict[str, tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]] = {}
    broad_maps: dict[str, np.ndarray] = {}
    for trial_index, package in enumerate(reference_packages):
        rng = np.random.default_rng(seed + 1009 * (trial_index + 1))
        broad_maps[package.trial_id] = nonidentity_permutation(
            package.node_features.shape[0], rng
        )
        match_maps[package.trial_id] = matched_replacement_indices(
            package, rng=rng, candidate_count=candidate_count
        )

    for model_index, encoder in enumerate(encoders):
        normalized = [encoder.normalizer.transform(p) for p in raw_packages]
        real_embeddings = {
            package.trial_id: encode_package(encoder, package, device=device)
            for package in normalized
        }
        scale = latent_scale(real_embeddings.values())
        for trial_index, package in enumerate(normalized):
            trial_id = package.trial_id
            real = real_embeddings[trial_id]
            broad = broad_maps[trial_id]
            matched, similar, diagnostics = match_maps[trial_id]
            node_summary: dict[str, float] | None = None
            perturbations: list[tuple[str, Any]] = [
                ("zero", perturb_package(package, "zero", rng=np.random.default_rng(seed))),
                (
                    "shuffled",
                    perturb_package(
                        package,
                        "shuffled",
                        rng=np.random.default_rng(seed),
                        permutation=broad,
                    ),
                ),
                (
                    "matched",
                    perturb_package(
                        package,
                        "matched",
                        rng=np.random.default_rng(seed),
                        permutation=matched,
                    ),
                ),
                (
                    "value_only",
                    perturb_package(
                        package,
                        "value_only",
                        rng=np.random.default_rng(seed),
                        permutation=broad,
                    ),
                ),
                (
                    "confidence_only",
                    perturb_package(
                        package,
                        "confidence_only",
                        rng=np.random.default_rng(seed),
                        permutation=broad,
                    ),
                ),
                (
                    "node_shuffled",
                    perturb_package(
                        package,
                        "node_shuffled",
                        rng=np.random.default_rng(seed),
                        permutation=broad,
                    ),
                ),
                (
                    "native_input",
                    apply_native_edge_mode(
                        package,
                        encoder.native_edge_mode,
                        rng=np.random.default_rng(seed + trial_index + model_index * 100),
                    ),
                ),
            ]
            summaries: dict[str, dict[str, float]] = {}
            for perturbation, changed in perturbations:
                alternative = encode_package(encoder, changed, device=device)
                summary = displacement_summary(real, alternative, scale=scale)
                summaries[perturbation] = summary
                if perturbation == "node_shuffled":
                    node_summary = summary
                rows.append(
                    {
                        "model": encoder.name,
                        "trial_id": trial_id,
                        "perturbation": perturbation,
                        "trait_index": "",
                        "latent_scale": scale,
                        **summary,
                    }
                )
            node_denominator = max(
                float(node_summary["mean_normalized_displacement"] if node_summary else 0.0),
                1e-12,
            )
            for row in rows[-len(perturbations) :]:
                row["edge_to_node_sensitivity_ratio"] = (
                    row["mean_normalized_displacement"] / node_denominator
                    if row["perturbation"] != "node_shuffled"
                    else 1.0
                )

            for trait, trait_name in enumerate(TRAIT_NAMES):
                changed = perturb_package(
                    package,
                    "trait",
                    rng=np.random.default_rng(seed),
                    permutation=broad,
                    trait=trait,
                )
                alternative = encode_package(encoder, changed, device=device)
                summary = displacement_summary(real, alternative, scale=scale)
                rows.append(
                    {
                        "model": encoder.name,
                        "trial_id": trial_id,
                        "perturbation": f"trait:{trait_name}",
                        "trait_index": trait,
                        "latent_scale": scale,
                        "edge_to_node_sensitivity_ratio": summary[
                            "mean_normalized_displacement"
                        ]
                        / node_denominator,
                        **summary,
                    }
                )

            different_distance = np.linalg.norm(real - real[matched], axis=1) / scale
            similar_distance = np.linalg.norm(real - real[similar], axis=1) / scale
            pair_rows.append(
                {
                    "model": encoder.name,
                    "trial_id": trial_id,
                    "n_pairs": int(real.shape[0]),
                    "mean_node_distance": float(np.mean(diagnostics["node_distance"])),
                    "mean_edge_distance_different": float(np.mean(diagnostics["edge_distance"])),
                    "mean_edge_distance_similar": float(np.mean(diagnostics["control_edge_distance"])),
                    "mean_latent_distance_edge_different": float(np.mean(different_distance)),
                    "mean_latent_distance_edge_similar": float(np.mean(similar_distance)),
                    "matched_edge_separation_ratio": float(
                        np.mean(different_distance) / max(np.mean(similar_distance), 1e-12)
                    ),
                }
            )

    write_csv(output / "edge_sensitivity_by_trial.csv", rows)
    write_csv(output / "node_matched_edge_different.csv", pair_rows)
    plots = _plot_summaries(output, rows)
    manifest = {
        "status": "completed",
        "protocol": "social_gnn_edge_sensitivity_v1",
        "experiment_config": str(Path(experiment_config).resolve()),
        "experiment_dir": str(Path(experiment_dir).resolve()),
        "output_dir": str(output.resolve()),
        "device": str(device),
        "seed": seed,
        "models": [
            {
                "name": encoder.name,
                "checkpoint": encoder.checkpoint_path,
                "epoch": encoder.epoch,
                "native_edge_mode": encoder.native_edge_mode,
                "parameter_count": encoder.parameter_count,
            }
            for encoder in encoders
        ],
        "trials": [package.trial_id for package in raw_packages],
        "excluded_trials": sorted(excluded),
        "notes": [
            "All perturbations keep node inputs fixed except the explicit node_shuffled control.",
            "Shuffled and matched replacements move values and confidence together for both directed edges.",
            "Per-trait replacement couples both directions, closing the reverse-edge copy shortcut for symmetric traits.",
            "Displacement is normalized by the real-embedding RMS radius, so scalar latent inflation cannot improve the score.",
            "A large displacement is sensitivity evidence only, not proof of useful relational semantics.",
        ],
        "outputs": {
            "sensitivity_csv": str((output / "edge_sensitivity_by_trial.csv").resolve()),
            "matched_pairs_csv": str((output / "node_matched_edge_different.csv").resolve()),
            "plots": plots,
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--candidate-count", type=int, default=96)
    parser.add_argument("--no-random-control", action="store_true")
    parser.add_argument("--exclude-trial", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir or str(
        Path("evaluation_runs")
        / f"edge_sensitivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    manifest = evaluate_edge_sensitivity(
        experiment_config=args.experiment_config,
        experiment_dir=args.experiment_dir,
        output_dir=output,
        device_name=args.device,
        seed=args.seed,
        include_random=not args.no_random_control,
        candidate_count=args.candidate_count,
        exclude_trials=tuple(args.exclude_trial),
    )
    print(manifest["output_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
