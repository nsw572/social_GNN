"""Frozen linear/nonlinear probes for Social-GNN node and edge retention."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.linear_model import Ridge

from .evaluation import (
    TRAIT_NAMES,
    apply_native_edge_mode,
    discover_encoder_checkpoints,
    encode_package,
    finite_mean,
    invariant_edge_targets,
    load_encoder,
    load_trials,
    make_random_encoder,
    resolve_evaluation_device,
    unique_edge_targets,
    weighted_regression_metrics,
    write_csv,
    write_json,
)


def _stack(mapping: Mapping[str, np.ndarray], ids: Sequence[str]) -> np.ndarray:
    return np.concatenate([mapping[trial_id] for trial_id in ids], axis=0)


def _weighted_target_scale(y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = np.maximum(w.sum(axis=0), 1e-12)
    mean = (w * y).sum(axis=0) / total
    variance = (w * np.square(y - mean)).sum(axis=0) / total
    std = np.sqrt(np.maximum(variance, 0.0))
    std[std < 1e-8] = 1.0
    return mean, std


def _fit_linear_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_w: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    validation_w: np.ndarray,
    *,
    alphas: Sequence[float],
) -> tuple[list[Ridge], dict[str, Any]]:
    x_mean = train_x.mean(axis=0)
    x_std = train_x.std(axis=0)
    x_std[x_std < 1e-8] = 1.0
    train_z = (train_x - x_mean) / x_std
    validation_z = (validation_x - x_mean) / x_std
    y_mean, y_std = _weighted_target_scale(train_y, train_w)
    train_target = (train_y - y_mean) / y_std

    best_alpha = None
    best_score = -np.inf
    alpha_scores: dict[str, float] = {}
    for alpha in alphas:
        scores: list[float] = []
        for output in range(train_y.shape[1]):
            model = Ridge(alpha=float(alpha))
            model.fit(
                train_z,
                train_target[:, output],
                sample_weight=train_w[:, output],
            )
            prediction = model.predict(validation_z) * y_std[output] + y_mean[output]
            scores.append(
                weighted_regression_metrics(
                    validation_y[:, output], prediction, validation_w[:, output]
                )["r2"]
            )
        score = finite_mean(scores)
        alpha_scores[str(alpha)] = score
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)
    if best_alpha is None:
        raise RuntimeError("linear probe failed to select a finite regularization value")
    models: list[Ridge] = []
    for output in range(train_y.shape[1]):
        model = Ridge(alpha=best_alpha)
        model.fit(
            train_z,
            train_target[:, output],
            sample_weight=train_w[:, output],
        )
        models.append(model)
    state = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "best_alpha": best_alpha,
        "validation_mean_r2_by_alpha": alpha_scores,
    }
    return models, state


def _predict_linear(models: Sequence[Ridge], state: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    z = (x - state["x_mean"]) / state["x_std"]
    columns = [
        model.predict(z) * state["y_std"][index] + state["y_mean"][index]
        for index, model in enumerate(models)
    ]
    return np.stack(columns, axis=1)


class _ProbeMLP(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def _fit_mlp_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_w: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    validation_w: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    hidden_dim: int = 32,
    epochs: int = 30,
    batch_size: int = 1024,
) -> tuple[_ProbeMLP, dict[str, Any]]:
    x_mean = train_x.mean(axis=0)
    x_std = train_x.std(axis=0)
    x_std[x_std < 1e-8] = 1.0
    y_mean, y_std = _weighted_target_scale(train_y, train_w)
    train_z = ((train_x - x_mean) / x_std).astype(np.float32)
    validation_z = ((validation_x - x_mean) / x_std).astype(np.float32)
    train_target = ((train_y - y_mean) / y_std).astype(np.float32)
    validation_target = ((validation_y - y_mean) / y_std).astype(np.float32)
    train_weight = train_w.astype(np.float32)
    validation_weight = validation_w.astype(np.float32)

    torch.manual_seed(seed)
    model = _ProbeMLP(train_x.shape[1], train_y.shape[1], hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(train_z.shape[0])
        for start in range(0, order.size, batch_size):
            indices = order[start : start + batch_size]
            x_tensor = torch.from_numpy(train_z[indices]).to(device)
            y_tensor = torch.from_numpy(train_target[indices]).to(device)
            w_tensor = torch.from_numpy(train_weight[indices]).to(device)
            prediction = model(x_tensor)
            loss = (torch.square(prediction - y_tensor) * w_tensor).sum() / w_tensor.sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            prediction = model(torch.from_numpy(validation_z).to(device))
            target = torch.from_numpy(validation_target).to(device)
            weight = torch.from_numpy(validation_weight).to(device)
            validation_loss = float(
                ((torch.square(prediction - target) * weight).sum() / weight.sum().clamp_min(1.0)).cpu()
            )
        history.append({"epoch": epoch + 1, "validation_weighted_mse": validation_loss})
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("MLP probe did not produce a finite validation loss")
    model.load_state_dict(best_state)
    state = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "best_validation_weighted_mse": best_loss,
        "history": history,
    }
    return model, state


@torch.inference_mode()
def _predict_mlp(
    model: _ProbeMLP,
    state: Mapping[str, Any],
    x: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray:
    z = ((x - state["x_mean"]) / state["x_std"]).astype(np.float32)
    prediction = model(torch.from_numpy(z).to(device)).cpu().numpy()
    return prediction * state["y_std"] + state["y_mean"]


def _metric_rows(
    *,
    model_name: str,
    input_mode: str,
    probe: str,
    target_kind: str,
    split: str,
    names: Sequence[str],
    truth: np.ndarray,
    prediction: np.ndarray,
    confidence: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for output, name in enumerate(names):
        base_trait = next((trait for trait in TRAIT_NAMES if name.startswith(trait)), name)
        subsets = {
            "confidence_weighted_all": confidence[:, output],
            "high_confidence_ge_0.75": confidence[:, output]
            * (confidence[:, output] >= 0.75),
            "low_confidence_lt_0.50": confidence[:, output]
            * ((confidence[:, output] > 0) & (confidence[:, output] < 0.50)),
        }
        for subset, weight in subsets.items():
            metrics = weighted_regression_metrics(
                truth[:, output], prediction[:, output], weight
            )
            rows.append(
                {
                    "model": model_name,
                    "input_mode": input_mode,
                    "probe": probe,
                    "target_kind": target_kind,
                    "split": split,
                    "target": name,
                    "base_trait": base_trait,
                    "subset": subset,
                    **metrics,
                }
            )
    return rows


def _plot_results(output_dir: Path, rows: list[dict[str, Any]], split: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [
        row
        for row in rows
        if row["target_kind"] == "edge"
        and row["split"] == split
        and row["probe"] == "linear"
        and row["subset"] == "confidence_weighted_all"
        and row["input_mode"] == "native"
    ]
    models = sorted({row["model"] for row in selected})
    figure, axis = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(TRAIT_NAMES))
    width = 0.8 / max(len(models), 1)
    for model_index, model in enumerate(models):
        values = [
            finite_mean(
                [row["r2"] for row in selected if row["model"] == model and row["base_trait"] == trait]
            )
            for trait in TRAIT_NAMES
        ]
        axis.bar(x + (model_index - (len(models) - 1) / 2) * width, values, width, label=model)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x, TRAIT_NAMES, rotation=35, ha="right")
    axis.set_ylabel("Confidence-weighted held-out R²")
    axis.set_title(f"Frozen linear edge probes ({split}; symmetric traits counted once)")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = output_dir / f"linear_edge_probe_{split}.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return [str(path.resolve())]


def evaluate_information_retention(
    *,
    experiment_config: str | Path,
    experiment_dir: str | Path,
    output_dir: str | Path,
    validation_trials: Sequence[str],
    test_trials: Sequence[str],
    exclude_trials: Sequence[str] = (),
    device_name: str = "auto",
    seed: int = 20261013,
    include_random: bool = True,
    random_control_count: int = 1,
    run_mlp: bool = True,
    mlp_epochs: int = 30,
    native_only: bool = False,
) -> dict[str, Any]:
    config, raw_packages = load_trials(experiment_config)
    package_by_id = {package.trial_id: package for package in raw_packages}
    validation_ids = list(validation_trials)
    test_ids = list(test_trials)
    excluded_ids = {str(value) for value in exclude_trials}
    overlap = set(validation_ids) & set(test_ids)
    if overlap:
        raise ValueError(f"validation and test trials overlap: {sorted(overlap)}")
    split_excluded_overlap = (set(validation_ids) | set(test_ids)) & excluded_ids
    if split_excluded_overlap:
        raise ValueError(
            "excluded trials overlap an evaluated split: "
            f"{sorted(split_excluded_overlap)}"
        )
    unknown = (set(validation_ids) | set(test_ids) | excluded_ids) - set(package_by_id)
    if unknown:
        raise ValueError(f"unknown probe split trial IDs: {sorted(unknown)}")
    train_ids = [
        package.trial_id
        for package in raw_packages
        if package.trial_id not in set(validation_ids) | set(test_ids) | excluded_ids
    ]
    if not train_ids or not validation_ids or not test_ids:
        raise ValueError("probe protocol requires non-empty disjoint train/validation/test trial sets")

    device = resolve_evaluation_device(device_name)
    model_config = config.get("model", {})
    node_dim = raw_packages[0].node_features.shape[-1]
    number_of_nodes = raw_packages[0].node_features.shape[1]
    checkpoints = discover_encoder_checkpoints(experiment_dir)
    encoders = []
    for name, path in checkpoints.items():
        loaded = load_encoder(
            path,
            model_config=model_config,
            node_dim=node_dim,
            number_of_nodes=number_of_nodes,
            device=device,
            name=name,
            vae_config=config.get("vae", {}),
        )
        encoders.append(loaded)
        if loaded.objective == "vae":
            encoders.append(
                load_encoder(
                    path,
                    model_config=model_config,
                    node_dim=node_dim,
                    number_of_nodes=number_of_nodes,
                    device=device,
                    name=f"{name}_pre_vae_g_t",
                    vae_config=config.get("vae", {}),
                    representation="g_t",
                )
            )
    if include_random:
        for random_index in range(random_control_count):
            encoders.append(
                make_random_encoder(
                    reference=encoders[0],
                    model_config=model_config,
                    node_dim=node_dim,
                    number_of_nodes=number_of_nodes,
                    device=device,
                    seed=seed + random_index * 10_007,
                    name=f"random_untrained_seed_{random_index}",
                    vae_config=config.get("vae", {}),
                )
            )

    edge_y: dict[str, np.ndarray] = {}
    edge_w: dict[str, np.ndarray] = {}
    invariant_edge_y: dict[str, np.ndarray] = {}
    invariant_edge_w: dict[str, np.ndarray] = {}
    node_ordered_y: dict[str, np.ndarray] = {}
    node_ordered_w: dict[str, np.ndarray] = {}
    node_invariant_y: dict[str, np.ndarray] = {}
    node_invariant_w: dict[str, np.ndarray] = {}
    edge_names: list[str] | None = None
    invariant_edge_names: list[str] | None = None
    node_ordered_names = [f"node_{animal}_feature_{feature}" for animal in range(number_of_nodes) for feature in range(node_dim)]
    node_invariant_names = [f"node_mean_feature_{feature}" for feature in range(node_dim)] + [
        f"node_abs_difference_feature_{feature}" for feature in range(node_dim)
    ]
    for package in raw_packages:
        values, confidence, names = unique_edge_targets(package)
        edge_y[package.trial_id] = values
        edge_w[package.trial_id] = confidence
        edge_names = names
        invariant_values, invariant_confidence, invariant_names = invariant_edge_targets(
            package
        )
        invariant_edge_y[package.trial_id] = invariant_values
        invariant_edge_w[package.trial_id] = invariant_confidence
        invariant_edge_names = invariant_names
        node_ordered_y[package.trial_id] = package.node_features.reshape(package.node_features.shape[0], -1).astype(np.float64)
        node_ordered_w[package.trial_id] = np.repeat(package.node_mask, node_dim, axis=1).astype(np.float64)
        if number_of_nodes != 2:
            raise ValueError("invariant node probe currently expects exactly two animals")
        node_invariant_y[package.trial_id] = np.concatenate(
            (
                package.node_features.mean(axis=1),
                np.abs(package.node_features[:, 0] - package.node_features[:, 1]),
            ),
            axis=1,
        ).astype(np.float64)
        pair_valid = np.all(package.node_mask, axis=1, keepdims=True).astype(np.float64)
        node_invariant_w[package.trial_id] = np.repeat(pair_valid, node_dim * 2, axis=1)
    assert edge_names is not None
    assert invariant_edge_names is not None

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    tuning_records: list[dict[str, Any]] = []

    representation_sets: list[tuple[str, str, dict[str, np.ndarray]]] = []
    for encoder_index, encoder in enumerate(encoders):
        normalized = [encoder.normalizer.transform(package) for package in raw_packages]
        for input_mode in (("native",) if native_only else ("native", "real")):
            embeddings: dict[str, np.ndarray] = {}
            for trial_index, package in enumerate(normalized):
                changed = (
                    apply_native_edge_mode(
                        package,
                        encoder.native_edge_mode,
                        rng=np.random.default_rng(seed + encoder_index * 1000 + trial_index),
                    )
                    if input_mode == "native"
                    else package
                )
                embeddings[package.trial_id] = encode_package(
                    encoder, changed, device=device
                ).astype(np.float64)
            representation_sets.append((encoder.name, input_mode, embeddings))

    # Raw-node and direct-fusion controls are diagnostics, not candidate GNNs.
    raw_node = {
        package.trial_id: package.node_features.reshape(package.node_features.shape[0], -1).astype(np.float64)
        for package in raw_packages
    }
    direct_fusion = {
        trial_id: np.concatenate((raw_node[trial_id], edge_y[trial_id], edge_w[trial_id]), axis=1)
        for trial_id in package_by_id
    }
    representation_sets.extend(
        [
            ("raw_node_baseline", "native", raw_node),
            ("direct_node_edge_fusion_diagnostic", "native", direct_fusion),
        ]
    )

    for representation_index, (model_name, input_mode, features) in enumerate(representation_sets):
        train_x = _stack(features, train_ids)
        validation_x = _stack(features, validation_ids)
        test_x = _stack(features, test_ids)
        for target_kind, targets, weights, names in (
            ("edge", edge_y, edge_w, edge_names),
            (
                "edge_invariant",
                invariant_edge_y,
                invariant_edge_w,
                invariant_edge_names,
            ),
            ("node_ordered", node_ordered_y, node_ordered_w, node_ordered_names),
            ("node_invariant", node_invariant_y, node_invariant_w, node_invariant_names),
        ):
            train_y = _stack(targets, train_ids)
            validation_y = _stack(targets, validation_ids)
            test_y = _stack(targets, test_ids)
            train_w = _stack(weights, train_ids)
            validation_w = _stack(weights, validation_ids)
            test_w = _stack(weights, test_ids)
            linear_models, linear_state = _fit_linear_probe(
                train_x,
                train_y,
                train_w,
                validation_x,
                validation_y,
                validation_w,
                alphas=(0.01, 0.1, 1.0, 10.0, 100.0),
            )
            tuning_records.append(
                {
                    "model": model_name,
                    "input_mode": input_mode,
                    "target_kind": target_kind,
                    "probe": "linear",
                    "selection": {
                        "best_alpha": linear_state["best_alpha"],
                        "validation_mean_r2_by_alpha": linear_state[
                            "validation_mean_r2_by_alpha"
                        ],
                    },
                }
            )
            for split, x, y, w in (
                ("validation", validation_x, validation_y, validation_w),
                ("test", test_x, test_y, test_w),
            ):
                prediction = _predict_linear(linear_models, linear_state, x)
                metric_rows.extend(
                    _metric_rows(
                        model_name=model_name,
                        input_mode=input_mode,
                        probe="linear",
                        target_kind=target_kind,
                        split=split,
                        names=names,
                        truth=y,
                        prediction=prediction,
                        confidence=w,
                    )
                )
            if run_mlp:
                mlp_model, mlp_state = _fit_mlp_probe(
                    train_x,
                    train_y,
                    train_w,
                    validation_x,
                    validation_y,
                    validation_w,
                    device=device,
                    seed=seed + representation_index * 17 + (
                        0 if target_kind == "edge" else (1 if target_kind == "node_ordered" else 2)
                    ),
                    epochs=mlp_epochs,
                )
                tuning_records.append(
                    {
                        "model": model_name,
                        "input_mode": input_mode,
                        "target_kind": target_kind,
                        "probe": "mlp",
                        "selection": {
                            "hidden_dim": mlp_state["hidden_dim"],
                            "epochs": mlp_state["epochs"],
                            "best_validation_weighted_mse": mlp_state[
                                "best_validation_weighted_mse"
                            ],
                        },
                    }
                )
                for split, x, y, w in (
                    ("validation", validation_x, validation_y, validation_w),
                    ("test", test_x, test_y, test_w),
                ):
                    prediction = _predict_mlp(
                        mlp_model, mlp_state, x, device=device
                    )
                    metric_rows.extend(
                        _metric_rows(
                            model_name=model_name,
                            input_mode=input_mode,
                            probe="mlp",
                            target_kind=target_kind,
                            split=split,
                            names=names,
                            truth=y,
                            prediction=prediction,
                            confidence=w,
                        )
                    )

    write_csv(output / "probe_metrics.csv", metric_rows)
    write_json(output / "probe_selection.json", tuning_records)
    plots = _plot_results(output, metric_rows, "validation")
    plots.extend(_plot_results(output, metric_rows, "test"))
    summary_rows: list[dict[str, Any]] = []
    for model_name, input_mode, _features in representation_sets:
        for probe in ("linear", "mlp") if run_mlp else ("linear",):
            for target_kind in (
                "edge",
                "edge_invariant",
                "node_ordered",
                "node_invariant",
            ):
                split_scores = {}
                for split in ("validation", "test"):
                    selected = [
                        row["r2"]
                        for row in metric_rows
                        if row["model"] == model_name
                        and row["input_mode"] == input_mode
                        and row["probe"] == probe
                        and row["target_kind"] == target_kind
                        and row["split"] == split
                        and row["subset"] == "confidence_weighted_all"
                    ]
                    split_scores[f"{split}_mean_r2"] = finite_mean(selected)
                summary_rows.append(
                    {
                        "model": model_name,
                        "input_mode": input_mode,
                        "probe": probe,
                        "target_kind": target_kind,
                        **split_scores,
                    }
                )
    write_csv(output / "probe_summary.csv", summary_rows)
    legacy_validation = set(config.get("split", {}).get("validation_trials", []))
    manifest = {
        "status": "completed",
        "protocol": "social_gnn_information_retention_v1",
        "experiment_config": str(Path(experiment_config).resolve()),
        "experiment_dir": str(Path(experiment_dir).resolve()),
        "output_dir": str(output.resolve()),
        "device": str(device),
        "seed": seed,
        "probe_split": {
            "train": train_ids,
            "validation": validation_ids,
            "test": test_ids,
            "excluded": sorted(excluded_ids),
        },
        "legacy_encoder_validation_trials": sorted(legacy_validation),
        "legacy_encoder_test_leakage_warning": bool(legacy_validation & set(test_ids)),
        "models": [
            {
                "name": encoder.name,
                "checkpoint": encoder.checkpoint_path,
                "epoch": encoder.epoch,
                "native_edge_mode": encoder.native_edge_mode,
                "parameter_count": encoder.parameter_count,
                "objective": encoder.objective,
                "representation": encoder.representation,
            }
            for encoder in encoders
        ],
        "notes": [
            "Encoder weights are frozen; probe hyperparameters are selected on whole validation trials only.",
            "Validation and test summaries are stored separately; architecture selection must use validation_mean_r2.",
            "A5_C5 was inspected previously, so its test rows are exploratory rather than a definitive independent test.",
            "Symmetric edge traits are represented once; directed traits retain both directions (11 outputs total).",
            "Native mode is the fair model comparison. Real mode is a pathway diagnostic and may expose untrained random edge weights in the node-only model.",
            "The direct fusion control contains target edge measurements and is only a diagnostic retention ceiling, never a selected model.",
            "Probe retention does not by itself establish downstream behavioral usefulness.",
        ],
        "outputs": {
            "metrics_csv": str((output / "probe_metrics.csv").resolve()),
            "summary_csv": str((output / "probe_summary.csv").resolve()),
            "selection_json": str((output / "probe_selection.json").resolve()),
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
    parser.add_argument("--validation-trial", action="append")
    parser.add_argument("--test-trial", action="append")
    parser.add_argument("--exclude-trial", action="append", default=[])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--mlp-epochs", type=int, default=30)
    parser.add_argument("--skip-mlp", action="store_true")
    parser.add_argument("--no-random-control", action="store_true")
    parser.add_argument("--random-control-count", type=int, default=1)
    parser.add_argument("--native-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir or str(
        Path("evaluation_runs")
        / f"information_retention_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    manifest = evaluate_information_retention(
        experiment_config=args.experiment_config,
        experiment_dir=args.experiment_dir,
        output_dir=output,
        validation_trials=args.validation_trial or ["A4_C4"],
        test_trials=args.test_trial or ["A5_C5"],
        exclude_trials=args.exclude_trial,
        device_name=args.device,
        seed=args.seed,
        include_random=not args.no_random_control,
        random_control_count=args.random_control_count,
        run_mlp=not args.skip_mlp,
        mlp_epochs=args.mlp_epochs,
        native_only=args.native_only,
    )
    print(manifest["output_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
