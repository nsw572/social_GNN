"""Audit node retention at raw, message-passing, and graph-pooling stages."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.linear_model import Ridge

from .evaluation import (
    LoadedEncoder,
    apply_native_edge_mode,
    load_encoder,
    load_trials,
    make_random_encoder,
    resolve_evaluation_device,
    weighted_regression_metrics,
    write_csv,
    write_json,
)
from .graph_builder import compose_edge_inputs
from .models import SocialV0


@dataclass(frozen=True)
class ProbeState:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray


class ProbeMLP(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def _state(x: np.ndarray, y: np.ndarray) -> ProbeState:
    x_mean, x_std = x.mean(axis=0), x.std(axis=0)
    y_mean, y_std = y.mean(axis=0), y.std(axis=0)
    x_std[x_std < 1e-8] = 1.0
    y_std[y_std < 1e-8] = 1.0
    return ProbeState(x_mean, x_std, y_mean, y_std)


def _normalize_x(value: np.ndarray, state: ProbeState) -> np.ndarray:
    return ((value - state.x_mean) / state.x_std).astype(np.float32)


def _normalize_y(value: np.ndarray, state: ProbeState) -> np.ndarray:
    return ((value - state.y_mean) / state.y_std).astype(np.float32)


def _fit_linear(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    internal_x: np.ndarray,
    internal_y: np.ndarray,
    final_train_x: np.ndarray,
    final_train_y: np.ndarray,
    validation_x: np.ndarray,
    *,
    alphas: Sequence[float],
) -> tuple[np.ndarray, dict[str, Any]]:
    state = _state(fit_x, fit_y)
    fit_z = _normalize_x(fit_x, state)
    internal_z = _normalize_x(internal_x, state)
    fit_target = _normalize_y(fit_y, state)
    scores: dict[str, float] = {}
    best: tuple[float, float] | None = None
    for alpha in alphas:
        model = Ridge(alpha=float(alpha)).fit(fit_z, fit_target)
        prediction = model.predict(internal_z) * state.y_std + state.y_mean
        score = float(
            np.mean(
                [
                    weighted_regression_metrics(
                        internal_y[:, index], prediction[:, index], np.ones(internal_y.shape[0])
                    )["r2"]
                    for index in range(internal_y.shape[1])
                ]
            )
        )
        scores[str(alpha)] = score
        if best is None or score > best[0]:
            best = (score, float(alpha))
    if best is None:
        raise RuntimeError("linear probe failed to select alpha")
    final_state = _state(final_train_x, final_train_y)
    model = Ridge(alpha=best[1]).fit(
        _normalize_x(final_train_x, final_state),
        _normalize_y(final_train_y, final_state),
    )
    prediction = model.predict(_normalize_x(validation_x, final_state))
    prediction = prediction * final_state.y_std + final_state.y_mean
    return prediction, {
        "selected_alpha": best[1],
        "internal_validation_r2": best[0],
        "internal_r2_by_alpha": scores,
    }


def _fit_mlp_epochs(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
) -> tuple[int, list[float]]:
    state = _state(train_x, train_y)
    x = _normalize_x(train_x, state)
    y = _normalize_y(train_y, state)
    vx = torch.from_numpy(_normalize_x(validation_x, state)).to(device)
    torch.manual_seed(seed)
    model = ProbeMLP(x.shape[1], y.shape[1], hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    best_epoch, best_score = 1, -np.inf
    history: list[float] = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(x.shape[0])
        for start in range(0, order.size, batch_size):
            indices = order[start : start + batch_size]
            prediction = model(torch.from_numpy(x[indices]).to(device))
            target = torch.from_numpy(y[indices]).to(device)
            loss = torch.square(prediction - target).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = model(vx).cpu().numpy() * state.y_std + state.y_mean
        score = float(
            np.mean(
                [
                    weighted_regression_metrics(
                        validation_y[:, index], prediction[:, index], np.ones(validation_y.shape[0])
                    )["r2"]
                    for index in range(validation_y.shape[1])
                ]
            )
        )
        history.append(score)
        if score > best_score:
            best_epoch, best_score = epoch, score
    return best_epoch, history


def _fit_mlp_final(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
) -> np.ndarray:
    state = _state(train_x, train_y)
    x = _normalize_x(train_x, state)
    y = _normalize_y(train_y, state)
    torch.manual_seed(seed)
    model = ProbeMLP(x.shape[1], y.shape[1], hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    for _epoch in range(epochs):
        model.train()
        order = rng.permutation(x.shape[0])
        for start in range(0, order.size, batch_size):
            indices = order[start : start + batch_size]
            prediction = model(torch.from_numpy(x[indices]).to(device))
            target = torch.from_numpy(y[indices]).to(device)
            loss = torch.square(prediction - target).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction = model(
            torch.from_numpy(_normalize_x(validation_x, state)).to(device)
        ).cpu().numpy()
    return prediction * state.y_std + state.y_mean


def _invariant_stage(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(dtype=value.dtype).unsqueeze(-1)
    mean = (value * weight).sum(dim=2) / weight.sum(dim=2).clamp_min(1.0)
    pair_valid = mask.all(dim=2).to(dtype=value.dtype).unsqueeze(-1)
    difference = torch.abs(value[:, :, 0] - value[:, :, 1]) * pair_valid
    return torch.cat((mean, difference), dim=-1)


@torch.inference_mode()
def _stage_representations(
    encoder: LoadedEncoder,
    package,
    *,
    device: torch.device,
    chunk_size: int,
    seed: int,
) -> dict[str, np.ndarray]:
    normalized = encoder.normalizer.transform(package)
    normalized = apply_native_edge_mode(
        normalized, encoder.native_edge_mode, rng=np.random.default_rng(seed)
    )
    backbone = encoder.model.encoder
    if not isinstance(backbone, SocialV0):
        raise TypeError("stage audit requires a SocialV0 encoder")
    chunks: dict[str, list[np.ndarray]] = {
        "input_projection": [],
        "gine_1": [],
        "gine_2": [],
        "graph_pool": [],
    }
    length = normalized.node_features.shape[0]
    for start in range(0, length, chunk_size):
        stop = min(length, start + chunk_size)
        nodes = torch.from_numpy(
            np.array(normalized.node_features[start:stop], copy=True)
        ).unsqueeze(0).to(device)
        mask = torch.from_numpy(
            np.array(normalized.node_mask[start:stop], copy=True)
        ).unsqueeze(0).to(device)
        values = torch.from_numpy(
            np.array(normalized.edge_values[start:stop], copy=True)
        ).unsqueeze(0).to(device)
        confidence = torch.from_numpy(
            np.array(normalized.edge_confidence[start:stop], copy=True)
        ).unsqueeze(0).to(device)
        graph, stages = backbone(
            nodes,
            compose_edge_inputs(values, confidence),
            node_mask=mask,
            return_stages=True,
        )
        names = ("input_projection", "gine_1", "gine_2")
        if len(stages) != len(names):
            raise ValueError(f"expected three node stages, got {len(stages)}")
        for name, stage in zip(names, stages):
            chunks[name].append(_invariant_stage(stage, mask)[0].cpu().numpy())
        chunks["graph_pool"].append(graph[0].cpu().numpy())
    return {name: np.concatenate(values).astype(np.float64) for name, values in chunks.items()}


def _stack(mapping: Mapping[str, np.ndarray], ids: Sequence[str], valid) -> np.ndarray:
    return np.concatenate([mapping[value][valid[value]] for value in ids], axis=0)


def _metric_rows(
    *,
    model: str,
    representation: str,
    probe: str,
    truth: np.ndarray,
    prediction: np.ndarray,
    feature_names: Sequence[str],
) -> list[dict[str, Any]]:
    if truth.ndim != 2 or truth.shape[1] % 2 != 0:
        raise ValueError(
            "node-retention target must be [samples, 2 * node_dim]"
        )
    node_dim = truth.shape[1] // 2
    if prediction.shape != truth.shape:
        raise ValueError(
            f"prediction shape {prediction.shape} does not match target {truth.shape}"
        )
    if len(feature_names) != truth.shape[1]:
        raise ValueError(
            f"received {len(feature_names)} feature names for {truth.shape[1]} targets"
        )
    groups = {
        "node_mean": np.arange(node_dim),
        "node_abs_difference": np.arange(node_dim, 2 * node_dim),
        "combined": np.arange(2 * node_dim),
    }
    rows: list[dict[str, Any]] = []
    per_feature: list[dict[str, Any]] = []
    for index, name in enumerate(feature_names):
        metrics = weighted_regression_metrics(
            truth[:, index], prediction[:, index], np.ones(truth.shape[0])
        )
        row = {
            "model": model,
            "representation": representation,
            "probe": probe,
            "target_group": "feature",
            "target": name,
            "target_variance": float(np.var(truth[:, index])),
            **metrics,
        }
        rows.append(row)
        per_feature.append(row)
    for name, indices in groups.items():
        selected = [per_feature[index] for index in indices]
        rows.append(
            {
                "model": model,
                "representation": representation,
                "probe": probe,
                "target_group": name,
                "target": "__aggregate__",
                "target_variance": float(np.mean([row["target_variance"] for row in selected])),
                "r2": float(np.mean([row["r2"] for row in selected])),
                "mae": float(np.mean([row["mae"] for row in selected])),
                "rmse": float(np.mean([row["rmse"] for row in selected])),
                "weight": float(truth.shape[0]),
            }
        )
    return rows


def evaluate_node_stage_retention(
    *,
    experiment_config: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    train_trials: Sequence[str],
    validation_trial: str,
    internal_validation_trial: str,
    device_name: str = "auto",
    seed: int = 20261013,
    include_random: bool = True,
    mlp_epochs: int = 30,
) -> dict[str, Any]:
    config, packages = load_trials(experiment_config)
    by_id = {package.trial_id: package for package in packages}
    requested = set(train_trials) | {validation_trial, internal_validation_trial}
    if unknown := requested - set(by_id):
        raise ValueError(f"unknown trial IDs: {sorted(unknown)}")
    if internal_validation_trial not in train_trials:
        raise ValueError("internal validation trial must be one of train_trials")
    fit_ids = [value for value in train_trials if value != internal_validation_trial]
    if not fit_ids:
        raise ValueError("at least one probe-fit trial is required")
    audit_packages = [package for package in packages if package.trial_id in requested]
    device = resolve_evaluation_device(device_name)
    first = packages[0]
    node_dims = {package.node_features.shape[-1] for package in packages}
    if len(node_dims) != 1:
        raise ValueError(
            f"all trials must share one node feature dimension, got {sorted(node_dims)}"
        )
    node_dim = first.node_features.shape[-1]
    trained = load_encoder(
        checkpoint,
        model_config=config.get("model", {}),
        node_dim=node_dim,
        number_of_nodes=first.node_features.shape[1],
        device=device,
        name="trained_graph_mean",
    )
    encoders = [trained]
    if include_random:
        encoders.append(
            make_random_encoder(
                reference=trained,
                model_config=config.get("model", {}),
                node_dim=node_dim,
                number_of_nodes=first.node_features.shape[1],
                device=device,
                seed=seed,
                name="random_untrained",
            )
        )

    target: dict[str, np.ndarray] = {}
    valid: dict[str, np.ndarray] = {}
    raw_ordered: dict[str, np.ndarray] = {}
    raw_invariant: dict[str, np.ndarray] = {}
    for package in audit_packages:
        nodes = package.node_features.astype(np.float64)
        node_target = np.concatenate(
            (nodes.mean(axis=1), np.abs(nodes[:, 0] - nodes[:, 1])), axis=1
        )
        target[package.trial_id] = node_target
        valid[package.trial_id] = package.node_mask.all(axis=1)
        raw_ordered[package.trial_id] = nodes.reshape(nodes.shape[0], -1)
        raw_invariant[package.trial_id] = node_target.copy()

    representation_sets: list[tuple[str, str, dict[str, np.ndarray]]] = [
        ("raw", "raw_ordered", raw_ordered),
        ("raw", "H0_exact_invariant", raw_invariant),
    ]
    shapes: dict[str, Any] = {}
    for encoder_index, encoder in enumerate(encoders):
        stage_maps: dict[str, dict[str, np.ndarray]] = {}
        for trial_index, package in enumerate(audit_packages):
            stages = _stage_representations(
                encoder,
                package,
                device=device,
                chunk_size=4096,
                seed=seed + encoder_index * 1000 + trial_index,
            )
            for stage, values in stages.items():
                stage_maps.setdefault(stage, {})[package.trial_id] = values
                shapes[f"{encoder.name}:{stage}:{package.trial_id}"] = list(values.shape)
        for stage, values in stage_maps.items():
            representation_sets.append((encoder.name, stage, values))

    final_train_y = _stack(target, train_trials, valid)
    fit_y = _stack(target, fit_ids, valid)
    internal_y = _stack(target, [internal_validation_trial], valid)
    validation_y = _stack(target, [validation_trial], valid)
    feature_names = [f"node_mean_feature_{value}" for value in range(node_dim)] + [
        f"node_abs_difference_feature_{value}" for value in range(node_dim)
    ]
    rows: list[dict[str, Any]] = []
    tuning: list[dict[str, Any]] = []
    for representation_index, (model_name, representation, mapping) in enumerate(representation_sets):
        fit_x = _stack(mapping, fit_ids, valid)
        internal_x = _stack(mapping, [internal_validation_trial], valid)
        final_train_x = _stack(mapping, train_trials, valid)
        validation_x = _stack(mapping, [validation_trial], valid)
        linear_prediction, linear_tuning = _fit_linear(
            fit_x,
            fit_y,
            internal_x,
            internal_y,
            final_train_x,
            final_train_y,
            validation_x,
            alphas=(0.01, 0.1, 1.0, 10.0, 100.0),
        )
        rows.extend(
            _metric_rows(
                model=model_name,
                representation=representation,
                probe="linear",
                truth=validation_y,
                prediction=linear_prediction,
                feature_names=feature_names,
            )
        )
        probe_seed = seed + representation_index * 101
        selected_epoch, history = _fit_mlp_epochs(
            fit_x,
            fit_y,
            internal_x,
            internal_y,
            device=device,
            seed=probe_seed,
            hidden_dim=64,
            epochs=mlp_epochs,
            batch_size=2048,
        )
        mlp_prediction = _fit_mlp_final(
            final_train_x,
            final_train_y,
            validation_x,
            device=device,
            seed=probe_seed,
            hidden_dim=64,
            epochs=selected_epoch,
            batch_size=2048,
        )
        rows.extend(
            _metric_rows(
                model=model_name,
                representation=representation,
                probe="mlp",
                truth=validation_y,
                prediction=mlp_prediction,
                feature_names=feature_names,
            )
        )
        tuning.append(
            {
                "model": model_name,
                "representation": representation,
                "linear": linear_tuning,
                "mlp_selected_epoch": selected_epoch,
                "mlp_internal_r2_by_epoch": history,
            }
        )

    # The exact deterministic transform is a non-learned ceiling.
    rows.extend(
        _metric_rows(
            model="raw",
            representation="H0_exact_invariant",
            probe="deterministic",
            truth=validation_y,
            prediction=validation_y.copy(),
            feature_names=feature_names,
        )
    )
    summary = [row for row in rows if row["target"] == "__aggregate__"]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "node_stage_probe_metrics.csv", rows)
    write_csv(output / "node_stage_probe_summary.csv", summary)
    manifest = {
        "status": "completed",
        "protocol": "social_gnn_node_stage_retention_v1",
        "checkpoint": str(Path(checkpoint).resolve()),
        "experiment_config": str(Path(experiment_config).resolve()),
        "train_trials": list(train_trials),
        "probe_fit_trials": fit_ids,
        "probe_internal_validation_trial": internal_validation_trial,
        "model_validation_trial": validation_trial,
        "a5_used": False,
        "device": str(device),
        "seed": seed,
        "representations": shapes,
        "tuning": tuning,
        "notes": [
            "H1/H2 probes use mean plus absolute node difference and are permutation invariant.",
            "Probe hyperparameters and MLP epoch are selected on one training trial, never A4_C4.",
            "The deterministic H0 transform is the exact ceiling; learned raw probes diagnose probe limitations.",
        ],
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-trial", action="append", required=True)
    parser.add_argument("--validation-trial", required=True)
    parser.add_argument("--internal-validation-trial", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--mlp-epochs", type=int, default=30)
    parser.add_argument("--no-random-control", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = evaluate_node_stage_retention(
        experiment_config=args.experiment_config,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        train_trials=args.train_trial,
        validation_trial=args.validation_trial,
        internal_validation_trial=args.internal_validation_trial,
        device_name=args.device,
        seed=args.seed,
        include_random=not args.no_random_control,
        mlp_epochs=args.mlp_epochs,
    )
    print(Path(args.output_dir).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
