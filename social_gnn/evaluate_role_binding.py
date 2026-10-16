"""Frozen probes for permutation-invariant cross-trait role correspondence.

The primary target is C_fr = (f_01-f_10)(r_01-r_10).  Two deliberately
limited extensions use the directed nose-to-partner-centroid distance.  They
test whether two directed measurements remain attached to the same animal;
they are information diagnostics, not behavioral labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .evaluate_information_retention import (
    _fit_linear_probe,
    _fit_mlp_probe,
    _metric_rows,
    _predict_linear,
    _predict_mlp,
    _stack,
)
from .evaluation import (
    encode_package,
    load_encoder,
    load_trials,
    resolve_evaluation_device,
    write_csv,
    write_json,
)


ROLE_TARGETS = (
    ("facing_x_radial_approach", 3, 5),
    ("facing_x_nose_to_partner_centroid", 3, 2),
    ("radial_approach_x_nose_to_partner_centroid", 5, 2),
)


def role_binding_targets(package) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return invariant cross-trait products and conservative confidence."""

    values: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for _name, first, second in ROLE_TARGETS:
        first_delta = package.edge_values[:, 0, 1, first] - package.edge_values[:, 1, 0, first]
        second_delta = package.edge_values[:, 0, 1, second] - package.edge_values[:, 1, 0, second]
        values.append(first_delta * second_delta)
        confidence = np.minimum.reduce(
            (
                package.edge_confidence[:, 0, 1, first],
                package.edge_confidence[:, 1, 0, first],
                package.edge_confidence[:, 0, 1, second],
                package.edge_confidence[:, 1, 0, second],
            )
        )
        weights.append(confidence)
    return (
        np.stack(values, axis=1).astype(np.float64),
        np.stack(weights, axis=1).astype(np.float64),
        [item[0] for item in ROLE_TARGETS],
    )


def marginal_invariant_inputs(package) -> np.ndarray:
    """Marginal summaries that intentionally omit cross-trait role binding."""

    node_mean = package.node_features.mean(axis=1)
    node_abs = np.abs(package.node_features[:, 0] - package.node_features[:, 1])
    edge_01 = package.edge_values[:, 0, 1]
    edge_10 = package.edge_values[:, 1, 0]
    conf_01 = package.edge_confidence[:, 0, 1]
    conf_10 = package.edge_confidence[:, 1, 0]
    return np.concatenate(
        (
            node_mean,
            node_abs,
            (edge_01 + edge_10) / 2.0,
            np.abs(edge_01 - edge_10),
            (conf_01 + conf_10) / 2.0,
            np.abs(conf_01 - conf_10),
        ),
        axis=1,
    ).astype(np.float64)


def ordered_raw_inputs(package) -> np.ndarray:
    return np.concatenate(
        (
            package.node_features.reshape(package.timesteps, -1),
            package.edge_values.reshape(package.timesteps, -1),
            package.edge_confidence.reshape(package.timesteps, -1),
        ),
        axis=1,
    ).astype(np.float64)


def _parse_checkpoint(values: Sequence[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--checkpoint must be NAME=PATH")
        name, raw_path = value.split("=", 1)
        path = Path(raw_path)
        if not name or not path.is_file():
            raise FileNotFoundError(f"invalid checkpoint specification {value!r}")
        output[name] = path
    return output


def _counterfactual_rows(
    model: str,
    embedding: np.ndarray,
    marginal: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
) -> list[dict[str, Any]]:
    """Match real observations on marginals but require a role-target contrast."""

    rows: list[dict[str, Any]] = []
    mx = (marginal - marginal.mean(axis=0)) / np.maximum(marginal.std(axis=0), 1e-8)
    gx = (embedding - embedding.mean(axis=0)) / np.maximum(embedding.std(axis=0), 1e-8)
    for output, (name, _first, _second) in enumerate(ROLE_TARGETS):
        valid = confidence[:, output] >= 0.5
        indices = np.flatnonzero(valid)
        if indices.size < 4:
            continue
        y = target[:, output]
        threshold = float(np.nanmedian(np.abs(y[indices] - np.nanmedian(y[indices]))))
        pairs: list[tuple[int, int, float]] = []
        # Deterministic capped subset prevents quadratic work from dominating.
        for source in indices[:: max(1, indices.size // 1000)]:
            candidates = indices[np.abs(y[indices] - y[source]) > max(threshold, 1e-8)]
            if candidates.size == 0:
                continue
            distance = np.mean(np.square(mx[candidates] - mx[source]), axis=1)
            target_index = int(candidates[int(np.argmin(distance))])
            pairs.append((int(source), target_index, float(np.min(distance))))
        if not pairs:
            continue
        graph_distance = np.asarray(
            [np.mean(np.square(gx[left] - gx[right])) for left, right, _ in pairs]
        )
        target_gap = np.asarray([abs(y[left] - y[right]) for left, right, _ in pairs])
        rows.append(
            {
                "model": model,
                "target": name,
                "real_pair_count": len(pairs),
                "median_marginal_distance": float(np.median([item[2] for item in pairs])),
                "median_embedding_distance": float(np.median(graph_distance)),
                "median_role_target_gap": float(np.median(target_gap)),
                "embedding_distance_target_gap_correlation": float(
                    np.corrcoef(graph_distance, target_gap)[0, 1]
                    if len(pairs) > 2 and np.std(graph_distance) > 0 and np.std(target_gap) > 0
                    else np.nan
                ),
            }
        )
    return rows


def evaluate_role_binding(
    *,
    experiment_config: str | Path,
    checkpoints: Mapping[str, Path],
    output_dir: str | Path,
    train_trials: Sequence[str],
    validation_trials: Sequence[str],
    evaluation_trials: Sequence[str],
    device_name: str,
    seed: int,
    mlp_epochs: int = 30,
) -> dict[str, Any]:
    config, packages = load_trials(experiment_config)
    by_id = {package.trial_id: package for package in packages}
    used = set(train_trials) | set(validation_trials) | set(evaluation_trials)
    unknown = used - set(by_id)
    if unknown:
        raise ValueError(f"unknown role-probe trials: {sorted(unknown)}")
    if len(used) != sum(map(len, (train_trials, validation_trials, evaluation_trials))):
        raise ValueError("role-probe train/validation/evaluation splits must be disjoint")
    device = resolve_evaluation_device(device_name)
    node_dim = packages[0].node_dim
    number_of_nodes = packages[0].animals
    model_config = config.get("model", {})
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

    targets: dict[str, np.ndarray] = {}
    weights: dict[str, np.ndarray] = {}
    marginals: dict[str, np.ndarray] = {}
    ordered: dict[str, np.ndarray] = {}
    names: list[str] | None = None
    for package in packages:
        targets[package.trial_id], weights[package.trial_id], names = role_binding_targets(package)
        marginals[package.trial_id] = marginal_invariant_inputs(package)
        ordered[package.trial_id] = ordered_raw_inputs(package)
    assert names is not None

    representations: list[tuple[str, dict[str, np.ndarray]]] = [
        ("raw_invariant_marginals", marginals),
        ("raw_ordered", ordered),
    ]
    checkpoint_manifest: list[dict[str, Any]] = []
    for encoder in encoders:
        embedding = {
            package.trial_id: encode_package(
                encoder,
                encoder.normalizer.transform(package),
                device=device,
            ).astype(np.float64)
            for package in packages
        }
        representations.append((encoder.name, embedding))
        checkpoint_manifest.append(
            {
                "name": encoder.name,
                "checkpoint": encoder.checkpoint_path,
                "epoch": encoder.epoch,
                "objective": encoder.objective,
                "parameter_count": encoder.parameter_count,
            }
        )

    train_y, val_y, eval_y = (_stack(targets, ids) for ids in (train_trials, validation_trials, evaluation_trials))
    train_w, val_w, eval_w = (_stack(weights, ids) for ids in (train_trials, validation_trials, evaluation_trials))
    rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    counterfactuals: list[dict[str, Any]] = []
    for index, (model_name, features) in enumerate(representations):
        train_x, val_x, eval_x = (_stack(features, ids) for ids in (train_trials, validation_trials, evaluation_trials))
        linear, linear_state = _fit_linear_probe(
            train_x, train_y, train_w, val_x, val_y, val_w,
            alphas=(0.01, 0.1, 1.0, 10.0, 100.0),
        )
        selections.append({"model": model_name, "probe": "linear", "best_alpha": linear_state["best_alpha"]})
        mlp, mlp_state = _fit_mlp_probe(
            train_x, train_y, train_w, val_x, val_y, val_w,
            device=device, seed=seed + index * 101, epochs=mlp_epochs,
        )
        selections.append(
            {
                "model": model_name,
                "probe": "mlp",
                "hidden_dim": mlp_state["hidden_dim"],
                "selected_epoch_count": mlp_state["epochs"],
                "best_validation_weighted_mse": mlp_state["best_validation_weighted_mse"],
            }
        )
        for split, x, truth, confidence in (
            ("validation", val_x, val_y, val_w),
            ("evaluation", eval_x, eval_y, eval_w),
        ):
            for probe, prediction in (
                ("linear", _predict_linear(linear, linear_state, x)),
                ("mlp", _predict_mlp(mlp, mlp_state, x, device=device)),
            ):
                rows.extend(
                    _metric_rows(
                        model_name=model_name,
                        input_mode="native",
                        probe=probe,
                        target_kind="role_binding",
                        split=split,
                        names=names,
                        truth=truth,
                        prediction=prediction,
                        confidence=confidence,
                    )
                )
        evaluation_features = _stack(features, evaluation_trials)
        evaluation_marginals = _stack(marginals, evaluation_trials)
        counterfactuals.extend(
            _counterfactual_rows(
                model_name, evaluation_features, evaluation_marginals, eval_y, eval_w
            )
        )

    # The deterministic ceiling is an identity computation, kept explicit.
    rows.extend(
        _metric_rows(
            model_name="raw_deterministic_ceiling",
            input_mode="native",
            probe="deterministic",
            target_kind="role_binding",
            split="evaluation",
            names=names,
            truth=eval_y,
            prediction=eval_y.copy(),
            confidence=eval_w,
        )
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "role_binding_metrics.csv", rows)
    write_csv(output / "matched_real_counterfactuals.csv", counterfactuals)
    write_json(output / "probe_selection.json", selections)
    manifest = {
        "status": "completed",
        "protocol": "social_gnn_cross_trait_role_binding_v1",
        "splits": {
            "probe_train": list(train_trials),
            "probe_validation": list(validation_trials),
            "evaluation": list(evaluation_trials),
            "explicitly_not_used": ["A5_C5"],
        },
        "targets": [
            {"name": name, "trait_indices": [first, second], "permutation_invariant": True}
            for name, first, second in ROLE_TARGETS
        ],
        "checkpoints": checkpoint_manifest,
        "notes": [
            "Products of two directed differences are invariant because both factors change sign under animal swap.",
            "Confidence is the minimum of all four directed measurements used by each target.",
            "Targets diagnose recoverable role correspondence and are not behavioral ground truth.",
            "The six upstream latent axes have no stable biological names, so no post-hoc node-motion role target was invented.",
            "Counterfactual pairs are matched real observations, not synthetic biological examples.",
        ],
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--checkpoint", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-trial", action="append", default=[])
    parser.add_argument("--validation-trial", action="append", default=[])
    parser.add_argument("--evaluation-trial", action="append", default=[])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--mlp-epochs", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = evaluate_role_binding(
        experiment_config=args.experiment_config,
        checkpoints=_parse_checkpoint(args.checkpoint),
        output_dir=args.output_dir,
        train_trials=args.train_trial or ["A1_C1", "A3_C3"],
        validation_trials=args.validation_trial or ["A4_B4"],
        evaluation_trials=args.evaluation_trial or ["A4_C4"],
        device_name=args.device,
        seed=args.seed,
        mlp_epochs=args.mlp_epochs,
    )
    print(manifest["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
