"""Evaluate full-input unordered node-set reconstruction on held-out trials."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .data import collate_social_trials
from .evaluate_role_binding import _parse_checkpoint
from .evaluation import load_encoder, load_trials, resolve_evaluation_device, write_csv, write_json


@torch.inference_mode()
def evaluate_set_reconstruction(
    *,
    experiment_config: str | Path,
    checkpoints: Mapping[str, Path],
    trial_ids: Sequence[str],
    output_dir: str | Path,
    device_name: str,
) -> dict[str, Any]:
    config, packages = load_trials(experiment_config)
    by_id = {item.trial_id: item for item in packages}
    unknown = set(trial_ids) - set(by_id)
    if unknown:
        raise ValueError(f"unknown reconstruction trials: {sorted(unknown)}")
    device = resolve_evaluation_device(device_name)
    model_config = config.get("model", {})
    rows: list[dict[str, Any]] = []
    for name, path in checkpoints.items():
        encoder = load_encoder(
            path,
            model_config=model_config,
            node_dim=packages[0].node_dim,
            number_of_nodes=packages[0].animals,
            device=device,
            name=name,
        )
        target_mode = getattr(encoder.model, "target_mode", "summary")
        if target_mode not in {"node_set", "joint_set"}:
            rows.append(
                {
                    "model": name,
                    "trial_id": "",
                    "target_mode": target_mode,
                    "status": "not_applicable_summary_decoder",
                    "standardized_node_rmse": np.nan,
                    "physical_node_rmse": np.nan,
                    "physical_node_mae": np.nan,
                    "valid_patch_count": 0,
                }
            )
            continue
        for trial_id in trial_ids:
            raw = by_id[trial_id]
            normalized = encoder.normalizer.transform(raw)
            batch = collate_social_trials([normalized.to_torch_sample()])
            tensor_batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            output = encoder.model(tensor_batch, view="full")
            prediction = output["node_prediction"][0].cpu().numpy().astype(np.float64)
            target_z = normalized.node_features.astype(np.float64)
            valid = normalized.node_mask.all(axis=1)
            identity_error = np.mean(np.square(prediction - target_z), axis=(1, 2))
            swap_error = np.mean(np.square(prediction - target_z[:, ::-1]), axis=(1, 2))
            if target_mode == "joint_set":
                edge_prediction = output["edge_prediction"][0].cpu().numpy().astype(np.float64)
                edge_target = normalized.edge_values.astype(np.float64)
                edge_weight = normalized.edge_confidence.astype(np.float64)
                edge_identity = (
                    np.square(edge_prediction - edge_target) * edge_weight
                ).sum(axis=(1, 2, 3)) / np.maximum(edge_weight.sum(axis=(1, 2, 3)), 1e-12)
                swap_target = edge_target[:, ::-1, ::-1]
                swap_weight = edge_weight[:, ::-1, ::-1]
                edge_swap = (
                    np.square(edge_prediction - swap_target) * swap_weight
                ).sum(axis=(1, 2, 3)) / np.maximum(swap_weight.sum(axis=(1, 2, 3)), 1e-12)
                choose_swap = (swap_error + edge_swap) < (identity_error + edge_identity)
            else:
                choose_swap = swap_error < identity_error
            matched_z = target_z.copy()
            matched_raw = raw.node_features.astype(np.float64).copy()
            matched_z[choose_swap] = matched_z[choose_swap, ::-1]
            matched_raw[choose_swap] = matched_raw[choose_swap, ::-1]
            prediction_raw = (
                prediction * encoder.normalizer.node_std.reshape(1, 1, -1)
                + encoder.normalizer.node_mean.reshape(1, 1, -1)
            )
            difference_z = (prediction - matched_z)[valid]
            difference_raw = (prediction_raw - matched_raw)[valid]
            rows.append(
                {
                    "model": name,
                    "trial_id": trial_id,
                    "target_mode": target_mode,
                    "status": "evaluated",
                    "standardized_node_rmse": float(np.sqrt(np.mean(np.square(difference_z)))),
                    "physical_node_rmse": float(np.sqrt(np.mean(np.square(difference_raw)))),
                    "physical_node_mae": float(np.mean(np.abs(difference_raw))),
                    "valid_patch_count": int(valid.sum()),
                    "swapped_match_fraction": float(choose_swap[valid].mean()),
                }
            )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "unordered_node_set_reconstruction.csv", rows)
    manifest = {
        "status": "completed",
        "protocol": "social_gnn_unordered_node_set_reconstruction_v1",
        "trials": list(trial_ids),
        "notes": [
            "Assignment is selected independently per patch by the exact two-permutation minimum.",
            "Joint-set models choose one permutation using the combined standardized node and confidence-weighted edge error.",
            "Summary decoders are marked not applicable rather than compared using a different target.",
        ],
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--checkpoint", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--trial", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = evaluate_set_reconstruction(
        experiment_config=args.experiment_config,
        checkpoints=_parse_checkpoint(args.checkpoint),
        trial_ids=args.trial or ["A4_C4"],
        output_dir=args.output_dir,
        device_name=args.device,
    )
    print(manifest["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
