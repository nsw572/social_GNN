"""Measure frozen patch-encoder parameter count and inference throughput."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from .evaluation import apply_native_edge_mode, encode_package, load_encoder, load_trials


def evaluate_encoder_cost(
    *,
    experiment_config: str | Path,
    checkpoints: list[str],
    output_dir: str | Path,
    trial_id: str = "A4_C4",
    device_name: str = "cuda:0",
    repeats: int = 5,
    chunk_size: int = 4096,
) -> dict[str, object]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    config, packages = load_trials(experiment_config)
    by_id = {value.trial_id: value for value in packages}
    if trial_id not in by_id:
        raise ValueError(f"unknown trial {trial_id!r}")
    package = by_id[trial_id]
    device = torch.device(device_name)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for specification in checkpoints:
        if "=" not in specification:
            raise ValueError("checkpoint must be NAME=PATH")
        name, path = specification.split("=", 1)
        encoder = load_encoder(
            path,
            model_config=config.get("model", {}),
            node_dim=package.node_features.shape[-1],
            number_of_nodes=package.node_features.shape[1],
            device=device,
            name=name,
        )
        normalized = encoder.normalizer.transform(package)
        normalized = apply_native_edge_mode(
            normalized,
            encoder.native_edge_mode,
            rng=np.random.default_rng(734),
        )
        encode_package(encoder, normalized, device=device, chunk_size=chunk_size)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        elapsed: list[float] = []
        for _repeat in range(repeats):
            started = time.perf_counter()
            embedding = encode_package(
                encoder, normalized, device=device, chunk_size=chunk_size
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed.append(time.perf_counter() - started)
        median = float(np.median(elapsed))
        rows.append(
            {
                "name": name,
                "objective": encoder.objective,
                "trial_id": trial_id,
                "patches": int(embedding.shape[0]),
                "embedding_dim": int(embedding.shape[1]),
                "parameter_count": int(encoder.parameter_count),
                "median_inference_s": median,
                "patches_per_second": float(embedding.shape[0] / median),
                "repeat_seconds": json.dumps(elapsed),
                "device": str(device),
                "peak_cuda_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda"
                    else None
                ),
                "checkpoint": str(Path(path).resolve()),
            }
        )
    path = output / "encoder_inference_cost.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "contract": "social_encoder_inference_cost_v1",
        "experiment_config": str(Path(experiment_config).resolve()),
        "trial_id": trial_id,
        "repeats": repeats,
        "chunk_size": chunk_size,
        "device": str(device),
        "results": rows,
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trial-id", default="A4_C4")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=4096)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate_encoder_cost(
        experiment_config=args.experiment_config,
        checkpoints=args.checkpoint,
        output_dir=args.output_dir,
        trial_id=args.trial_id,
        device_name=args.device,
        repeats=args.repeats,
        chunk_size=args.chunk_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
