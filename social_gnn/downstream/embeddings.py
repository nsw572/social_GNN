"""Frozen, clock-preserving Social-GNN embedding extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from social_gnn.evaluation import (
    apply_native_edge_mode,
    encode_package,
    load_encoder,
    load_trials,
    make_random_encoder,
)


@dataclass(frozen=True)
class FrozenEmbeddingTrial:
    """One full trial on its authoritative social clock."""

    trial_id: str
    graph_embeddings: np.ndarray
    patch_start_s: np.ndarray
    patch_end_s: np.ndarray
    valid_time_mask: np.ndarray
    encoder_checkpoint: str
    encoder_config: str

    def __post_init__(self) -> None:
        trial = self.trial_id or "<unknown>"
        embedding = np.asarray(self.graph_embeddings)
        start = np.asarray(self.patch_start_s)
        end = np.asarray(self.patch_end_s)
        valid = np.asarray(self.valid_time_mask)
        if embedding.ndim != 2 or embedding.shape[0] < 1:
            raise ValueError(f"{trial}: graph_embeddings must have shape [T,D]")
        length = embedding.shape[0]
        if start.shape != (length,) or end.shape != (length,):
            raise ValueError(f"{trial}: patch clocks must have shape [T]")
        if valid.shape != (length,) or valid.dtype.kind != "b":
            raise ValueError(f"{trial}: valid_time_mask must be boolean [T]")
        if not np.isfinite(embedding).all():
            raise ValueError(f"{trial}: graph_embeddings contain nonfinite values")
        if not np.isfinite(start).all() or not np.isfinite(end).all():
            raise ValueError(f"{trial}: patch clock contains nonfinite values")
        if np.any(end <= start):
            raise ValueError(f"{trial}: every patch must have positive duration")
        if np.any(np.diff(start) <= 0) or np.any(np.diff(end) <= 0):
            raise ValueError(f"{trial}: patch clock must be strictly chronological")
        for value in (embedding, start, end, valid):
            value.setflags(write=False)

    @property
    def patch_center_s(self) -> np.ndarray:
        return 0.5 * (self.patch_start_s + self.patch_end_s)

    def contiguous_valid_runs(
        self,
        *,
        gap_factor: float = 1.5,
    ) -> list[np.ndarray]:
        """Return original indices for valid runs without bridging clock gaps."""

        centers = self.patch_center_s
        valid_indices = np.flatnonzero(self.valid_time_mask)
        if valid_indices.size == 0:
            return []
        nominal_stride = float(np.median(np.diff(centers)))
        if not np.isfinite(nominal_stride) or nominal_stride <= 0:
            raise ValueError(f"{self.trial_id}: cannot derive a positive clock stride")
        breaks = np.ones(valid_indices.size, dtype=bool)
        if valid_indices.size > 1:
            consecutive = np.diff(valid_indices) == 1
            clock_contiguous = (
                np.diff(centers[valid_indices]) <= nominal_stride * gap_factor
            )
            breaks[1:] = ~(consecutive & clock_contiguous)
        starts = np.flatnonzero(breaks)
        stops = np.r_[starts[1:], valid_indices.size]
        return [valid_indices[a:b] for a, b in zip(starts, stops) if b > a]

    def model_sequences(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        runs = self.contiguous_valid_runs()
        return [self.graph_embeddings[index] for index in runs], runs

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            trial_id=np.asarray(self.trial_id),
            graph_embeddings=self.graph_embeddings,
            patch_start_s=self.patch_start_s,
            patch_end_s=self.patch_end_s,
            patch_center_s=self.patch_center_s,
            valid_time_mask=self.valid_time_mask,
            encoder_checkpoint=np.asarray(self.encoder_checkpoint),
            encoder_config=np.asarray(self.encoder_config),
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> "FrozenEmbeddingTrial":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                trial_id=str(data["trial_id"].item()),
                graph_embeddings=np.asarray(data["graph_embeddings"], dtype=np.float32),
                patch_start_s=np.asarray(data["patch_start_s"], dtype=np.float64),
                patch_end_s=np.asarray(data["patch_end_s"], dtype=np.float64),
                valid_time_mask=np.asarray(data["valid_time_mask"], dtype=bool),
                encoder_checkpoint=str(data["encoder_checkpoint"].item()),
                encoder_config=str(data["encoder_config"].item()),
            )


def load_embedding_directory(path: str | Path) -> list[FrozenEmbeddingTrial]:
    trials = [FrozenEmbeddingTrial.load(value) for value in sorted(Path(path).glob("*_g_t.npz"))]
    if not trials:
        raise FileNotFoundError(f"no *_g_t.npz files found in {Path(path)}")
    identifiers = [trial.trial_id for trial in trials]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"duplicate trial IDs in embedding directory: {identifiers}")
    return trials


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {name}, but CUDA is unavailable")
    return device


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def extract_frozen_embeddings(
    *,
    experiment_config: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    device_name: str = "auto",
    chunk_size: int = 4096,
    random_seed: int | None = None,
) -> dict[str, Any]:
    """Extract full-clock embeddings and keep invalid patches explicit."""

    config_path = Path(experiment_config).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config, packages = load_trials(config_path)
    first = packages[0]
    device = _device(device_name)
    reference_encoder = load_encoder(
        checkpoint_path,
        model_config=config.get("model", {}),
        node_dim=first.node_features.shape[-1],
        number_of_nodes=first.node_features.shape[1],
        device=device,
    )
    encoder = (
        reference_encoder
        if random_seed is None
        else make_random_encoder(
            reference=reference_encoder,
            model_config=config.get("model", {}),
            node_dim=first.node_features.shape[-1],
            number_of_nodes=first.node_features.shape[1],
            device=device,
            seed=random_seed,
            name=f"random_untrained_seed_{random_seed}",
        )
    )
    encoder_description = (
        str(checkpoint_path)
        if random_seed is None
        else f"random_untrained_seed_{random_seed}; reference={checkpoint_path}"
    )
    records: list[dict[str, Any]] = []
    for trial_index, package in enumerate(packages):
        normalized = encoder.normalizer.transform(package)
        normalized = apply_native_edge_mode(
            normalized,
            encoder.native_edge_mode,
            rng=np.random.default_rng(941 + trial_index),
        )
        embedding = encode_package(
            encoder, normalized, device=device, chunk_size=chunk_size
        )
        valid = np.asarray(package.node_mask, dtype=bool).all(axis=1)
        trial = FrozenEmbeddingTrial(
            trial_id=package.trial_id,
            graph_embeddings=embedding,
            patch_start_s=np.array(package.patch_start_s, copy=True),
            patch_end_s=np.array(package.patch_end_s, copy=True),
            valid_time_mask=np.array(valid, copy=True),
            encoder_checkpoint=encoder_description,
            encoder_config=str(config_path),
        )
        path = trial.save(output / f"{trial.trial_id}_g_t.npz")
        runs = trial.contiguous_valid_runs()
        records.append(
            {
                "trial_id": trial.trial_id,
                "path": str(path),
                "shape": list(trial.graph_embeddings.shape),
                "valid_timesteps": int(valid.sum()),
                "invalid_timesteps": int((~valid).sum()),
                "contiguous_sequences": len(runs),
            }
        )
    manifest = {
        "contract": "frozen_social_embedding_v1",
        "experiment_config": str(config_path),
        "encoder_checkpoint": str(checkpoint_path),
        "encoder_sha256": _sha256(checkpoint_path),
        "random_untrained_seed": random_seed,
        "encoder_name": encoder.name,
        "encoder_objective": encoder.objective,
        "encoder_epoch": encoder.epoch,
        "embedding_dim": int(records[0]["shape"][1]),
        "device": str(device),
        "split": config.get("split", {}),
        "trials": records,
        "notes": [
            "Rows remain in authoritative chronological order.",
            "Invalid patches remain in the NPZ and are excluded only through valid_time_mask.",
            "Temporal models receive separate contiguous runs and never bridge trials or invalid gaps.",
        ],
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument(
        "--random-seed",
        type=int,
        help="Export a deterministic untrained architecture control using this seed",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = extract_frozen_embeddings(
        experiment_config=args.experiment_config,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device_name=args.device,
        chunk_size=args.chunk_size,
        random_seed=args.random_seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
