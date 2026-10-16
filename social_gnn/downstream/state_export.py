"""Export temporal states, bouts, raw-feature summaries, and diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from social_gnn.evaluation import TRAIT_NAMES, load_trials

from .embeddings import load_embedding_directory
from .evaluation import (
    DecodedTrial,
    bout_rows,
    decode_trial,
    normalized_transition_matrix,
    segmentation_metrics,
    transition_counts,
)
from .gaussian_hmm import GaussianHMM
from .gaussian_hsmm import GaussianHSMM
from .preprocessing import EmbeddingPreprocessor
from .visualization import save_diagnostic_plots


def load_temporal_model(path: str | Path):
    with np.load(path, allow_pickle=False) as data:
        model_type = str(data["model_type"].item())
    if model_type == "gaussian_hmm_v1":
        return GaussianHMM.load(path)
    if model_type == "gaussian_hsmm_v1":
        return GaussianHSMM.load(path)
    raise ValueError(f"unsupported temporal checkpoint type {model_type!r}")


def _patch_bout_fields(decoded: DecodedTrial, bouts: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    length = decoded.predicted_state.size
    bout_id = np.full(length, -1, dtype=np.int64)
    start = np.full(length, np.nan)
    end = np.full(length, np.nan)
    duration = np.full(length, np.nan)
    for row in bouts:
        index = np.arange(row["start_index"], row["end_index_inclusive"] + 1)
        bout_id[index] = row["bout_id"]
        start[index] = row["bout_start_s"]
        end[index] = row["bout_end_s"]
        duration[index] = row["bout_duration_s"]
    return {"bout_id": bout_id, "bout_start_s": start, "bout_end_s": end, "bout_duration_s": duration}


def _raw_state_summary(
    decoded: list[DecodedTrial],
    experiment_config: str | Path,
    n_states: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _config, packages = load_trials(experiment_config)
    package_by_id = {package.trial_id: package for package in packages}
    summary: list[dict[str, Any]] = []
    representatives: list[dict[str, Any]] = []
    for state in range(n_states):
        node_values: list[np.ndarray] = []
        edge_values: list[np.ndarray] = []
        edge_confidence: list[np.ndarray] = []
        candidates: list[tuple[float, str, int, float]] = []
        for value in decoded:
            package = package_by_id[value.trial.trial_id]
            index = np.flatnonzero(value.predicted_state == state)
            if index.size == 0:
                continue
            node_values.append(package.node_features[index])
            edge_values.append(package.edge_values[index])
            edge_confidence.append(package.edge_confidence[index])
            for patch in index:
                candidates.append(
                    (
                        float(value.state_confidence[patch]),
                        value.trial.trial_id,
                        int(patch),
                        float(value.trial.patch_center_s[patch]),
                    )
                )
        record: dict[str, Any] = {"state": state}
        if node_values:
            nodes = np.concatenate(node_values)
            edges = np.concatenate(edge_values)
            confidence = np.concatenate(edge_confidence)
            record["patches"] = int(nodes.shape[0])
            for animal in range(nodes.shape[1]):
                for feature in range(nodes.shape[2]):
                    record[f"node_{animal}_feature_{feature}_mean"] = float(
                        nodes[:, animal, feature].mean()
                    )
            for source, target in ((0, 1), (1, 0)):
                for trait, name in enumerate(TRAIT_NAMES):
                    weight = confidence[:, source, target, trait]
                    denominator = weight.sum()
                    record[f"edge_{source}_to_{target}_{name}_mean"] = (
                        float(
                            np.dot(edges[:, source, target, trait], weight)
                            / denominator
                        )
                        if denominator > 0
                        else float("nan")
                    )
                    record[f"edge_{source}_to_{target}_{name}_confidence"] = float(
                        weight.mean()
                    )
        else:
            record["patches"] = 0
        summary.append(record)
        for rank, candidate in enumerate(sorted(candidates, reverse=True)[:10], start=1):
            confidence_value, trial_id, patch, center = candidate
            representatives.append(
                {
                    "state": state,
                    "rank": rank,
                    "trial_id": trial_id,
                    "patch_index": patch,
                    "patch_center_s": center,
                    "state_confidence": confidence_value,
                }
            )
    return pd.DataFrame(summary), pd.DataFrame(representatives)


def export_states(
    *,
    embedding_dir: str | Path,
    checkpoint: str | Path,
    preprocessor_path: str | Path,
    output_dir: str | Path,
    experiment_config: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model, metadata = load_temporal_model(checkpoint)
    preprocessor = EmbeddingPreprocessor.load(preprocessor_path)
    trials = load_embedding_directory(embedding_dir)
    decoded = [decode_trial(model, preprocessor, trial) for trial in trials]
    all_bouts: list[dict[str, Any]] = []
    patch_files: list[str] = []
    for value in decoded:
        bouts = bout_rows(value)
        all_bouts.extend(bouts)
        fields = _patch_bout_fields(value, bouts)
        npz_path = output / f"{value.trial.trial_id}_states.npz"
        np.savez_compressed(
            npz_path,
            trial_id=np.asarray(value.trial.trial_id),
            patch_start_s=value.trial.patch_start_s,
            patch_end_s=value.trial.patch_end_s,
            patch_center_s=value.trial.patch_center_s,
            graph_embedding=value.trial.graph_embeddings,
            valid_time_mask=value.trial.valid_time_mask,
            predicted_state=value.predicted_state,
            state_posterior_probabilities=value.posterior,
            state_confidence=value.state_confidence,
            **fields,
        )
        table = pd.DataFrame(
            {
                "trial_id": value.trial.trial_id,
                "patch_index": np.arange(value.predicted_state.size),
                "patch_start_s": value.trial.patch_start_s,
                "patch_end_s": value.trial.patch_end_s,
                "patch_center_s": value.trial.patch_center_s,
                "valid_time_mask": value.trial.valid_time_mask,
                "predicted_state": value.predicted_state,
                "state_confidence": value.state_confidence,
                **fields,
            }
        )
        csv_path = output / f"{value.trial.trial_id}_states.csv"
        table.to_csv(csv_path, index=False)
        patch_files.extend((str(npz_path), str(csv_path)))
    bout_frame = pd.DataFrame(all_bouts)
    bout_frame.to_csv(output / "bouts.csv", index=False)
    counts = transition_counts(decoded, model.n_states)
    matrix = normalized_transition_matrix(counts)
    pd.DataFrame(counts).to_csv(output / "transition_counts.csv", index=False)
    pd.DataFrame(matrix).to_csv(output / "transition_probabilities.csv", index=False)
    state_summary = []
    for state in range(model.n_states):
        rows = bout_frame[bout_frame["state"] == state]
        occupancy = sum(np.sum(value.predicted_state == state) for value in decoded)
        state_summary.append(
            {
                "state": state,
                "occupancy_timesteps": int(occupancy),
                "occupancy_fraction": float(
                    occupancy
                    / sum(np.sum(value.predicted_state >= 0) for value in decoded)
                ),
                "number_of_bouts": int(rows.shape[0]),
                "mean_duration_s": float(rows["bout_duration_s"].mean()),
                "median_duration_s": float(rows["bout_duration_s"].median()),
            }
        )
    state_frame = pd.DataFrame(state_summary)
    if experiment_config is not None:
        raw, representatives = _raw_state_summary(decoded, experiment_config, model.n_states)
        state_frame = state_frame.merge(raw, on="state", how="left")
        representatives.to_csv(output / "representative_patches.csv", index=False)
    state_frame.to_csv(output / "state_summary.csv", index=False)
    plots = save_diagnostic_plots(decoded, output / "plots", model.n_states)
    metrics = segmentation_metrics(decoded, model.n_states)
    manifest = {
        "contract": "social_temporal_state_export_v1",
        "model_checkpoint": str(Path(checkpoint).resolve()),
        "preprocessor": str(Path(preprocessor_path).resolve()),
        "embedding_dir": str(Path(embedding_dir).resolve()),
        "model_metadata": metadata,
        "timing_convention": (
            "Bout boundaries are Voronoi boundaries between adjacent patch centers; "
            "this counts stride time once even if source patches overlap. Invalid gaps "
            "and trial boundaries terminate bouts."
        ),
        "metrics": metrics,
        "patch_files": patch_files,
        "bout_table": str((output / "bouts.csv").resolve()),
        "state_summary": str((output / "state_summary.csv").resolve()),
        "plots": plots,
        "warning": "State numbers are provisional and have no automatic behavioral names.",
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preprocessor", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-config")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = export_states(
        embedding_dir=args.embedding_dir,
        checkpoint=args.checkpoint,
        preprocessor_path=args.preprocessor,
        output_dir=args.output_dir,
        experiment_config=args.experiment_config,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
