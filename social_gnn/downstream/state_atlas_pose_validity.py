"""Component 3: state x pose-validity decomposition for State Atlas v1."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .state_atlas_viz_common import (
    AtlasViewTrial,
    POSE_VALIDITY_CHANNELS,
    iter_atlas_trials,
    robust_stats,
)


def compute_pose_validity_summary(
    trials: Iterable[AtlasViewTrial],
    *,
    n_states: int = 12,
) -> pd.DataFrame:
    """Compute aggregate and trial-stratified pose validity by HMM state."""

    trial_list = list(trials)
    global_baselines: dict[str, float] = {}
    trial_baselines: dict[tuple[str, str], float] = {}
    for label, trait, source, target, _kind in POSE_VALIDITY_CHANNELS:
        valid_count = 0
        eligible_count = 0
        for trial in trial_list:
            eligible = trial.valid_time_mask
            valid = eligible & trial.edge_valid[:, source, target, trait]
            valid_count += int(valid.sum())
            eligible_count += int(eligible.sum())
            trial_baselines[(trial.trial_id, label)] = float(valid.sum() / max(eligible.sum(), 1))
        global_baselines[label] = valid_count / max(eligible_count, 1)

    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, list[AtlasViewTrial]]] = [
        ("ALL", trial_list),
        *((trial.trial_id, [trial]) for trial in trial_list),
    ]
    for scope, scope_trials in scopes:
        for state in range(n_states):
            for label, trait, source, target, kind in POSE_VALIDITY_CHANNELS:
                eligible_count = 0
                valid_count = 0
                confidence_chunks: list[np.ndarray] = []
                coverage_chunks: list[np.ndarray] = []
                for trial in scope_trials:
                    eligible = trial.valid_time_mask & (trial.predicted_state == state)
                    valid = eligible & trial.edge_valid[:, source, target, trait]
                    eligible_count += int(eligible.sum())
                    valid_count += int(valid.sum())
                    confidence_chunks.append(trial.edge_confidence[valid, source, target, trait])
                    coverage_chunks.append(trial.edge_coverage[valid, source, target, trait])
                confidence = np.concatenate(confidence_chunks) if confidence_chunks else np.asarray([])
                coverage = np.concatenate(coverage_chunks) if coverage_chunks else np.asarray([])
                validity = valid_count / max(eligible_count, 1)
                global_baseline = global_baselines[label]
                within_trial_baseline = (
                    global_baseline if scope == "ALL" else trial_baselines[(scope, label)]
                )
                confidence_stats = robust_stats(confidence)
                coverage_stats = robust_stats(coverage)
                rows.append(
                    {
                        "trial_scope": scope,
                        "state": state,
                        "pose_channel": label,
                        "channel_kind": kind,
                        "trait_index": trait,
                        "source_index": source,
                        "target_index": target,
                        "valid_count": valid_count,
                        "total_eligible_count": eligible_count,
                        "validity_fraction": validity,
                        "global_validity_baseline": global_baseline,
                        "delta_from_global_validity": validity - global_baseline,
                        "scope_validity_baseline": within_trial_baseline,
                        "delta_from_scope_validity": validity - within_trial_baseline,
                        "confidence_on_valid_median": confidence_stats["median"],
                        "confidence_on_valid_iqr": confidence_stats["iqr"],
                        "confidence_on_valid_p10": confidence_stats["p10"],
                        "confidence_on_valid_p90": confidence_stats["p90"],
                        "coverage_on_valid_median": coverage_stats["median"],
                        "coverage_on_valid_iqr": coverage_stats["iqr"],
                        "coverage_on_valid_p10": coverage_stats["p10"],
                        "coverage_on_valid_p90": coverage_stats["p90"],
                    }
                )
    return pd.DataFrame(rows)


def _short_label(label: str) -> str:
    return (
        label.replace("source_nose_to_target_centroid_distance", "nose→target center")
        .replace("source_faces_target", "faces target")
        .replace("nose_to_nose_distance", "nose distance")
        .replace("head_direction_alignment", "head alignment")
    )


def _matrix(summary: pd.DataFrame, value: str) -> pd.DataFrame:
    labels = [channel[0] for channel in POSE_VALIDITY_CHANNELS]
    aggregate = summary[summary.trial_scope.eq("ALL")]
    return aggregate.pivot(index="state", columns="pose_channel", values=value).reindex(
        index=range(12), columns=labels
    )


def _save_matrix_figure(
    matrix: pd.DataFrame,
    path: Path,
    *,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    percentage: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(12.5, 7.2))
    image = axis.imshow(matrix.to_numpy(), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    axis.set_xticks(
        range(len(matrix.columns)),
        [_short_label(value) for value in matrix.columns],
        rotation=42,
        ha="right",
    )
    axis.set_yticks(range(12), [str(state) for state in range(12)])
    axis.set_xlabel("Pose trait channel (directed channels remain separate)")
    axis.set_ylabel("Sticky-HMM state")
    axis.set_title(title)
    axis.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    axis.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.4)
    axis.tick_params(which="minor", bottom=False, left=False)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = float(matrix.iloc[row, column])
            if not np.isfinite(value):
                text = "NA"
            elif percentage:
                text = f"{value:+.1%}" if vmin < 0 else f"{value:.1%}"
            else:
                text = f"{value:.2f}"
            red, green, blue, _alpha = image.cmap(image.norm(value))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            color = "black" if luminance >= 0.54 else "white"
            axis.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color=color,
            )
    colorbar = fig.colorbar(image, ax=axis, shrink=0.85)
    colorbar.set_label(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def generate_pose_validity_matrix(
    atlas_root: str | Path,
    output_dir: str | Path,
    *,
    trial_ids: Sequence[str],
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = compute_pose_validity_summary(
        iter_atlas_trials(atlas_root, tuple(trial_ids))
    )
    summary_path = output / "state_pose_validity_summary.csv"
    summary.to_csv(summary_path, index=False)

    absolute = _matrix(summary, "validity_fraction")
    delta = _matrix(summary, "delta_from_global_validity")
    confidence = _matrix(summary, "confidence_on_valid_median")
    absolute.to_csv(output / "state_pose_validity_absolute.csv")
    delta.to_csv(output / "state_pose_validity_delta_from_global.csv")
    confidence.to_csv(output / "state_pose_confidence_on_valid.csv")

    absolute_path = output / "state_pose_validity_absolute.png"
    delta_path = output / "state_pose_validity_delta_from_global.png"
    confidence_path = output / "state_pose_confidence_on_valid.png"
    _save_matrix_figure(
        absolute,
        absolute_path,
        title="State × pose validity fraction",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        percentage=True,
    )
    bound = max(float(np.nanmax(np.abs(delta.to_numpy()))), 0.05)
    _save_matrix_figure(
        delta,
        delta_path,
        title="State pose validity minus channel-specific global baseline",
        cmap="RdBu_r",
        vmin=-bound,
        vmax=bound,
        percentage=True,
    )
    _save_matrix_figure(
        confidence,
        confidence_path,
        title="Median confidence among valid pose observations",
        cmap="Purples",
        vmin=0.0,
        vmax=1.0,
        percentage=False,
    )
    return {
        "summary": str(summary_path.resolve()),
        "absolute": str(absolute_path.resolve()),
        "delta_from_global": str(delta_path.resolve()),
        "confidence_on_valid": str(confidence_path.resolve()),
    }


__all__ = ["compute_pose_validity_summary", "generate_pose_validity_matrix"]
