"""Component 2: state-wise trait fingerprints for State Atlas v1."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .state_atlas_viz_common import (
    AtlasViewTrial,
    FINGERPRINT_CHANNELS,
    fingerprint_channel_values,
    iter_atlas_trials,
    robust_stats,
)


def summarize_state_fingerprints(
    trials: Iterable[AtlasViewTrial],
    *,
    n_states: int = 12,
) -> tuple[pd.DataFrame, dict[tuple[int, str], np.ndarray]]:
    """Return robust state x channel statistics plus underlying finite values."""

    trial_list = list(trials)
    rows: list[dict[str, Any]] = []
    distributions: dict[tuple[int, str], np.ndarray] = {}
    for state in range(n_states):
        for label, trait_index, source, target, kind in FINGERPRINT_CHANNELS:
            chunks: list[np.ndarray] = []
            eligible = 0
            usable = 0
            for trial in trial_list:
                state_mask = trial.valid_time_mask & (trial.predicted_state == state)
                values = fingerprint_channel_values(
                    trial, (label, trait_index, source, target, kind)
                )
                selected = values[state_mask]
                finite = selected[np.isfinite(selected)]
                chunks.append(finite)
                eligible += int(state_mask.sum())
                if kind == "quality":
                    usable += int(np.sum(finite > 0))
                else:
                    usable += int(len(finite))
            merged = np.concatenate(chunks) if chunks else np.asarray([], dtype=float)
            distributions[(state, label)] = merged
            rows.append(
                {
                    "state": state,
                    "channel": label,
                    "channel_kind": kind,
                    "trait_index": trait_index,
                    "source_index": source,
                    "target_index": target,
                    "eligible_count": eligible,
                    "finite_value_count": int(len(merged)),
                    "usable_support_count": usable,
                    "usable_support_fraction": usable / max(eligible, 1),
                    **robust_stats(merged),
                }
            )
    return pd.DataFrame(rows), distributions


def robust_standardized_medians(summary: pd.DataFrame) -> pd.DataFrame:
    raw = summary.pivot(index="state", columns="channel", values="median")
    ordered = [channel[0] for channel in FINGERPRINT_CHANNELS]
    raw = raw.reindex(index=range(12), columns=ordered)
    result = raw.copy()
    for column in result.columns:
        values = result[column].to_numpy(dtype=float)
        median = float(np.nanmedian(values))
        q25, q75 = np.nanquantile(values, [0.25, 0.75])
        scale = float(q75 - q25)
        result[column] = (values - median) / (scale if scale > 1e-12 else 1.0)
    return result


def _short_label(label: str) -> str:
    replacements = {
        "source_nose_to_target_centroid_distance": "nose→target centroid",
        "source_radial_approach_speed": "radial approach",
        "movement_direction_alignment": "movement alignment",
        "head_direction_alignment": "head alignment",
        "nose_to_nose_distance": "nose distance",
        "centroid_distance": "centroid distance",
        "pair_closing_speed": "pair closing",
        "source_faces_target": "faces target",
    }
    for source, target in replacements.items():
        label = label.replace(source, target)
    return label


def _save_heatmap(summary: pd.DataFrame, output: Path) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = summary.pivot(index="state", columns="channel", values="median").reindex(
        index=range(12), columns=[channel[0] for channel in FINGERPRINT_CHANNELS]
    )
    standardized = robust_standardized_medians(summary)
    fig, axis = plt.subplots(figsize=(18, 8))
    image = axis.imshow(
        standardized.to_numpy(),
        cmap="coolwarm",
        vmin=-2.5,
        vmax=2.5,
        aspect="auto",
    )
    axis.set_xticks(
        range(len(standardized.columns)),
        [_short_label(value) for value in standardized.columns],
        rotation=55,
        ha="right",
        fontsize=9,
    )
    axis.set_yticks(range(12), [str(value) for value in range(12)])
    axis.set_xlabel("Trait channel (directions retained explicitly)")
    axis.set_ylabel("Sticky-HMM state")
    axis.set_title(
        "State Fingerprint: robust-standardized state medians\n"
        "Cell text is the raw median; color is (median - cross-state median) / cross-state IQR"
    )
    for row in range(12):
        for column in range(len(standardized.columns)):
            raw_value = raw.iloc[row, column]
            text = "NA" if not np.isfinite(raw_value) else f"{raw_value:.2g}"
            color = "white" if abs(float(standardized.iloc[row, column])) > 1.35 else "black"
            axis.text(column, row, text, ha="center", va="center", fontsize=7, color=color)
    colorbar = fig.colorbar(image, ax=axis, shrink=0.82)
    colorbar.set_label("Cross-state robust standardized median")
    fig.tight_layout()
    png = output / "state_fingerprint_heatmap.png"
    fig.savefig(png, dpi=220)
    plt.close(fig)
    csv = output / "state_fingerprint_heatmap_raw_medians.csv"
    raw.to_csv(csv)
    return png, csv


def _sample(values: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    if len(values) <= limit:
        return values
    return values[rng.choice(len(values), size=limit, replace=False)]


def _save_distributions(
    distributions: dict[tuple[int, str], np.ndarray],
    output: Path,
    *,
    sample_per_state: int,
    random_seed: int,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(random_seed)
    fig, axes = plt.subplots(5, 3, figsize=(20, 27))
    axes_flat = axes.ravel()
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, 12))
    for axis_index, (label, _trait, _source, _target, _kind) in enumerate(FINGERPRINT_CHANNELS):
        axis = axes_flat[axis_index]
        values_by_state = [
            _sample(distributions[(state, label)], sample_per_state, rng)
            for state in range(12)
        ]
        positions = np.arange(12)
        nonempty_positions = [position for position, values in zip(positions, values_by_state) if len(values)]
        nonempty_values = [values for values in values_by_state if len(values)]
        if nonempty_values:
            violin = axis.violinplot(
                nonempty_values,
                positions=nonempty_positions,
                widths=0.82,
                showmeans=False,
                showmedians=False,
                showextrema=False,
                points=80,
            )
            for body, position in zip(violin["bodies"], nonempty_positions):
                body.set_facecolor(colors[position])
                body.set_edgecolor("black")
                body.set_alpha(0.72)
            medians = [np.median(values) for values in nonempty_values]
            axis.scatter(nonempty_positions, medians, s=12, c="black", zorder=3)
        axis.set_title(_short_label(label), fontsize=11)
        axis.set_xticks(positions, [str(state) for state in range(12)], fontsize=8)
        axis.set_xlabel("State")
        axis.grid(axis="y", alpha=0.2)
    for axis in axes_flat[len(FINGERPRINT_CHANNELS) :]:
        axis.axis("off")
    fig.suptitle(
        "State Fingerprint distributions (deterministic samples; black dot = median)",
        fontsize=17,
        y=0.997,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.992])
    path = output / "state_fingerprint_distributions.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def generate_state_fingerprint(
    atlas_root: str | Path,
    output_dir: str | Path,
    *,
    trial_ids: Sequence[str],
    sample_per_state: int = 2500,
    random_seed: int = 20261102,
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary, distributions = summarize_state_fingerprints(
        iter_atlas_trials(atlas_root, tuple(trial_ids))
    )
    summary_path = output / "state_fingerprint_summary.csv"
    summary.to_csv(summary_path, index=False)
    heatmap_path, raw_path = _save_heatmap(summary, output)
    distribution_path = _save_distributions(
        distributions,
        output,
        sample_per_state=sample_per_state,
        random_seed=random_seed,
    )
    return {
        "summary": str(summary_path.resolve()),
        "heatmap": str(heatmap_path.resolve()),
        "raw_median_matrix": str(raw_path.resolve()),
        "distributions": str(distribution_path.resolve()),
    }


__all__ = [
    "generate_state_fingerprint",
    "robust_standardized_medians",
    "summarize_state_fingerprints",
]
