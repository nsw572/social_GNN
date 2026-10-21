"""Aggregate downstream runs, aligned partition agreement, and common plots."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix


KEYS = ["encoder", "encoder_seed", "algorithm", "downstream_seed", "K", "max_duration_patches"]


def _read_tables(roots: Iterable[str | Path], filename: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for root in roots:
        for path in sorted(Path(root).resolve().rglob(filename)):
            frame = pd.read_csv(path)
            frame["source_table"] = str(path)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _assignment_path(row: pd.Series, trial_id: str) -> Path:
    files = json.loads(row["assignment_files"])
    matches = [Path(value) for value in files if value.endswith(f"{trial_id}_states.npz")]
    if len(matches) != 1:
        raise ValueError(f"expected one assignment for {trial_id}, found {matches}")
    return matches[0]


def _load_labels(row: pd.Series, trial_id: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(_assignment_path(row, trial_id), allow_pickle=False) as data:
        labels = np.asarray(data["predicted_state"], dtype=np.int64)
        valid = np.asarray(data["valid_time_mask"], dtype=bool)
        center = np.asarray(data["patch_center_s"], dtype=np.float64)
    return labels[valid], center[valid]


def _run_id(row: pd.Series) -> str:
    duration = "na" if pd.isna(row["max_duration_patches"]) else str(int(row["max_duration_patches"]))
    return (
        f"{row['encoder']}|e{int(row['encoder_seed'])}|{row['algorithm']}|"
        f"k{int(row['K'])}|s{int(row['downstream_seed'])}|d{duration}"
    )


def _occupancy_rows(completed: pd.DataFrame, trial_id: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for _, row in completed.iterrows():
        labels, _center = _load_labels(row, trial_id)
        counts = np.bincount(labels, minlength=int(row["K"]))
        fraction = counts / max(counts.sum(), 1)
        for state, value in enumerate(fraction):
            result.append(
                {
                    **{key: row[key] for key in KEYS},
                    "run_id": _run_id(row),
                    "trial_id": trial_id,
                    "state": state,
                    "patches": int(counts[state]),
                    "occupancy_fraction": float(value),
                }
            )
    return result


def _partition_stability(completed: pd.DataFrame, trial_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouping = ["encoder", "algorithm", "K", "max_duration_patches"]
    normalized = completed.copy()
    normalized["max_duration_patches"] = normalized["max_duration_patches"].fillna(-1)
    for key, group in normalized.groupby(grouping, dropna=False, sort=True):
        labels = [(_run_id(row), *_load_labels(row, trial_id)) for _, row in group.iterrows()]
        ari: list[float] = []
        nmi: list[float] = []
        for (_left_id, left, left_time), (_right_id, right, right_time) in combinations(labels, 2):
            if left.shape != right.shape or not np.allclose(left_time, right_time):
                raise ValueError("partition stability requires identical aligned patch clocks")
            ari.append(adjusted_rand_score(left, right))
            nmi.append(normalized_mutual_info_score(left, right))
        rows.append(
            {
                "encoder": key[0],
                "algorithm": key[1],
                "K": int(key[2]),
                "max_duration_patches": None if key[3] == -1 else int(key[3]),
                "runs": len(labels),
                "pairwise_comparisons": len(ari),
                "mean_pairwise_ari": float(np.mean(ari)) if ari else float("nan"),
                "mean_pairwise_nmi": float(np.mean(nmi)) if nmi else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _cross_agreement(
    completed: pd.DataFrame, trial_id: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare algorithms/encoders at matched K and random seeds."""

    rows: list[dict[str, Any]] = []
    contingency_rows: list[dict[str, Any]] = []
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for (_k, encoder_seed, downstream_seed), group in completed.groupby(
        ["K", "encoder_seed", "downstream_seed"], sort=True
    ):
        values = list(group.iterrows())
        for (_left_index, left), (_right_index, right) in combinations(values, 2):
            left_id, right_id = _run_id(left), _run_id(right)
            if left_id not in cache:
                cache[left_id] = _load_labels(left, trial_id)
            if right_id not in cache:
                cache[right_id] = _load_labels(right, trial_id)
            left_labels, left_time = cache[left_id]
            right_labels, right_time = cache[right_id]
            if left_labels.shape != right_labels.shape or not np.allclose(left_time, right_time):
                raise ValueError("cross-model agreement requires aligned patch clocks")
            rows.append(
                {
                    "K": int(left["K"]),
                    "encoder_seed": int(encoder_seed),
                    "downstream_seed": int(downstream_seed),
                    "left_run": left_id,
                    "right_run": right_id,
                    "left_encoder": left["encoder"],
                    "right_encoder": right["encoder"],
                    "left_algorithm": left["algorithm"],
                    "right_algorithm": right["algorithm"],
                    "ari": float(adjusted_rand_score(left_labels, right_labels)),
                    "nmi": float(
                        normalized_mutual_info_score(left_labels, right_labels)
                    ),
                }
            )
            table = contingency_matrix(left_labels, right_labels)
            for left_state, right_state in zip(*np.nonzero(table)):
                contingency_rows.append(
                    {
                        "K": int(left["K"]),
                        "encoder_seed": int(encoder_seed),
                        "downstream_seed": int(downstream_seed),
                        "left_run": left_id,
                        "right_run": right_id,
                        "left_state": int(left_state),
                        "right_state": int(right_state),
                        "patches": int(table[left_state, right_state]),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(contingency_rows)


def _aggregate_runs(completed: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    normalized = completed.copy()
    normalized["max_duration_patches"] = normalized["max_duration_patches"].fillna(-1)
    metrics = [
        "train_fit_per_patch",
        "validation_fit_per_patch",
        "silhouette_validation",
        "effective_entropy_states",
        "largest_state_occupancy",
        "near_empty_states",
        "bouts_per_minute",
        "median_bout_duration_s",
        "bout_duration_q10_s",
        "bout_duration_q90_s",
        "fraction_one_timestep_bouts",
        "fraction_bouts_at_duration_cap",
        "observed_mean_bout_patches",
        "fitted_duration_expected_patches",
        "fitted_to_observed_duration_ratio",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in normalized.groupby(
        ["encoder", "algorithm", "K", "max_duration_patches"], sort=True
    ):
        row: dict[str, Any] = {
            "encoder": key[0],
            "algorithm": key[1],
            "K": int(key[2]),
            "max_duration_patches": None if key[3] == -1 else int(key[3]),
            "runs": int(group.shape[0]),
            "encoder_seeds": int(group["encoder_seed"].nunique()),
            "downstream_seeds": int(group["downstream_seed"].nunique()),
            "converged_fraction": float(group["converged"].astype(float).mean()),
        }
        for metric in metrics:
            if metric in group:
                row[f"{metric}_mean"] = float(group[metric].mean())
                row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        rows.append(row)
    summary = pd.DataFrame(rows)
    if not stability.empty:
        merge = stability.copy()
        merge["max_duration_patches"] = merge["max_duration_patches"].fillna(-1)
        summary["max_duration_patches"] = summary["max_duration_patches"].fillna(-1)
        summary = summary.merge(
            merge,
            on=["encoder", "algorithm", "K", "max_duration_patches"],
            how="left",
            suffixes=("", "_stability"),
        )
        summary["max_duration_patches"] = summary["max_duration_patches"].replace(-1, np.nan)
    return summary


def _line_plot(summary: pd.DataFrame, metric: str, output: Path, ylabel: str) -> None:
    if metric not in summary or summary[metric].notna().sum() == 0:
        return
    encoders = list(summary["encoder"].drop_duplicates())
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for axis, encoder in zip(axes.flat, encoders):
        subset = summary[summary["encoder"] == encoder]
        for (algorithm, duration), group in subset.groupby(
            ["algorithm", "max_duration_patches"], dropna=False
        ):
            label = algorithm if pd.isna(duration) else f"{algorithm} Dmax={int(duration)}"
            axis.plot(group["K"], group[metric], marker="o", label=label)
        axis.set_title(encoder)
        axis.set_xlabel("Requested K")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _agreement_heatmap(agreement: pd.DataFrame, output: Path) -> None:
    if agreement.empty:
        return
    values = agreement.copy()
    values["left"] = values["left_encoder"] + "/" + values["left_algorithm"]
    values["right"] = values["right_encoder"] + "/" + values["right_algorithm"]
    both = pd.concat(
        (
            values[["left", "right", "ari"]],
            values.rename(columns={"left": "right", "right": "left"})[
                ["left", "right", "ari"]
            ],
        ),
        ignore_index=True,
    )
    matrix = both.pivot_table(index="left", columns="right", values="ari", aggfunc="mean")
    labels = sorted(set(matrix.index) | set(matrix.columns))
    matrix = matrix.reindex(index=labels, columns=labels)
    np.fill_diagonal(matrix.values, 1.0)
    figure, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(matrix.values, vmin=0, vmax=1, cmap="viridis")
    axis.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    axis.set_yticks(range(len(labels)), labels, fontsize=7)
    axis.set_title("Mean A4_C4 partition agreement (matched K/seeds)")
    figure.colorbar(image, ax=axis, label="ARI")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _occupancy_plot(occupancy: pd.DataFrame, output: Path) -> None:
    if occupancy.empty:
        return
    grouped = (
        occupancy.groupby(["encoder", "algorithm", "state"], as_index=False)[
            "occupancy_fraction"
        ].mean()
    )
    pivot = grouped.pivot_table(
        index=["encoder", "algorithm"], columns="state", values="occupancy_fraction", fill_value=0
    )
    axis = pivot.plot(kind="bar", stacked=True, figsize=(14, 7), colormap="tab20")
    axis.set_ylabel("Mean occupancy fraction")
    axis.set_title("State occupancy (state IDs are run-local)")
    axis.legend(title="Run-local state", ncol=2, fontsize=7)
    axis.figure.tight_layout()
    axis.figure.savefig(output, dpi=180)
    plt.close(axis.figure)


def _bout_plot(bouts: pd.DataFrame, output: Path) -> None:
    if bouts.empty or "bout_duration_s" not in bouts:
        return
    algorithms = list(bouts["algorithm"].drop_duplicates())
    values = []
    rng = np.random.default_rng(813)
    for value in algorithms:
        duration = bouts.loc[bouts["algorithm"] == value, "bout_duration_s"].dropna().to_numpy()
        if duration.size > 5000:
            duration = rng.choice(duration, 5000, replace=False)
        values.append(duration)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.boxplot(values, labels=algorithms, showfliers=False)
    axis.set_ylabel("Median bout duration (s)")
    axis.set_title("Bout-duration distribution across fitted runs")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _read_bouts(completed: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, row in completed.iterrows():
        path = Path(row["bout_table"])
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        for key in KEYS:
            frame[key] = row[key]
        frame["run_id"] = _run_id(row)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _chronological_plot(completed: pd.DataFrame, trial_id: str, output: Path) -> None:
    candidates = completed[
        (completed["encoder"] == "graph_hybrid_p2")
        & (completed["K"] == 6)
        & (completed["downstream_seed"] == completed["downstream_seed"].min())
        & (completed["encoder_seed"] == completed["encoder_seed"].min())
    ].copy()
    if candidates.empty:
        return
    rows = []
    for algorithm in ("kmeans", "gmm", "hmm", "hsmm"):
        subset = candidates[candidates["algorithm"] == algorithm]
        if algorithm == "hsmm" and not subset.empty:
            preferred = subset[subset["max_duration_patches"] == 50]
            if not preferred.empty:
                subset = preferred
        if not subset.empty:
            rows.append(subset.iloc[0])
    if not rows:
        return
    figure, axes = plt.subplots(len(rows), 1, figsize=(14, 1.8 * len(rows)), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, row in zip(axes, rows):
        labels, center = _load_labels(row, trial_id)
        limit = min(labels.size, 1200)
        axis.imshow(
            labels[:limit][None, :],
            aspect="auto",
            interpolation="nearest",
            cmap="tab20",
            extent=(center[0], center[limit - 1], 0, 1),
        )
        axis.set_yticks([])
        axis.set_ylabel(str(row["algorithm"]).upper(), rotation=0, labelpad=30)
    axes[-1].set_xlabel("A4_C4 physical time (s); first 1,200 valid patches")
    figure.suptitle("Representative chronological partitions: Graph Hybrid P2, K=6")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _measurement_heatmap(measurements: pd.DataFrame, output: Path) -> None:
    if measurements.empty:
        return
    subset = measurements[
        (measurements["encoder"] == "graph_hybrid_p2")
        & (measurements["algorithm"] == "hmm")
        & (measurements["K"] == 6)
    ]
    if subset.empty:
        return
    first_seed = subset["encoder_seed"].min()
    first_downstream = subset["downstream_seed"].min()
    subset = subset[
        (subset["encoder_seed"] == first_seed)
        & (subset["downstream_seed"] == first_downstream)
    ].sort_values("state")
    columns = [
        value
        for value in subset.columns
        if value.endswith("_pair_mean")
        or value.endswith("_abs_pair_difference")
        or value.endswith("_confidence_weighted_mean")
    ]
    if not columns:
        return
    matrix = subset[columns].to_numpy(dtype=float)
    mean = np.nanmean(matrix, axis=0)
    scale = np.nanstd(matrix, axis=0)
    matrix = (matrix - mean) / np.where(scale > 1e-12, scale, 1.0)
    labels = [
        value.replace("node_feature_", "n").replace("edge_", "e:").replace(
            "_confidence_weighted_mean", ""
        )
        for value in columns
    ]
    figure, axis = plt.subplots(figsize=(max(12, len(columns) * 0.45), 5))
    image = axis.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    axis.set_xticks(range(len(columns)), labels, rotation=90, fontsize=7)
    axis.set_yticks(range(matrix.shape[0]), [f"state {int(value)}" for value in subset["state"]])
    axis.set_title("Original-measurement signatures (z-scored across run-local states)")
    figure.colorbar(image, ax=axis, label="Across-state z score")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _encoder_dashboard(summary: pd.DataFrame, output: Path) -> None:
    required = {"effective_entropy_states_mean", "bouts_per_minute_mean"}
    if summary.empty or not required.issubset(summary.columns):
        return
    figure, axis = plt.subplots(figsize=(10, 7))
    for (encoder, algorithm), group in summary.groupby(["encoder", "algorithm"]):
        axis.scatter(
            group["effective_entropy_states_mean"],
            group["bouts_per_minute_mean"],
            label=f"{encoder}/{algorithm}",
            alpha=0.75,
        )
    axis.set_xlabel("Entropy effective state count")
    axis.set_ylabel("Bouts per minute")
    axis.set_title("Encoder x downstream fragmentation dashboard")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def aggregate_benchmarks(
    *,
    input_dirs: list[str | Path],
    output_dir: str | Path,
    trial_id: str = "A4_C4",
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs = _read_tables(input_dirs, "runs.csv")
    if runs.empty:
        raise FileNotFoundError("no runs.csv files found")
    runs.to_csv(output / "all_runs.csv", index=False)
    completed = runs[runs["status"] == "completed"].copy()
    failures = runs[runs["status"] != "completed"].copy()
    failures.to_csv(output / "failures.csv", index=False)
    occupancy = pd.DataFrame(_occupancy_rows(completed, trial_id))
    occupancy.to_csv(output / "state_occupancy.csv", index=False)
    stability = _partition_stability(completed, trial_id)
    stability.to_csv(output / "partition_stability.csv", index=False)
    agreement, contingencies = _cross_agreement(completed, trial_id)
    agreement.to_csv(output / "cross_model_agreement.csv", index=False)
    contingencies.to_csv(output / "cross_model_contingency.csv", index=False)
    summary = _aggregate_runs(completed, stability)
    summary.to_csv(output / "model_summary.csv", index=False)
    measurements = _read_tables(input_dirs, "state_measurements.csv")
    measurements.to_csv(output / "state_measurements.csv", index=False)
    bouts = _read_bouts(completed)
    bouts.to_csv(output / "all_bouts.csv", index=False)

    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    _occupancy_plot(occupancy, plots / "occupancy_distributions.png")
    _line_plot(summary, "effective_entropy_states_mean", plots / "effective_states_vs_k.png", "Entropy effective state count")
    _line_plot(summary, "mean_pairwise_ari", plots / "stability_ari_vs_k.png", "Mean pairwise ARI")
    _line_plot(summary, "bouts_per_minute_mean", plots / "bouts_per_minute_vs_k.png", "Bouts per minute")
    _line_plot(summary, "validation_fit_per_patch_mean", plots / "validation_fit_vs_k.png", "Validation fit per patch (within representation only)")
    _line_plot(summary, "train_fit_per_patch_mean", plots / "train_fit_vs_k.png", "Training fit per patch (within representation only)")
    _bout_plot(bouts, plots / "bout_duration_distributions.png")
    _agreement_heatmap(agreement, plots / "cross_algorithm_agreement.png")
    _chronological_plot(completed, trial_id, plots / "representative_chronological_states.png")
    _measurement_heatmap(measurements, plots / "state_measurement_signatures.png")
    _encoder_dashboard(summary, plots / "encoder_comparison_dashboard.png")

    manifest = {
        "contract": "social_downstream_aggregate_v1",
        "input_dirs": [str(Path(value).resolve()) for value in input_dirs],
        "output_dir": str(output),
        "agreement_trial": trial_id,
        "total_runs": int(runs.shape[0]),
        "completed_runs": int(completed.shape[0]),
        "failed_runs": int(failures.shape[0]),
        "notes": [
            "ARI/NMI compare aligned partitions and do not imply behavioral accuracy.",
            "State numbers are run-local; occupancy stacks do not align state semantics across runs.",
            "Raw fit values must not be compared across encoder observation spaces.",
            "A4_C4 is reused development/validation data, not an independent test.",
        ],
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trial-id", default="A4_C4")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = aggregate_benchmarks(
        input_dirs=args.input_dir,
        output_dir=args.output_dir,
        trial_id=args.trial_id,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
