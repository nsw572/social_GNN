"""Extend the frozen Graph Full sticky-HMM K sweep through K=30.

This runner deliberately reuses the completed K<=12 artifacts and fits only
the requested new Graph Full models.  It is resumable at the individual model
checkpoint level and keeps the fixed-prior primary sweep separate from the
K-normalized prior sensitivity control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .k_selection import (
    CorrespondenceThresholds,
    DEV_TRIAL,
    ENCODER_SEEDS,
    EXCLUDED_TRIAL,
    FIXED_PRIOR_STRENGTH,
    HMM_SEEDS,
    OccupancyThresholds,
    PCA_DIM,
    TARGET_DWELL_S,
    TIMESTEP_S,
    TRAIN_TRIALS,
    _run_workers,
    add_reference_deltas,
    cross_k_tables,
    prior_policy_stability_tables,
    seedwise_rankings,
    stability_tables,
)


ARCHITECTURE = "graph_full"
REUSED_K_GRID = (4, 5, 6, 7, 8, 10, 12)
EXTENSION_K_GRID = (14, 16, 18, 20, 24, 27, 30)
FULL_K_GRID = REUSED_K_GRID + EXTENSION_K_GRID
PRIOR_CONTROL_K_GRID = (16, 20, 24, 30)
REFERENCE_K = 12
DEFAULT_OUTPUT = (
    "evaluation_runs/gnn_9dim_0p04_downstream_v2/k_selection_k30"
)
DEFAULT_BASELINE = (
    "evaluation_runs/gnn_9dim_0p04_downstream_v2/k_selection_v1"
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _frame(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def _completed(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame.status.eq("completed")].copy()


def _graph_fixed(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[frame.architecture.eq(ARCHITECTURE)].copy()
    if "prior_policy" in result:
        result = result[result.prior_policy.eq("fixed")].copy()
    return result


def _load_baseline_table(baseline: Path, name: str) -> pd.DataFrame:
    path = baseline / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"required completed K<=12 artifact is missing: {path}")
    return _graph_fixed(pd.read_csv(path))


def _save_collection(output: Path, prefix: str, collection: dict[str, Any]) -> None:
    mapping = {
        "runs": "runs",
        "occupancy": "occupancy_by_state",
        "temporal": "temporal_metrics",
        "per_state": "per_state_temporal_metrics",
        "transitions": "transition_matrices",
        "prior": "prior_diagnostics",
        "failures": "failures",
    }
    for key, suffix in mapping.items():
        _frame(collection.get(key, [])).to_csv(
            output / f"{prefix}_{suffix}.csv", index=False
        )


def _load_collection(output: Path, prefix: str) -> dict[str, Any]:
    mapping = {
        "runs": "runs",
        "occupancy": "occupancy_by_state",
        "temporal": "temporal_metrics",
        "per_state": "per_state_temporal_metrics",
        "transitions": "transition_matrices",
        "prior": "prior_diagnostics",
        "failures": "failures",
    }
    result: dict[str, Any] = {}
    for key, suffix in mapping.items():
        path = output / f"{prefix}_{suffix}.csv"
        if not path.exists():
            raise FileNotFoundError(f"cannot aggregate existing extension; missing {path}")
        try:
            result[key] = pd.read_csv(path).to_dict("records")
        except pd.errors.EmptyDataError:
            result[key] = []
    result["reused_runs"] = 0
    result["input_hashes"] = []
    return result


def _restore_initial_fit_cost(runs: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Keep fit-time RSS/runtime when a resumable aggregation reloads checkpoints."""

    if not path.exists():
        return runs
    original = pd.read_csv(path)
    required = {
        "architecture",
        "encoder_seed",
        "hmm_seed",
        "K",
        "elapsed_s",
        "peak_rss_mb",
    }
    if not required.issubset(original.columns):
        return runs
    keys = ["architecture", "encoder_seed", "hmm_seed", "K"]
    cost = original[keys + ["elapsed_s", "peak_rss_mb"]].rename(
        columns={
            "elapsed_s": "initial_fit_elapsed_s",
            "peak_rss_mb": "initial_fit_peak_rss_mb",
        }
    )
    result = runs.merge(cost, on=keys, how="left", validate="one_to_one")
    use = result.K.gt(12) & result.initial_fit_elapsed_s.notna()
    result.loc[use, "elapsed_s"] = result.loc[use, "initial_fit_elapsed_s"]
    result.loc[use, "peak_rss_mb"] = result.loc[use, "initial_fit_peak_rss_mb"]
    return result


def _add_artifact_hashes(runs: pd.DataFrame) -> pd.DataFrame:
    """Attach content hashes to every persisted model and state assignment."""

    result = runs.copy()
    for path_column, hash_column in (
        ("checkpoint", "checkpoint_sha256"),
        ("assignment", "assignment_sha256"),
    ):
        if path_column not in result.columns:
            result[hash_column] = ""
            continue

        def hash_if_present(value: object) -> str:
            if pd.isna(value) or not str(value).strip():
                return ""
            path = Path(str(value))
            return _sha256(path) if path.is_file() else ""

        result[hash_column] = result[path_column].map(hash_if_present)
    return result


def _input_hash_records(*frames: pd.DataFrame) -> list[dict[str, str]]:
    """Return a stable, deduplicated audit table for frozen inputs."""

    columns = [
        "architecture",
        "encoder_seed",
        "embedding_dir",
        "embedding_manifest_sha256",
        "preprocessor",
        "preprocessor_sha256",
    ]
    usable = [frame for frame in frames if not frame.empty]
    if not usable:
        return []
    joined = pd.concat(usable, ignore_index=True, sort=False)
    available = [column for column in columns if column in joined.columns]
    if not available:
        return []
    audit = joined[available].drop_duplicates().sort_values(available).fillna("")
    return [
        {column: str(value) for column, value in record.items()}
        for record in audit.to_dict("records")
    ]


def _assert_unique_runs(frame: pd.DataFrame, policies: bool = False) -> None:
    keys = ["architecture", "encoder_seed", "hmm_seed", "K"]
    if policies:
        keys.append("prior_policy")
    duplicate = frame.duplicated(keys, keep=False)
    if duplicate.any():
        raise ValueError(
            "duplicate run identities: "
            + frame.loc[duplicate, keys].to_dict("records").__repr__()
        )


def _add_k12_deltas(runs: pd.DataFrame) -> pd.DataFrame:
    result = runs.copy()
    baseline = _completed(result[result.K.eq(REFERENCE_K)])[
        [
            "architecture",
            "encoder_seed",
            "hmm_seed",
            "dev_log_likelihood_total",
            "dev_log_likelihood_per_point",
            "aic_train",
            "bic_train",
        ]
    ].rename(
        columns={
            "dev_log_likelihood_total": "k12_dev_log_likelihood_total",
            "dev_log_likelihood_per_point": "k12_dev_log_likelihood_per_point",
            "aic_train": "k12_aic_train",
            "bic_train": "k12_bic_train",
        }
    )
    if baseline.duplicated(["architecture", "encoder_seed", "hmm_seed"]).any():
        raise ValueError("multiple fixed-prior K=12 references")
    result = result.merge(
        baseline,
        on=["architecture", "encoder_seed", "hmm_seed"],
        how="left",
        validate="many_to_one",
    )
    result["dev_delta_log_likelihood_vs_k12"] = (
        result.dev_log_likelihood_total - result.k12_dev_log_likelihood_total
    )
    result["dev_delta_log_likelihood_per_point_vs_k12"] = (
        result.dev_log_likelihood_per_point
        - result.k12_dev_log_likelihood_per_point
    )
    result["delta_aic_vs_k12"] = result.aic_train - result.k12_aic_train
    result["delta_bic_vs_k12"] = result.bic_train - result.k12_bic_train
    return result


def _drop_stale_reference_columns(runs: pd.DataFrame) -> pd.DataFrame:
    """Remove derived columns copied from the reused K<=12 summary."""

    prefixes = (
        "reference_",
        "dev_delta_log_likelihood_vs_k6",
        "dev_delta_log_likelihood_per_point_vs_k6",
        "delta_aic_vs_k6",
        "delta_bic_vs_k6",
        "k12_",
        "dev_delta_log_likelihood_vs_k12",
        "dev_delta_log_likelihood_per_point_vs_k12",
        "delta_aic_vs_k12",
        "delta_bic_vs_k12",
    )
    stale = [
        column
        for column in runs.columns
        if any(column.startswith(prefix) for prefix in prefixes)
    ]
    return runs.drop(columns=stale, errors="ignore")


def _marginal_gains(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, subset in _completed(runs).groupby(
        ["architecture", "encoder_seed", "hmm_seed"], sort=True
    ):
        ordered = subset.sort_values("K")
        previous = None
        for _, row in ordered.iterrows():
            record = row.to_dict()
            if previous is None:
                record.update(
                    {
                        "previous_K": np.nan,
                        "added_states": np.nan,
                        "added_parameters": np.nan,
                        "marginal_dev_log_likelihood_total": np.nan,
                        "marginal_dev_log_likelihood_per_point": np.nan,
                        "marginal_dev_ll_per_added_state": np.nan,
                        "marginal_dev_ll_per_added_parameter": np.nan,
                    }
                )
            else:
                added_k = int(row.K - previous.K)
                added_p = int(row.parameter_count - previous.parameter_count)
                gain = float(
                    row.dev_log_likelihood_total
                    - previous.dev_log_likelihood_total
                )
                record.update(
                    {
                        "previous_K": int(previous.K),
                        "added_states": added_k,
                        "added_parameters": added_p,
                        "marginal_dev_log_likelihood_total": gain,
                        "marginal_dev_log_likelihood_per_point": float(
                            row.dev_log_likelihood_per_point
                            - previous.dev_log_likelihood_per_point
                        ),
                        "marginal_dev_ll_per_added_state": gain / added_k,
                        "marginal_dev_ll_per_added_parameter": gain / added_p,
                    }
                )
            rows.append(record)
            previous = row
    return pd.DataFrame(rows)


def _mean_sd(frame: pd.DataFrame, group: list[str], columns: list[str]) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {}
    for column in columns:
        aggregations[f"{column}_mean"] = (column, "mean")
        aggregations[f"{column}_sd"] = (column, "std")
    return frame.groupby(group, dropna=False).agg(**aggregations).reset_index()


def _full_range_summary(marginal: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dev_delta_log_likelihood_per_point_vs_k12",
        "delta_aic_vs_k12",
        "delta_bic_vs_k12",
        "marginal_dev_log_likelihood_per_point",
        "marginal_dev_ll_per_added_state",
        "marginal_dev_ll_per_added_parameter",
        "dev_effective_states",
        "dev_effective_state_ratio",
        "dev_largest_state_occupancy",
        "dev_smallest_nonempty_state_occupancy",
        "dev_bouts_per_minute",
        "dev_median_bout_duration_s",
        "dev_fraction_one_step_bouts",
        "dev_fraction_bouts_shorter_0p20_s",
        "learned_self_transition_mean",
    ]
    existing = [value for value in columns if value in marginal]
    result = _mean_sd(_completed(marginal), ["architecture", "K"], existing)
    counts = (
        marginal.groupby(["architecture", "K"])
        .agg(
            requested_runs=("status", "size"),
            completed_runs=("status", lambda value: int(value.eq("completed").sum())),
            converged_runs=("converged", lambda value: int(value.fillna(False).sum())),
        )
        .reset_index()
    )
    return result.merge(counts, on=["architecture", "K"], how="left")


def _classify_refinement(
    cross_k: pd.DataFrame, cross_summary: pd.DataFrame
) -> pd.DataFrame:
    selected = cross_k[
        cross_k.left_K.eq(REFERENCE_K) & cross_k.right_K.gt(REFERENCE_K)
    ].copy()
    if selected.empty:
        return selected
    aggregate = (
        selected.groupby(["architecture", "right_K"])
        .agg(
            comparisons=("pair_id", "size"),
            mean_left_to_right_purity=("left_to_right_purity", "mean"),
            minimum_left_to_right_purity=("left_to_right_purity", "min"),
            mean_right_to_left_coverage=("right_to_left_coverage", "mean"),
            mean_variation_of_information=("variation_of_information", "mean"),
            mean_conditional_entropy_new_given_k12=(
                "conditional_entropy_right_given_left",
                "mean",
            ),
            mean_hungarian_matched_patch_fraction=(
                "hungarian_matched_patch_fraction",
                "mean",
            ),
        )
        .reset_index()
        .rename(columns={"right_K": "K"})
    )
    matched = cross_summary[
        cross_summary.hmm_seed_relation.eq("matched_hmm_seed")
    ][["architecture", "K", "ARI_mean", "ARI_min"]].rename(
        columns={"ARI_mean": "cross_encoder_ari_mean", "ARI_min": "cross_encoder_ari_min"}
    )
    aggregate = aggregate.merge(matched, on=["architecture", "K"], how="left")

    def classify(row: pd.Series) -> str:
        if (
            row.mean_left_to_right_purity >= 0.80
            and row.minimum_left_to_right_purity >= 0.65
            and row.cross_encoder_ari_mean >= 0.50
        ):
            return "reproducible_refinement"
        if (
            row.mean_left_to_right_purity < 0.60
            or row.cross_encoder_ari_mean < 0.30
        ):
            return "unstable_refinement"
        return "ambiguous_refinement"

    aggregate["classification"] = aggregate.apply(classify, axis=1)
    aggregate["classification_rule"] = (
        "reproducible if mean purity>=0.80, minimum purity>=0.65, and matched-seed "
        "cross-encoder ARI>=0.50; unstable if mean purity<0.60 or ARI<0.30; otherwise ambiguous"
    )
    return aggregate


def _prior_summary(prior: pd.DataFrame) -> pd.DataFrame:
    return (
        prior.groupby(["architecture", "K", "prior_policy"])
        .agg(
            state_rows=("source_state", "size"),
            prior_row_mass_mean=("prior_row_mass", "mean"),
            prior_to_xi_ratio_mean=("prior_to_xi_ratio", "mean"),
            prior_to_xi_ratio_median=("prior_to_xi_ratio", "median"),
            prior_to_xi_ratio_max=("prior_to_xi_ratio", "max"),
            learned_self_transition_mean=("learned_self_transition", "mean"),
            learned_self_transition_sd=("learned_self_transition", "std"),
        )
        .reset_index()
    )


def _policy_summary(policy_runs: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    stability = prior_policy_stability_tables(policy_runs)
    summary = (
        _completed(policy_runs)
        .groupby(["architecture", "K", "prior_policy"])
        .agg(
            completed_runs=("status", "size"),
            dev_log_likelihood_per_point_mean=("dev_log_likelihood_per_point", "mean"),
            effective_state_ratio_mean=("dev_effective_state_ratio", "mean"),
            occupied_states_mean=("dev_occupied_states", "mean"),
            bouts_per_minute_mean=("dev_bouts_per_minute", "mean"),
            median_bout_duration_s_mean=("dev_median_bout_duration_s", "mean"),
            learned_self_transition_mean=("learned_self_transition_mean", "mean"),
        )
        .reset_index()
    )
    return summary, stability


def _likelihood_traces(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, run in _completed(runs).iterrows():
        path = Path(run.checkpoint)
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as data:
            history = np.asarray(data["history"], dtype=float)
        for iteration, value in enumerate(history, start=1):
            rows.append(
                {
                    "architecture": run.architecture,
                    "encoder_seed": int(run.encoder_seed),
                    "hmm_seed": int(run.hmm_seed),
                    "K": int(run.K),
                    "prior_policy": run.prior_policy,
                    "iteration": iteration,
                    "train_marginal_log_likelihood_before_m_step": float(value),
                }
            )
    return pd.DataFrame(rows)


def _make_figures(
    output: Path,
    marginal: pd.DataFrame,
    full_summary: pd.DataFrame,
    within: pd.DataFrame,
    cross: pd.DataFrame,
    prior: pd.DataFrame,
    policy: pd.DataFrame,
) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    complete = _completed(marginal)

    def seed_lines(column: str, ylabel: str, name: str) -> None:
        fig, axis = plt.subplots(figsize=(8.5, 5.0))
        for (encoder_seed, hmm_seed), group in complete.groupby(
            ["encoder_seed", "hmm_seed"]
        ):
            group = group.sort_values("K")
            axis.plot(
                group.K,
                group[column],
                alpha=0.55,
                marker="o",
                label=f"enc{encoder_seed}/hmm{hmm_seed}",
            )
        axis.set_xlabel("K")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=6, ncol=3)
        fig.tight_layout()
        fig.savefig(figures / name, dpi=180)
        plt.close(fig)

    seed_lines("aic_train", "train AIC", "aic_full_range.png")
    seed_lines("bic_train", "train BIC", "bic_full_range.png")
    seed_lines(
        "dev_delta_log_likelihood_per_point_vs_k12",
        "development delta LL/point vs K=12",
        "dev_ll_delta_vs_k12.png",
    )
    seed_lines(
        "marginal_dev_ll_per_added_parameter",
        "development LL gain per added parameter",
        "marginal_gain_per_parameter.png",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    summary = full_summary.sort_values("K")
    axes[0].plot(
        summary.K,
        summary.dev_effective_state_ratio_mean,
        marker="o",
        label="K_eff/K",
    )
    axes[0].plot(
        summary.K,
        summary.dev_largest_state_occupancy_mean,
        marker="o",
        label="largest occupancy",
    )
    axes[0].set_ylabel("fraction")
    axes[0].legend()
    axes[1].plot(
        summary.K,
        summary.dev_bouts_per_minute_mean,
        marker="o",
        label="bouts/min",
    )
    axes[1].set_ylabel("bouts/min")
    for axis in axes:
        axis.set_xlabel("K")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "state_use_and_fragmentation.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.5, 5.0))
    axis.plot(within.K, within.ARI_mean, marker="o", label="within encoder")
    matched = cross[cross.hmm_seed_relation.eq("matched_hmm_seed")].sort_values("K")
    axis.plot(matched.K, matched.ARI_mean, marker="o", label="cross encoder")
    axis.set_xlabel("K")
    axis.set_ylabel("mean ARI")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "partition_stability.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fixed = prior[prior.prior_policy.eq("fixed")].sort_values("K")
    axes[0].plot(fixed.K, fixed.prior_to_xi_ratio_mean, marker="o", label="mean")
    axes[0].plot(fixed.K, fixed.prior_to_xi_ratio_max, marker="o", label="maximum")
    axes[0].set_ylabel("prior/data xi ratio")
    axes[0].legend()
    for name, group in policy.groupby("prior_policy"):
        axes[1].plot(
            group.K,
            group.dev_log_likelihood_per_point_mean,
            marker="o",
            label=name,
        )
    axes[1].set_ylabel("development LL/point")
    axes[1].legend()
    for axis in axes:
        axis.set_xlabel("K")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "prior_leverage_and_policy.png", dpi=180)
    plt.close(fig)


def _modal_winner(rankings: pd.DataFrame, criterion: str) -> int:
    winners = rankings[
        rankings.criterion.eq(criterion) & rankings.seedwise_winner
    ].K.astype(int)
    counts = winners.value_counts()
    return int(min(counts[counts.eq(counts.max())].index))


def _report(
    output: Path,
    full_summary: pd.DataFrame,
    rankings: pd.DataFrame,
    within: pd.DataFrame,
    cross: pd.DataFrame,
    refinement: pd.DataFrame,
    prior: pd.DataFrame,
    policy_alignment: pd.DataFrame,
    runs: pd.DataFrame,
) -> Path:
    completed = _completed(runs)
    aic_k = _modal_winner(rankings, "aic_train")
    bic_k = _modal_winner(rankings, "bic_train")
    aic_boundary = aic_k == max(FULL_K_GRID)
    bic_boundary = bic_k == max(FULL_K_GRID)
    k30 = full_summary[full_summary.K.eq(30)].iloc[0]
    k12 = full_summary[full_summary.K.eq(12)].iloc[0]
    matched = cross[cross.hmm_seed_relation.eq("matched_hmm_seed")].sort_values("K")
    k30_cross = matched[matched.K.eq(30)]
    k12_cross = matched[matched.K.eq(12)]
    ordered = full_summary.sort_values("K")
    last_gains = ordered[ordered.K.ge(20)][
        "marginal_dev_log_likelihood_per_point_mean"
    ]
    diminishing = bool(
        len(last_gains) >= 2
        and np.isfinite(last_gains).all()
        and abs(last_gains.iloc[-1]) < abs(last_gains.iloc[0])
    )
    fixed_prior = prior[prior.prior_policy.eq("fixed")]
    leverage30 = fixed_prior[fixed_prior.K.eq(30)]
    alignment30 = policy_alignment[policy_alignment.K.eq(30)]
    lines = [
        "# 0.04 s Social-GNN: Extended Sticky HMM State-Number Selection Through K=30",
        "",
        "## Executive summary",
        "",
        f"All {len(completed)}/{len(FULL_K_GRID) * len(ENCODER_SEEDS) * len(HMM_SEEDS)} fixed-prior Graph Full runs completed. "
        f"The modal seedwise AIC winner is K={aic_k} and the modal seedwise BIC winner is K={bic_k}.",
        (
            "At least one information criterion still selects the tested upper boundary, so the statistical optimum has not been bracketed."
            if aic_boundary or bic_boundary
            else "Both modal information-criterion winners are interior to the tested grid."
        ),
        f"K=30 mean effective-state ratio is {k30.dev_effective_state_ratio_mean:.3f}; "
        f"the mean occupied-state count is {completed[completed.K.eq(30)].dev_occupied_states.mean():.2f}/30.",
        "",
        "## Experimental contract",
        "",
        "- Graph Full only; frozen 64D embeddings and the existing train-fitted PCA16 transform.",
        "- Train: A1_C1, A3_C3, A4_B4; development: A4_C4; A5_C5 excluded.",
        "- 0.04 s clock, diagonal Gaussian emissions, 2.0 s target dwell, self-transition target 0.98.",
        "- Primary prior strength 50,000; no lower-K warm start; every invalid gap and trial boundary remains a sequence boundary.",
        "- K<=12 rows and checkpoints are reused from `k_selection_v1`; K=14..30 are new independent fits.",
        "",
        "## Statistical selection and diminishing returns",
        "",
        f"- AIC interior minimum: {'no' if aic_boundary else 'yes'}, modal K={aic_k}.",
        f"- BIC interior minimum: {'no' if bic_boundary else 'yes'}, modal K={bic_k}.",
        f"- Late-range development-likelihood gains show a simple decreasing-gain pattern: {'yes' if diminishing else 'no or seed-dependent'}.",
        f"- K=12 mean delta-BIC versus itself is 0 by definition; K=30 mean delta-BIC is {k30.delta_bic_vs_k12_mean:.2f}.",
        f"- K=30 development LL/point gain versus K=12 is {k30.dev_delta_log_likelihood_per_point_vs_k12_mean:.5f}.",
        "",
        "Raw likelihoods are ranked only within the same encoder/PCA seed. Cross-seed summaries use within-space deltas rather than treating absolute Gaussian likelihoods as commensurate.",
        "",
        "## Utilization and temporal structure",
        "",
        f"K=12 had mean K_eff/K={k12.dev_effective_state_ratio_mean:.3f}; K=30 has {k30.dev_effective_state_ratio_mean:.3f}. "
        f"Bouts/min changed from {k12.dev_bouts_per_minute_mean:.2f} to {k30.dev_bouts_per_minute_mean:.2f}.",
        f"At K=30, the mean largest occupancy is {k30.dev_largest_state_occupancy_mean:.3f} and the mean median bout duration is {k30.dev_median_bout_duration_s_mean:.3f} s.",
        "",
        "## Reproducibility",
        "",
        (
            f"Matched-seed cross-encoder mean ARI changes from {float(k12_cross.iloc[0].ARI_mean):.3f} at K=12 "
            f"to {float(k30_cross.iloc[0].ARI_mean):.3f} at K=30."
            if not k12_cross.empty and not k30_cross.empty
            else "Cross-encoder K=12/K=30 comparison is incomplete."
        ),
        f"Within-encoder K=30 mean ARI is {float(within[within.K.eq(30)].iloc[0].ARI_mean):.3f}.",
        "Pairwise stability values are dependent diagnostics on the same development trial, not biological replicates.",
        "",
        "## K=12 to higher-K refinement",
        "",
        "Classification uses an explicit conservative rule recorded in `cross_k_refinement.csv`; it does not assume nesting or assign behavioral semantics.",
        "",
    ]
    if refinement.empty:
        lines.append("Refinement analysis is unavailable because aligned assignments were incomplete.")
    else:
        lines.extend(
            [
                "| K | classification | mean K12 purity | mean VI | cross-encoder ARI |",
                "| ---: | --- | ---: | ---: | ---: |",
            ]
        )
        for _, row in refinement.sort_values("K").iterrows():
            lines.append(
                f"| {int(row.K)} | {row.classification} | {row.mean_left_to_right_purity:.3f} | "
                f"{row.mean_variation_of_information:.3f} | {row.cross_encoder_ari_mean:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Sticky-prior leverage",
            "",
            (
                f"At K=30, fixed-prior mean and maximum prior/data xi ratios are "
                f"{float(leverage30.iloc[0].prior_to_xi_ratio_mean):.3f} and "
                f"{float(leverage30.iloc[0].prior_to_xi_ratio_max):.3f}."
                if not leverage30.empty
                else "K=30 prior-leverage diagnostics are unavailable."
            ),
            (
                f"Matched fixed-versus-normalized K=30 partitions have mean ARI "
                f"{alignment30.ARI.mean():.3f} (minimum {alignment30.ARI.min():.3f})."
                if not alignment30.empty
                else "The K=30 normalized-prior control is incomplete."
            ),
            "The 50,000×6/K control is a heuristic sensitivity analysis, not an equivalent prior or a post-hoc replacement for the primary sweep.",
            "",
            "## Recommendation",
            "",
        ]
    )
    if aic_boundary or bic_boundary:
        lines.append(
            "The evidence does not justify declaring K=30 the true number of behaviors: at least one statistical curve remains boundary-seeking. K=12 remains a conservative candidate only if its reproducibility, temporal granularity, and video interpretability are preferable to the incremental likelihood gain. A reasonable K range should be chosen from the stable, occupied, non-fragmented region after blinded video review, not from AIC/BIC alone."
        )
    else:
        lines.append(
            f"The statistical curves bracket an interior region near K={min(aic_k, bic_k)}-{max(aic_k, bic_k)}, but selection still requires stability, temporal fragmentation, and blinded video review. Statistical Gaussian components are not automatically genuine behaviors."
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- A4_C4 is reused development data, not an untouched animal-pair test.",
            "- Strong MAP transition regularization makes AIC/BIC approximate diagnostics.",
            "- State correspondence is overlap-based and does not establish semantic identity.",
            "- High occupancy or stability is not evidence of biological correctness.",
        ]
    )
    path = output / "k30_extension_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_k30_extension(
    *,
    root: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT,
    baseline_dir: str | Path = DEFAULT_BASELINE,
    workers: int = 1,
    max_iter: int = 40,
    tolerance: float = 1e-4,
    covariance_floor: float = 1e-3,
    aggregate_existing: bool = False,
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    output = (root_path / output_dir).resolve() if not Path(output_dir).is_absolute() else Path(output_dir).resolve()
    baseline = (root_path / baseline_dir).resolve() if not Path(baseline_dir).is_absolute() else Path(baseline_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    if aggregate_existing:
        fixed = _load_collection(output, "extension_fixed")
        normalized = _load_collection(output, "extension_normalized")
    else:
        fixed = _run_workers(
            root=root_path,
            output=output,
            policy="fixed",
            k_by_architecture={ARCHITECTURE: EXTENSION_K_GRID},
            workers=workers,
            max_iter=max_iter,
            tolerance=tolerance,
            covariance_floor=covariance_floor,
            architectures=(ARCHITECTURE,),
            encoder_seeds=ENCODER_SEEDS,
            hmm_seeds=HMM_SEEDS,
        )
        _save_collection(output, "extension_fixed", fixed)

        normalized = _run_workers(
            root=root_path,
            output=output / "prior_policy",
            policy="normalized_6_over_k",
            k_by_architecture={ARCHITECTURE: PRIOR_CONTROL_K_GRID},
            workers=workers,
            max_iter=max_iter,
            tolerance=tolerance,
            covariance_floor=covariance_floor,
            architectures=(ARCHITECTURE,),
            encoder_seeds=ENCODER_SEEDS,
            hmm_seeds=HMM_SEEDS,
        )
        _save_collection(output, "extension_normalized", normalized)

    old_runs = _load_baseline_table(baseline, "runs")
    runs = pd.concat([old_runs, _frame(fixed["runs"])], ignore_index=True, sort=False)
    _assert_unique_runs(runs)
    runs = _restore_initial_fit_cost(
        runs, output / "extension_fixed_initial_fit_metrics.csv"
    )
    runs = _add_artifact_hashes(runs)
    runs = _drop_stale_reference_columns(runs)
    runs = add_reference_deltas(runs)
    runs = _add_k12_deltas(runs)
    marginal = _marginal_gains(runs)
    full_summary = _full_range_summary(marginal)

    table_specs = {
        "occupancy_by_state": "occupancy",
        "temporal_metrics": "temporal",
        "per_state_temporal_metrics": "per_state",
        "transition_matrices": "transitions",
        "prior_diagnostics": "prior",
    }
    combined: dict[str, pd.DataFrame] = {}
    for old_name, new_key in table_specs.items():
        combined[old_name] = pd.concat(
            [_load_baseline_table(baseline, old_name), _frame(fixed[new_key])],
            ignore_index=True,
            sort=False,
        )

    stability = stability_tables(runs)
    adjacent = tuple(zip((REFERENCE_K,) + EXTENSION_K_GRID[:-1], EXTENSION_K_GRID))
    cross_k = cross_k_tables(
        runs,
        combined["per_state_temporal_metrics"],
        output,
        k_grid=(REFERENCE_K,) + EXTENSION_K_GRID,
        reference_k=REFERENCE_K,
        adjacent_pairs=adjacent,
    )
    refinement = _classify_refinement(
        cross_k["cross_k_pairs"], stability["cross_summary"]
    )
    rankings = seedwise_rankings(runs)

    normalized_runs = _add_artifact_hashes(_frame(normalized["runs"]))
    fixed_control = runs[runs.K.isin(PRIOR_CONTROL_K_GRID)].copy()
    policy_runs = pd.concat(
        [fixed_control, normalized_runs], ignore_index=True, sort=False
    )
    _assert_unique_runs(policy_runs, policies=True)
    policy_summary, policy_stability = _policy_summary(policy_runs)
    prior_all = pd.concat(
        [combined["prior_diagnostics"], _frame(normalized["prior"])],
        ignore_index=True,
        sort=False,
    )
    prior_summary = _prior_summary(prior_all)
    traces = _likelihood_traces(
        pd.concat([runs, normalized_runs], ignore_index=True, sort=False)
    )

    occupancy_by_k = _mean_sd(
        _completed(runs),
        ["architecture", "K"],
        [
            "dev_occupied_states",
            "dev_effective_states",
            "dev_effective_state_ratio",
            "dev_largest_state_occupancy",
            "dev_smallest_nonempty_state_occupancy",
            "dev_states_below_one_percent",
        ],
    )
    temporal_by_k = _mean_sd(
        _completed(runs),
        ["architecture", "K"],
        [
            "dev_bouts_per_minute",
            "dev_mean_bout_duration_s",
            "dev_median_bout_duration_s",
            "dev_fraction_one_step_bouts",
            "dev_fraction_bouts_shorter_0p20_s",
            "learned_self_transition_mean",
            "learned_self_transition_min",
            "learned_self_transition_max",
        ],
    )
    convergence = runs[
        [
            column
            for column in (
                "architecture",
                "encoder_seed",
                "hmm_seed",
                "K",
                "status",
                "converged",
                "iterations",
                "final_em_history_log_likelihood",
                "elapsed_s",
                "peak_rss_mb",
                "reused_existing_artifact",
                "failure",
            )
            if column in runs
        ]
    ].copy()
    convergence["artifact_source"] = np.where(
        convergence.K.le(12), "reused_k_selection_v1", "k30_extension"
    )

    likelihood_columns = [
        "architecture",
        "encoder_seed",
        "hmm_seed",
        "K",
        "train_valid_points",
        "dev_valid_points",
        "train_log_likelihood_total",
        "train_log_likelihood_per_point",
        "dev_log_likelihood_total",
        "dev_log_likelihood_per_point",
        "dev_delta_log_likelihood_per_point_vs_k6",
        "dev_delta_log_likelihood_per_point_vs_k12",
        "parameter_count",
        "aic_train",
        "bic_train",
        "delta_aic_vs_k12",
        "delta_bic_vs_k12",
        "previous_K",
        "added_states",
        "added_parameters",
        "marginal_dev_log_likelihood_per_point",
        "marginal_dev_ll_per_added_state",
        "marginal_dev_ll_per_added_parameter",
    ]

    runs.to_csv(output / "run_level_metrics.csv", index=False)
    policy_runs.to_csv(output / "prior_policy_run_level_metrics.csv", index=False)
    full_summary.to_csv(output / "k_selection_full_range.csv", index=False)
    marginal[[value for value in likelihood_columns if value in marginal]].to_csv(
        output / "likelihood_aic_bic.csv", index=False
    )
    rankings.to_csv(output / "seedwise_k_rankings.csv", index=False)
    occupancy_by_k.to_csv(output / "occupancy_by_k.csv", index=False)
    combined["occupancy_by_state"].to_csv(output / "occupancy_by_state.csv", index=False)
    temporal_by_k.to_csv(output / "temporal_metrics_by_k.csv", index=False)
    combined["per_state_temporal_metrics"].to_csv(
        output / "state_duration_distributions.csv", index=False
    )
    combined["transition_matrices"].to_csv(
        output / "transition_matrices.csv", index=False
    )
    stability["within_summary"].to_csv(
        output / "within_encoder_ari_by_k.csv", index=False
    )
    stability["cross_summary"].to_csv(
        output / "cross_encoder_ari_by_k.csv", index=False
    )
    stability["within_pairwise"].to_csv(
        output / "within_encoder_pairwise.csv", index=False
    )
    stability["cross_pairwise"].to_csv(
        output / "cross_encoder_pairwise.csv", index=False
    )
    refinement.to_csv(output / "cross_k_refinement.csv", index=False)
    for name, frame in cross_k.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    prior_summary.to_csv(output / "prior_leverage_by_k.csv", index=False)
    policy_summary.to_csv(output / "prior_policy_sensitivity.csv", index=False)
    policy_stability["fixed_vs_normalized_alignment"].to_csv(
        output / "fixed_vs_normalized_alignment.csv", index=False
    )
    policy_stability["prior_policy_within_summary"].to_csv(
        output / "prior_policy_within_stability.csv", index=False
    )
    policy_stability["prior_policy_cross_summary"].to_csv(
        output / "prior_policy_cross_stability.csv", index=False
    )
    convergence.to_csv(output / "convergence_and_cost.csv", index=False)
    traces.to_csv(output / "likelihood_traces.csv", index=False)
    _frame(fixed["failures"]).to_csv(output / "fixed_failures.csv", index=False)
    _frame(normalized["failures"]).to_csv(
        output / "normalized_prior_failures.csv", index=False
    )

    _make_figures(
        output,
        marginal,
        full_summary,
        stability["within_summary"],
        stability["cross_summary"],
        prior_summary,
        policy_summary,
    )
    report = _report(
        output,
        full_summary,
        rankings,
        stability["within_summary"],
        stability["cross_summary"],
        refinement,
        prior_summary,
        policy_stability["fixed_vs_normalized_alignment"],
        runs,
    )
    config_path = root_path / "configs" / "social_gnn_hmm_k_selection_k30_0p04_v1.yaml"
    base_manifest = baseline / "manifest.json"
    manifest = {
        "contract": "formal_graph_full_hmm_k_selection_k30_0p04_v1",
        "title": "0.04 s Social-GNN: Extended Sticky HMM State-Number Selection Through K=30",
        "root": str(root_path),
        "output_dir": str(output),
        "architecture": ARCHITECTURE,
        "encoder_seeds": list(ENCODER_SEEDS),
        "hmm_seeds": list(HMM_SEEDS),
        "reused_k_grid": list(REUSED_K_GRID),
        "extension_k_grid": list(EXTENSION_K_GRID),
        "full_k_grid": list(FULL_K_GRID),
        "prior_control_k_grid": list(PRIOR_CONTROL_K_GRID),
        "train_trials": list(TRAIN_TRIALS),
        "development_trial": DEV_TRIAL,
        "excluded_trial": EXCLUDED_TRIAL,
        "timestep_s": TIMESTEP_S,
        "pca_dim": PCA_DIM,
        "target_dwell_s": TARGET_DWELL_S,
        "target_self_transition": 1.0 - TIMESTEP_S / TARGET_DWELL_S,
        "fixed_prior_strength": FIXED_PRIOR_STRENGTH,
        "normalized_prior_formula": "50000 * 6 / K",
        "warm_start": False,
        "aggregation_only": aggregate_existing,
        "primary_expected_new_runs": len(EXTENSION_K_GRID) * len(ENCODER_SEEDS) * len(HMM_SEEDS),
        "primary_completed_new_runs": int(_frame(fixed["runs"]).status.eq("completed").sum()),
        "secondary_expected_runs": len(PRIOR_CONTROL_K_GRID) * len(ENCODER_SEEDS) * len(HMM_SEEDS),
        "secondary_completed_runs": int(normalized_runs.status.eq("completed").sum()),
        "fixed_failures": len(fixed["failures"]),
        "normalized_failures": len(normalized["failures"]),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "reused_manifest": str(base_manifest),
        "reused_manifest_sha256": _sha256(base_manifest),
        "input_hashes": _input_hash_records(runs, normalized_runs),
        "checkpoint_hash_table": str(output / "run_level_metrics.csv"),
        "occupancy_thresholds": asdict(OccupancyThresholds()),
        "correspondence_thresholds": asdict(CorrespondenceThresholds()),
        "report": str(report),
        "elapsed_s": time.perf_counter() - started,
        "limitations": [
            "A4_C4 is reused development data, not an untouched biological test.",
            "Information criteria are approximate under strong MAP transition regularization.",
            "Statistical Gaussian components are not automatically genuine behaviors.",
        ],
    }
    _write_json(output / "experiment_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path.cwd()))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-dir", default=DEFAULT_BASELINE)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--covariance-floor", type=float, default=1e-3)
    parser.add_argument(
        "--aggregate-existing",
        action="store_true",
        help="Build final tables/report from completed extension CSVs without model scoring",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_k30_extension(
        root=args.root,
        output_dir=args.output_dir,
        baseline_dir=args.baseline_dir,
        workers=args.workers,
        max_iter=args.max_iter,
        tolerance=args.tolerance,
        covariance_floor=args.covariance_floor,
        aggregate_existing=args.aggregate_existing,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
