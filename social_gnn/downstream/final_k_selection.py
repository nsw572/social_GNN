"""Focused Graph Full Sticky-HMM selection among K=12, 13, and 14.

The runner reuses compatible K=12/K=14 artifacts, fits only missing cells,
keeps fixed and K-normalized sticky-prior policies separate, and applies a
declared reproducibility-first decision rule.  A4_C4 remains development data;
A5_C5 is never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .k_selection import (
    ENCODER_SEEDS,
    HMM_SEEDS,
    _run_workers,
    cross_k_tables,
    prior_policy_stability_tables,
    stability_tables,
)


ARCHITECTURE = "graph_full"
K_VALUES = (12, 13, 14)
REFERENCE_K = 12
DEFAULT_OUTPUT = (
    "evaluation_runs/gnn_9dim_0p04_downstream_v2/"
    "k12_k13_k14_final_selection"
)
BASELINE_K12 = "evaluation_runs/gnn_9dim_0p04_downstream_v2/k_selection_v1"
BASELINE_K30 = "evaluation_runs/gnn_9dim_0p04_downstream_v2/k_selection_k30"
CONFIG = "configs/social_gnn_hmm_k12_k14_final_selection_v1.yaml"
EXPECTED_PARAMETER_COUNT = {12: 527, 13: 584, 14: 643}


def parameter_count(k: int, dimension: int = 16) -> int:
    """Free parameters for K-state diagonal Gaussian HMM with D features."""

    return int(k * k + 2 * dimension * k - 1)


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


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.to_dict("records") if not frame.empty else []


def _localize_artifact_paths(frame: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Map historical Windows workspace paths when aggregating under WSL."""

    result = frame.copy()
    windows_prefix = "c:/users/dell/desktop/deepof/"
    for column in ("checkpoint", "assignment", "embedding_dir", "preprocessor"):
        if column not in result:
            continue

        def localize(value: object) -> object:
            if pd.isna(value):
                return value
            text = str(value).replace("\\", "/")
            if text.lower().startswith(windows_prefix):
                return str(root / text[len(windows_prefix) :])
            return text

        result[column] = result[column].map(localize)
    return result


def _save_collection(output: Path, prefix: str, collection: dict[str, Any]) -> None:
    names = {
        "runs": "runs",
        "occupancy": "occupancy_by_state",
        "temporal": "temporal_metrics",
        "per_state": "per_state_temporal_metrics",
        "transitions": "transition_matrices",
        "prior": "prior_diagnostics",
        "failures": "failures",
    }
    for key, suffix in names.items():
        pd.DataFrame(collection.get(key, [])).to_csv(
            output / f"{prefix}_{suffix}.csv", index=False
        )


def _load_collection(output: Path, prefix: str) -> dict[str, Any]:
    names = {
        "runs": "runs",
        "occupancy": "occupancy_by_state",
        "temporal": "temporal_metrics",
        "per_state": "per_state_temporal_metrics",
        "transitions": "transition_matrices",
        "prior": "prior_diagnostics",
        "failures": "failures",
    }
    return {
        key: _rows(_read(output / f"{prefix}_{suffix}.csv"))
        for key, suffix in names.items()
    }


def _filter(
    frame: pd.DataFrame,
    *,
    k: int | Sequence[int],
    policy: str | None = None,
) -> pd.DataFrame:
    values = {int(k)} if isinstance(k, (int, np.integer)) else {int(x) for x in k}
    selected = frame[
        frame.architecture.eq(ARCHITECTURE) & frame.K.astype(int).isin(values)
    ].copy()
    if policy is not None and "prior_policy" in selected:
        selected = selected[selected.prior_policy.eq(policy)].copy()
    return selected


def _historical_fixed(
    k12: Path, k30: Path, root: Path
) -> dict[str, pd.DataFrame]:
    result = {
        "runs": _filter(_read(k30 / "run_level_metrics.csv"), k=(12, 14), policy="fixed"),
        "occupancy": pd.concat(
            [
                _filter(_read(k12 / "occupancy_by_state.csv"), k=12, policy="fixed"),
                _filter(_read(k30 / "extension_fixed_occupancy_by_state.csv"), k=14, policy="fixed"),
            ],
            ignore_index=True,
            sort=False,
        ),
        "temporal": pd.concat(
            [
                _filter(_read(k12 / "temporal_metrics.csv"), k=12, policy="fixed"),
                _filter(_read(k30 / "extension_fixed_temporal_metrics.csv"), k=14, policy="fixed"),
            ],
            ignore_index=True,
            sort=False,
        ),
        "per_state": pd.concat(
            [
                _filter(_read(k12 / "per_state_temporal_metrics.csv"), k=12, policy="fixed"),
                _filter(_read(k30 / "extension_fixed_per_state_temporal_metrics.csv"), k=14, policy="fixed"),
            ],
            ignore_index=True,
            sort=False,
        ),
        "transitions": pd.concat(
            [
                _filter(_read(k12 / "transition_matrices.csv"), k=12, policy="fixed"),
                _filter(_read(k30 / "extension_fixed_transition_matrices.csv"), k=14, policy="fixed"),
            ],
            ignore_index=True,
            sort=False,
        ),
        "prior": pd.concat(
            [
                _filter(_read(k12 / "prior_diagnostics.csv"), k=12, policy="fixed"),
                _filter(_read(k30 / "extension_fixed_prior_diagnostics.csv"), k=14, policy="fixed"),
            ],
            ignore_index=True,
            sort=False,
        ),
    }
    result["runs"] = _localize_artifact_paths(result["runs"], root)
    return result


def _historical_normalized_k12(k12: Path, root: Path) -> dict[str, pd.DataFrame]:
    result = {
        "runs": _filter(
            _read(k12 / "prior_policy_runs.csv"),
            k=12,
            policy="normalized_6_over_k",
        ),
        "prior": _filter(
            _read(k12 / "prior_policy_prior_diagnostics.csv"),
            k=12,
            policy="normalized_6_over_k",
        ),
    }
    result["runs"] = _localize_artifact_paths(result["runs"], root)
    return result


def _assert_grid(frame: pd.DataFrame, *, policies: Sequence[str]) -> None:
    keys = ["architecture", "encoder_seed", "hmm_seed", "K", "prior_policy"]
    if frame.duplicated(keys).any():
        raise ValueError("duplicate HMM grid cells")
    expected = {
        (ARCHITECTURE, enc, hmm, k, policy)
        for policy in policies
        for enc in ENCODER_SEEDS
        for hmm in HMM_SEEDS
        for k in K_VALUES
    }
    observed = {
        (str(row.architecture), int(row.encoder_seed), int(row.hmm_seed), int(row.K), str(row.prior_policy))
        for row in frame.itertuples()
        if row.status == "completed"
    }
    missing = sorted(expected - observed)
    if missing:
        raise ValueError(f"incomplete K=12/13/14 grid: {missing}")


def _paired_likelihood(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    completed = runs[runs.status.eq("completed")]
    for (enc, hmm), group in completed.groupby(["encoder_seed", "hmm_seed"]):
        by_k = {int(row.K): row for row in group.itertuples()}
        for left, right in ((12, 13), (12, 14), (13, 14)):
            a, b = by_k[left], by_k[right]
            rows.append(
                {
                    "encoder_seed": int(enc),
                    "hmm_seed": int(hmm),
                    "left_K": left,
                    "right_K": right,
                    "dev_ll_difference_right_minus_left": float(
                        b.dev_log_likelihood_total - a.dev_log_likelihood_total
                    ),
                    "dev_ll_per_point_difference_right_minus_left": float(
                        b.dev_log_likelihood_per_point - a.dev_log_likelihood_per_point
                    ),
                    "aic_difference_right_minus_left": float(b.aic_train - a.aic_train),
                    "bic_difference_right_minus_left": float(b.bic_train - a.bic_train),
                    "right_has_better_dev_ll": bool(
                        b.dev_log_likelihood_total > a.dev_log_likelihood_total
                    ),
                }
            )
    return pd.DataFrame(rows)


def _bout_pairs(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (enc, hmm), group in runs[runs.status.eq("completed")].groupby(
        ["encoder_seed", "hmm_seed"]
    ):
        by_k = {int(row.K): row for row in group.itertuples()}
        for left, right in ((12, 13), (13, 14), (12, 14)):
            a, b = by_k[left], by_k[right]
            rows.append(
                {
                    "encoder_seed": int(enc),
                    "hmm_seed": int(hmm),
                    "left_K": left,
                    "right_K": right,
                    "delta_bouts_per_minute": float(b.dev_bouts_per_minute - a.dev_bouts_per_minute),
                    "delta_median_bout_duration_s": float(b.dev_median_bout_duration_s - a.dev_median_bout_duration_s),
                    "delta_fraction_shorter_0p20_s": float(b.dev_fraction_bouts_shorter_0p20_s - a.dev_fraction_bouts_shorter_0p20_s),
                    "delta_fraction_shorter_0p50_s": float(b.dev_fraction_bouts_shorter_0p50_s - a.dev_fraction_bouts_shorter_0p50_s),
                }
            )
    return pd.DataFrame(rows)


def _mean_summary(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for k, group in frame.groupby("K"):
        row: dict[str, Any] = {"K": int(k), "run_count": int(len(group))}
        for column in columns:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_sd"] = float(values.std(ddof=1))
            row[f"{column}_min"] = float(values.min())
            row[f"{column}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("K")


def _stability_summary(stability: dict[str, pd.DataFrame]) -> pd.DataFrame:
    within = stability["within_pairwise"]
    cross = stability["cross_pairwise"]
    rows: list[dict[str, Any]] = []
    for k in K_VALUES:
        w = within[within.K.eq(k)]
        c = cross[cross.K.eq(k)]
        matched = c[c.hmm_seed_relation.eq("matched_hmm_seed")]
        rows.append(
            {
                "K": k,
                "within_ari_mean": float(w.ARI.mean()),
                "within_ari_min": float(w.ARI.min()),
                "within_nmi_mean": float(w.NMI.mean()),
                "within_matched_state_f1_mean": float(w.mean_hungarian_f1.mean()),
                "within_boundary_f1_mean": float(w.boundary_f1_exact.mean()),
                "cross_matched_ari_mean": float(matched.ARI.mean()),
                "cross_matched_ari_min": float(matched.ARI.min()),
                "cross_all_ari_mean": float(c.ARI.mean()),
                "cross_all_nmi_mean": float(c.NMI.mean()),
                "cross_all_matched_state_f1_mean": float(c.mean_hungarian_f1.mean()),
                "cross_all_boundary_f1_mean": float(c.boundary_f1_exact.mean()),
            }
        )
    return pd.DataFrame(rows)


def _refinement_summary(
    pairs: pd.DataFrame,
    split_merge: pd.DataFrame,
    stability_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for left, right in ((12, 13), (13, 14), (12, 14)):
        pair = pairs[pairs.left_K.eq(left) & pairs.right_K.eq(right)]
        events = split_merge[
            split_merge.left_K.eq(left)
            & split_merge.right_K.eq(right)
            & split_merge.direction.eq("left_to_right")
            & split_merge.split_candidate.fillna(False)
        ]
        parent_ari = float(
            stability_summary.loc[
                stability_summary.K.eq(left), "cross_matched_ari_mean"
            ].iloc[0]
        )
        run_events = events.groupby(["encoder_seed", "hmm_seed"]).size()
        event_fraction = float(len(run_events) / 9.0)
        mean_margin = float(events.ambiguity_margin.mean()) if not events.empty else 0.0
        parent_correspondence = "credible" if parent_ari >= 0.50 else "ambiguous"
        supported = bool(
            event_fraction >= 2.0 / 3.0
            and float(pair.ARI.mean()) >= 0.65
            and mean_margin >= 0.20
            and parent_correspondence == "credible"
        )
        rows.append(
            {
                "left_K": left,
                "right_K": right,
                "mean_cross_k_ari": float(pair.ARI.mean()),
                "mean_cross_k_nmi": float(pair.NMI.mean()),
                "runs_with_split_event": int(len(run_events)),
                "fraction_runs_with_split_event": event_fraction,
                "mean_split_ambiguity_margin": mean_margin,
                "parent_cross_encoder_ari": parent_ari,
                "parent_correspondence": parent_correspondence,
                "additional_state_support": "supported" if supported else "ambiguous_or_inconsistent",
            }
        )
    return pd.DataFrame(rows)


def select_working_k(
    summary: pd.DataFrame,
    refinement: pd.DataFrame,
    *,
    cross_tolerance: float = 0.02,
    within_tolerance: float = 0.03,
) -> tuple[int, list[int]]:
    """Apply declared reproducibility tolerances, then structural/parsimony rule."""

    best_cross = float(summary.cross_matched_ari_mean.max())
    cross_ok = summary[
        summary.cross_matched_ari_mean >= best_cross - cross_tolerance
    ]
    best_within = float(cross_ok.within_ari_mean.max())
    eligible = sorted(
        int(value)
        for value in cross_ok.loc[
            cross_ok.within_ari_mean >= best_within - within_tolerance, "K"
        ]
    )
    if not eligible:
        raise ValueError("working-K decision has no valid candidates")
    selected = eligible[0]
    for candidate in eligible[1:]:
        relation = refinement[
            refinement.left_K.eq(selected) & refinement.right_K.eq(candidate)
        ]
        if not relation.empty and relation.iloc[0].additional_state_support == "supported":
            selected = candidate
    return selected, eligible


def _decision_matrix(
    runs: pd.DataFrame,
    stability_summary: pd.DataFrame,
    refinement: pd.DataFrame,
) -> pd.DataFrame:
    fit = _mean_summary(
        runs,
        [
            "dev_log_likelihood_per_point",
            "aic_train",
            "bic_train",
            "dev_effective_state_ratio",
            "dev_largest_state_occupancy",
            "dev_bouts_per_minute",
            "dev_median_bout_duration_s",
            "dev_fraction_bouts_shorter_0p20_s",
        ],
    )
    result = fit.merge(stability_summary, on="K", validate="one_to_one")
    support = {12: "reference"}
    support.update(
        {
            int(row.right_K): row.additional_state_support
            for row in refinement.itertuples()
            if int(row.left_K) in (12, 13)
        }
    )
    result["added_state_support"] = result.K.map(support)
    result["statistical_fit_rank"] = result.dev_log_likelihood_per_point_mean.rank(
        ascending=False, method="min"
    ).astype(int)
    result["cross_encoder_rank"] = result.cross_matched_ari_mean.rank(
        ascending=False, method="min"
    ).astype(int)
    result["within_encoder_rank"] = result.within_ari_mean.rank(
        ascending=False, method="min"
    ).astype(int)
    result["fragmentation_rank"] = result.dev_bouts_per_minute_mean.rank(
        ascending=True, method="min"
    ).astype(int)
    return result


def _make_figures(
    output: Path,
    fixed: pd.DataFrame,
    policy: pd.DataFrame,
    stability: dict[str, pd.DataFrame],
    cross_k: dict[str, pd.DataFrame],
    refinement: pd.DataFrame,
) -> list[str]:
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    paths: list[str] = []

    def save(name: str) -> None:
        path = figures / name
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(str(path))

    for column, name, ylabel in (
        ("dev_log_likelihood_per_point", "development_likelihood.png", "Dev LL / point"),
        ("dev_bouts_per_minute", "bouts_per_minute.png", "Bouts / min"),
        ("dev_median_bout_duration_s", "median_bout_duration.png", "Median bout (s)"),
        ("dev_effective_state_ratio", "effective_state_ratio.png", "K_eff / K"),
    ):
        plt.figure(figsize=(6.2, 4.2))
        fixed.boxplot(column=column, by="K", grid=False)
        plt.suptitle("")
        plt.title(column.replace("_", " "))
        plt.ylabel(ylabel)
        save(name)

    delta = fixed.copy()
    for column, name in (("aic_train", "aic_relative_k12.png"), ("bic_train", "bic_relative_k12.png")):
        pivot = delta.pivot_table(index=["encoder_seed", "hmm_seed"], columns="K", values=column)
        relative = pivot.subtract(pivot[12], axis=0)
        plt.figure(figsize=(6.2, 4.2))
        plt.boxplot([relative[k] for k in K_VALUES], labels=K_VALUES)
        plt.axhline(0, color="black", linewidth=0.8)
        plt.ylabel(f"{column} difference vs K=12")
        plt.xlabel("K")
        save(name)

    for frame, name, title in (
        (stability["within_pairwise"], "within_encoder_ari.png", "Within-encoder HMM-seed stability"),
        (stability["cross_pairwise"], "cross_encoder_ari.png", "Cross-encoder stability"),
    ):
        plt.figure(figsize=(6.2, 4.2))
        plt.boxplot([frame.loc[frame.K.eq(k), "ARI"] for k in K_VALUES], labels=K_VALUES)
        plt.ylim(0, 1)
        plt.xlabel("K")
        plt.ylabel("ARI")
        plt.title(title)
        save(name)

    for left, right in ((12, 13), (13, 14)):
        pair = cross_k["cross_k_pairs"].query("left_K == @left and right_K == @right").iloc[0]
        matrix = np.load(pair.matrix_path)["row_normalized"]
        plt.figure(figsize=(6.0, 4.8))
        plt.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        plt.colorbar(label="P(right state | left state)")
        plt.xlabel(f"K={right} state")
        plt.ylabel(f"K={left} state")
        plt.title(f"Representative K={left} to K={right} overlap")
        save(f"overlap_k{left}_k{right}.png")

    plt.figure(figsize=(6.2, 4.2))
    labels = [f"{row.left_K}->{row.right_K}" for row in refinement.itertuples()]
    plt.bar(labels, refinement.fraction_runs_with_split_event)
    plt.ylim(0, 1)
    plt.ylabel("Fraction of runs with split candidate")
    save("additional_state_refinement_recurrence.png")

    policy_summary = policy.groupby(["prior_policy", "K"]).dev_bouts_per_minute.mean().unstack(0)
    plt.figure(figsize=(6.2, 4.2))
    for column in policy_summary:
        plt.plot(policy_summary.index, policy_summary[column], marker="o", label=column)
    plt.xlabel("K")
    plt.ylabel("Mean bouts / min")
    plt.legend()
    save("fixed_vs_normalized_prior.png")
    return paths


def _report(
    output: Path,
    selected: int,
    eligible: list[int],
    decision: pd.DataFrame,
    refinement: pd.DataFrame,
    policy_stability: dict[str, pd.DataFrame],
    prior_selected: dict[str, int],
    confidence: str,
) -> Path:
    by_k = {int(row.K): row for row in decision.itertuples()}
    best_ll = int(decision.sort_values("dev_log_likelihood_per_point_mean", ascending=False).iloc[0].K)
    best_aic = int(decision.sort_values("aic_train_mean").iloc[0].K)
    best_bic = int(decision.sort_values("bic_train_mean").iloc[0].K)
    best_within = int(decision.sort_values("within_ari_mean", ascending=False).iloc[0].K)
    best_cross = int(decision.sort_values("cross_matched_ari_mean", ascending=False).iloc[0].K)
    support13 = refinement.query("left_K == 12 and right_K == 13").iloc[0]
    support14 = refinement.query("left_K == 13 and right_K == 14").iloc[0]
    robust = len(set(prior_selected.values())) == 1
    lines = [
        "# 0.04 s Social-GNN: Final Sticky HMM Working-State Selection (K=12/13/14)",
        "",
        "## Executive summary",
        "",
        f"- Best development likelihood: K={best_ll}.",
        f"- Best training AIC: K={best_aic}; best training BIC: K={best_bic}.",
        f"- Best within-encoder initialization stability: K={best_within}.",
        f"- Best matched-seed cross-encoder stability: K={best_cross}.",
        f"- K=13 added-state refinement: {support13.additional_state_support}.",
        f"- K=14 added-state refinement: {support14.additional_state_support}.",
        f"- Prior-policy recommendation agreement: {'yes' if robust else 'no'} ({prior_selected}).",
        "",
        f"**Selected working K: {selected}**",
        "",
        f"Confidence: **{confidence}**. Eligible candidates within the declared reproducibility tolerances were {eligible}; the predefined parsimony rule was applied unless a higher-K refinement met all support criteria.",
        "",
        "## Candidate metrics",
        "",
        "| K | dev LL/point | within ARI | cross matched ARI | bouts/min | median bout s | added-state support |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for k in K_VALUES:
        row = by_k[k]
        lines.append(
            f"| {k} | {row.dev_log_likelihood_per_point_mean:.6f} | {row.within_ari_mean:.3f} | {row.cross_matched_ari_mean:.3f} | {row.dev_bouts_per_minute_mean:.2f} | {row.dev_median_bout_duration_s_mean:.3f} | {row.added_state_support} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Likelihood and information criteria are reported but do not override reproducibility. Added-state support requires recurrence across runs, concentrated overlap, and credible correspondence of the lower-K parent partition. When parent correspondence is ambiguous, the refinement is deliberately classified as ambiguous rather than forced.",
            "",
            "Temporal fragmentation is evaluated on A4_C4 at the authoritative 0.04 s clock without transitions across invalid gaps. A4_C4 is reused development data and is not an untouched test set.",
            "",
            "The selected K is an operational default, not an estimate of the natural number of behaviors. No state has been assigned a behavioral name.",
            "",
            "## Prior-policy sensitivity",
            "",
            "The fixed 50,000-per-row prior is the primary analysis. The 50,000×6/K policy is a prespecified heuristic sensitivity control. Disagreement between policies lowers confidence rather than allowing post-hoc selection of the more favorable prior.",
            "",
            "## Limitations",
            "",
            "- One development animal pair is reused across all comparisons.",
            "- Pairwise ARIs share observations and fitted models and are not biological replicates.",
            "- Overlap-supported refinements are structural diagnostics, not biological validation.",
            "- A5_C5 remains locked and was not used.",
        ]
    )
    path = output / "final_k_selection_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_final_k_selection(
    *,
    root: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT,
    workers: int = 3,
    max_iter: int = 40,
    tolerance: float = 1e-4,
    covariance_floor: float = 1e-3,
    aggregate_existing: bool = False,
) -> dict[str, Any]:
    root = Path(root).resolve()
    output = (root / output_dir).resolve() if not Path(output_dir).is_absolute() else Path(output_dir).resolve()
    k12 = root / BASELINE_K12
    k30 = root / BASELINE_K30
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    if aggregate_existing:
        new_fixed = _load_collection(output, "new_fixed_k13")
        new_normalized = _load_collection(output, "new_normalized_k13_k14")
    else:
        new_fixed = _run_workers(
            root=root,
            output=output,
            policy="fixed",
            k_by_architecture={ARCHITECTURE: (13,)},
            workers=workers,
            max_iter=max_iter,
            tolerance=tolerance,
            covariance_floor=covariance_floor,
            architectures=(ARCHITECTURE,),
            encoder_seeds=ENCODER_SEEDS,
            hmm_seeds=HMM_SEEDS,
        )
        _save_collection(output, "new_fixed_k13", new_fixed)
        new_normalized = _run_workers(
            root=root,
            output=output / "prior_policy",
            policy="normalized_6_over_k",
            k_by_architecture={ARCHITECTURE: (13, 14)},
            workers=workers,
            max_iter=max_iter,
            tolerance=tolerance,
            covariance_floor=covariance_floor,
            architectures=(ARCHITECTURE,),
            encoder_seeds=ENCODER_SEEDS,
            hmm_seeds=HMM_SEEDS,
        )
        _save_collection(output, "new_normalized_k13_k14", new_normalized)

    historical_fixed = _historical_fixed(k12, k30, root)
    fixed = pd.concat(
        [historical_fixed["runs"], pd.DataFrame(new_fixed["runs"])],
        ignore_index=True,
        sort=False,
    )
    _assert_grid(fixed, policies=("fixed",))
    for k, expected in EXPECTED_PARAMETER_COUNT.items():
        actual = set(fixed.loc[fixed.K.eq(k), "parameter_count"].astype(int))
        if actual != {expected} or parameter_count(k) != expected:
            raise ValueError(f"K={k}: parameter count mismatch {actual} != {expected}")

    per_state = pd.concat(
        [historical_fixed["per_state"], pd.DataFrame(new_fixed["per_state"])],
        ignore_index=True,
        sort=False,
    )
    prior_fixed = pd.concat(
        [historical_fixed["prior"], pd.DataFrame(new_fixed["prior"])],
        ignore_index=True,
        sort=False,
    )
    historical_normalized = _historical_normalized_k12(k12, root)
    normalized = pd.concat(
        [historical_normalized["runs"], pd.DataFrame(new_normalized["runs"])],
        ignore_index=True,
        sort=False,
    )
    policy_runs = pd.concat([fixed, normalized], ignore_index=True, sort=False)
    _assert_grid(policy_runs, policies=("fixed", "normalized_6_over_k"))

    fixed.to_csv(output / "run_level_metrics.csv", index=False)
    paired = _paired_likelihood(fixed)
    paired.to_csv(output / "paired_likelihood_comparison.csv", index=False)
    _mean_summary(fixed, ["aic_train", "bic_train", "dev_log_likelihood_per_point"]).to_csv(
        output / "aic_bic_comparison.csv", index=False
    )
    fixed[
        [
            "encoder_seed", "hmm_seed", "K", "dev_occupied_states",
            "dev_effective_states", "dev_effective_state_ratio",
            "dev_largest_state_occupancy", "dev_smallest_nonempty_state_occupancy",
            "dev_mean_posterior_entropy",
        ]
    ].to_csv(output / "state_occupancy_comparison.csv", index=False)
    bouts = _bout_pairs(fixed)
    bouts.to_csv(output / "bout_comparison.csv", index=False)

    stability = stability_tables(fixed)
    stability["within_pairwise"].to_csv(output / "within_encoder_ari.csv", index=False)
    stability["cross_pairwise"].to_csv(output / "cross_encoder_ari.csv", index=False)
    stability_summary = _stability_summary(stability)
    stability_summary.to_csv(output / "stability_summary.csv", index=False)

    cross_k = cross_k_tables(
        fixed,
        per_state,
        output,
        k_grid=K_VALUES,
        reference_k=REFERENCE_K,
        adjacent_pairs=((12, 13), (13, 14), (12, 14)),
    )
    cross_k["cross_k_pairs"].to_csv(output / "cross_k_state_correspondence.csv", index=False)
    cross_k["cross_k_cells"].to_csv(output / "cross_k_overlap_cells.csv", index=False)
    cross_k["cross_k_split_merge"].to_csv(output / "cross_k_split_merge.csv", index=False)
    refinement = _refinement_summary(
        cross_k["cross_k_pairs"], cross_k["cross_k_split_merge"], stability_summary
    )
    refinement.to_csv(output / "new_state_refinement_diagnostics.csv", index=False)

    policy_runs.to_csv(output / "prior_policy_comparison.csv", index=False)
    prior_all = pd.concat(
        [
            prior_fixed,
            historical_normalized["prior"],
            pd.DataFrame(new_normalized["prior"]),
        ],
        ignore_index=True,
        sort=False,
    )
    prior_all.to_csv(output / "prior_data_leverage.csv", index=False)
    policy_stability = prior_policy_stability_tables(policy_runs)
    for name, frame in policy_stability.items():
        frame.to_csv(output / f"{name}.csv", index=False)

    decision = _decision_matrix(fixed, stability_summary, refinement)
    selected, eligible = select_working_k(decision, refinement)
    decision.to_csv(output / "candidate_decision_matrix.csv", index=False)

    prior_selected: dict[str, int] = {"fixed": selected}
    for policy, subset in policy_runs.groupby("prior_policy"):
        policy_tables = stability_tables(subset)
        policy_summary = _stability_summary(policy_tables)
        policy_decision = _decision_matrix(subset, policy_summary, refinement)
        prior_selected[str(policy)] = select_working_k(policy_decision, refinement)[0]
    robust = len(set(prior_selected.values())) == 1
    gap_to_best_cross = float(
        decision.cross_matched_ari_mean.max()
        - decision.loc[decision.K.eq(selected), "cross_matched_ari_mean"].iloc[0]
    )
    confidence = "moderate" if robust and gap_to_best_cross <= 0.02 else "provisional"

    representative = fixed[fixed.K.eq(selected) & fixed.status.eq("completed")].copy()
    median_fit = representative.train_log_likelihood_total.median()
    representative["distance_to_median_train_ll"] = (
        representative.train_log_likelihood_total - median_fit
    ).abs()
    representative_row = representative.sort_values(
        ["distance_to_median_train_ll", "encoder_seed", "hmm_seed"]
    ).iloc[0]
    selected_payload = {
        "contract": "social_gnn_selected_working_k_v1",
        "selected_K": int(selected),
        "confidence": confidence,
        "eligible_within_declared_reproducibility_tolerances": eligible,
        "architecture": ARCHITECTURE,
        "encoder_seeds_evaluated": list(ENCODER_SEEDS),
        "hmm_seeds_evaluated": list(HMM_SEEDS),
        "observation_dim": 16,
        "timestep_s": 0.04,
        "train_trials": ["A1_C1", "A3_C3", "A4_B4"],
        "development_trial": "A4_C4",
        "excluded_trial": "A5_C5",
        "primary_prior": {"strength_per_source_row": 50000, "target_self_transition": 0.98, "target_dwell_s": 2.0},
        "secondary_prior": {"formula": "50000 * 6 / K", "policy_recommendations": prior_selected},
        "decision_rule": {"cross_ari_tolerance": 0.02, "within_ari_tolerance": 0.03, "parsimony_if_tied": True},
        "representative_run_rule": "closest training log-likelihood to the within-K median; ties by encoder seed then HMM seed",
        "representative_checkpoint": str(representative_row.checkpoint),
        "representative_assignment": str(representative_row.assignment),
        "all_selected_k_checkpoints": sorted(str(value) for value in representative.checkpoint),
        "limitations": ["A4_C4 is reused development data", "working K is operational, not a natural behavior count"],
    }
    _write_json(output / "selected_working_k.json", selected_payload)
    figures = _make_figures(output, fixed, policy_runs, stability, cross_k, refinement)
    report = _report(
        output,
        selected,
        eligible,
        decision,
        refinement,
        policy_stability,
        prior_selected,
        confidence,
    )
    failures = pd.concat(
        [pd.DataFrame(new_fixed["failures"]), pd.DataFrame(new_normalized["failures"])],
        ignore_index=True,
        sort=False,
    )
    failures.to_csv(output / "failures.csv", index=False)
    config = root / CONFIG
    manifest = {
        "contract": "social_gnn_hmm_k12_k14_final_selection_v1",
        "status": "completed",
        "selected_K": int(selected),
        "confidence": confidence,
        "fixed_runs": int(len(fixed)),
        "normalized_runs": int(len(normalized)),
        "new_fixed_runs": int(len(pd.DataFrame(new_fixed["runs"]))),
        "new_normalized_runs": int(len(pd.DataFrame(new_normalized["runs"]))),
        "failed_runs": int(len(failures)),
        "reused_K12_K14_fixed_runs": 18,
        "reused_K12_normalized_runs": 9,
        "config": str(config),
        "config_sha256": _sha256(config),
        "historical_sources": [str(k12), str(k30)],
        "report": str(report),
        "selected_working_k": str(output / "selected_working_k.json"),
        "figures": figures,
        "elapsed_s": time.perf_counter() - started,
    }
    _write_json(output / "experiment_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path.cwd()))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--covariance-floor", type=float, default=1e-3)
    parser.add_argument("--aggregate-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_final_k_selection(
        root=args.root,
        output_dir=args.output_dir,
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
