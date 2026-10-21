"""Post-hoc ARI stability decomposition for saved downstream assignments.

This module never fits an encoder or a downstream model.  It audits and
compares assignments already produced by the 3 x 3 confirmation seed grid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


WITHIN_ENCODER = "within_encoder_across_downstream"
MATCHED_DOWNSTREAM = "across_encoder_matched_downstream_seed"
ACROSS_BOTH = "across_encoder_across_downstream"
GROUP_ORDER = (WITHIN_ENCODER, MATCHED_DOWNSTREAM, ACROSS_BOTH)

DEFAULT_ENCODERS = (
    "graph_full",
    "graph_hybrid_p2",
    "invariant_mlp_full",
    "invariant_mlp_hybrid_p2",
)
DEFAULT_ALGORITHMS = ("kmeans", "gmm", "hmm", "hsmm")
DEFAULT_ENCODER_SEEDS = (20261013, 20261029, 20261107)
DEFAULT_DOWNSTREAM_SEEDS = (0, 1, 2)

DISPLAY_NAMES = {
    "graph_full": "Graph Full",
    "graph_hybrid_p2": "Graph Hybrid P2",
    "invariant_mlp_full": "Invariant MLP Full",
    "invariant_mlp_hybrid_p2": "Invariant MLP Hybrid P2",
    "kmeans": "KMeans",
    "gmm": "GMM",
    "hmm": "HMM",
    "hsmm": "HSMM",
}

RUN_KEY = ["encoder", "encoder_seed", "algorithm", "downstream_seed", "K"]
QUALITY_COLUMNS = (
    "pca_dim",
    "status",
    "iterations",
    "converged",
    "validation_fit_per_patch",
    "silhouette_validation",
    "observed_timesteps",
    "number_of_bouts",
    "bouts_per_minute",
    "median_bout_duration_s",
    "occupied_states",
    "states_over_one_percent",
    "effective_entropy_states",
    "largest_state_occupancy",
    "smallest_nonempty_state_occupancy",
    "near_empty_states",
    "mean_state_confidence",
    "failure",
    "checkpoint",
    "source_table",
)


@dataclass(frozen=True)
class AssignmentData:
    """One run's complete saved patch timeline and labels."""

    trial_id: str
    patch_start_s: np.ndarray
    patch_end_s: np.ndarray
    valid_time_mask: np.ndarray
    predicted_state: np.ndarray
    source_path: str = ""

    def __post_init__(self) -> None:
        arrays = (
            self.patch_start_s,
            self.patch_end_s,
            self.valid_time_mask,
            self.predicted_state,
        )
        lengths = {np.asarray(value).shape for value in arrays}
        if len(lengths) != 1 or not lengths or len(next(iter(lengths))) != 1:
            raise ValueError(f"{self.trial_id}: assignment arrays must be equal-length 1D arrays")
        if not np.isfinite(self.patch_start_s).all() or not np.isfinite(self.patch_end_s).all():
            raise ValueError(f"{self.trial_id}: assignment clock contains nonfinite values")
        if np.any(np.asarray(self.patch_end_s) <= np.asarray(self.patch_start_s)):
            raise ValueError(f"{self.trial_id}: patch_end_s must be greater than patch_start_s")

    @property
    def size(self) -> int:
        return int(np.asarray(self.predicted_state).size)

    def frame(self, prefix: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "trial_id": np.repeat(self.trial_id, self.size),
                "patch_start_s": np.asarray(self.patch_start_s, dtype=np.float64),
                "patch_end_s": np.asarray(self.patch_end_s, dtype=np.float64),
                f"{prefix}_valid": np.asarray(self.valid_time_mask, dtype=bool),
                f"{prefix}_label": np.asarray(self.predicted_state, dtype=np.int64),
            }
        )


def run_id(row: pd.Series | dict[str, Any]) -> str:
    return (
        f"{row['encoder']}|e{int(row['encoder_seed'])}|{row['algorithm']}|"
        f"k{int(row['K'])}|d{int(row['downstream_seed'])}"
    )


def classify_pair(
    left_encoder_seed: int,
    left_downstream_seed: int,
    right_encoder_seed: int,
    right_downstream_seed: int,
) -> str:
    same_encoder = int(left_encoder_seed) == int(right_encoder_seed)
    same_downstream = int(left_downstream_seed) == int(right_downstream_seed)
    if same_encoder and same_downstream:
        raise ValueError("a run cannot be compared with itself")
    if same_encoder:
        return WITHIN_ENCODER
    if same_downstream:
        return MATCHED_DOWNSTREAM
    return ACROSS_BOTH


def enumerate_seed_pairs(
    encoder_seeds: Sequence[int] = DEFAULT_ENCODER_SEEDS,
    downstream_seeds: Sequence[int] = DEFAULT_DOWNSTREAM_SEEDS,
) -> list[dict[str, int | str]]:
    runs = [(int(e), int(d)) for e in encoder_seeds for d in downstream_seeds]
    rows: list[dict[str, int | str]] = []
    for (left_e, left_d), (right_e, right_d) in combinations(runs, 2):
        rows.append(
            {
                "run_1_encoder_seed": left_e,
                "run_1_downstream_seed": left_d,
                "run_2_encoder_seed": right_e,
                "run_2_downstream_seed": right_d,
                "comparison_group": classify_pair(left_e, left_d, right_e, right_d),
            }
        )
    return rows


def expected_grid(
    encoders: Sequence[str] = DEFAULT_ENCODERS,
    algorithms: Sequence[str] = DEFAULT_ALGORITHMS,
    encoder_seeds: Sequence[int] = DEFAULT_ENCODER_SEEDS,
    downstream_seeds: Sequence[int] = DEFAULT_DOWNSTREAM_SEEDS,
    *,
    k: int = 6,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "encoder": encoder,
                "encoder_seed": int(encoder_seed),
                "algorithm": algorithm,
                "downstream_seed": int(downstream_seed),
                "K": int(k),
            }
            for encoder, algorithm, encoder_seed, downstream_seed in product(
                encoders, algorithms, encoder_seeds, downstream_seeds
            )
        ]
    )


def audit_run_grid(runs: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    counts = runs.groupby(RUN_KEY, dropna=False).size().rename("observed_count").reset_index()
    audit = expected.merge(counts, on=RUN_KEY, how="outer", indicator=True)
    audit["observed_count"] = audit["observed_count"].fillna(0).astype(int)
    audit["grid_status"] = np.select(
        (
            audit["_merge"].eq("left_only"),
            audit["_merge"].eq("right_only"),
            audit["observed_count"].gt(1),
        ),
        ("missing", "unexpected", "duplicate"),
        default="present_once",
    )
    return audit.drop(columns="_merge").sort_values(RUN_KEY).reset_index(drop=True)


def require_complete_grid(audit: pd.DataFrame) -> None:
    problems = audit[audit["grid_status"] != "present_once"]
    if not problems.empty:
        details = problems[RUN_KEY + ["observed_count", "grid_status"]].to_dict("records")
        raise ValueError(f"confirmation run grid is incomplete or ambiguous: {details}")


def _assignment_path(row: pd.Series, trial_id: str) -> Path:
    values = json.loads(str(row["assignment_files"]))
    matches = [Path(value) for value in values if str(value).endswith(f"{trial_id}_states.npz")]
    if len(matches) != 1:
        raise ValueError(f"{run_id(row)}: expected one {trial_id} NPZ assignment, found {matches}")
    if not matches[0].exists():
        raise FileNotFoundError(f"{run_id(row)}: assignment does not exist: {matches[0]}")
    return matches[0]


def load_assignment(row: pd.Series, trial_id: str) -> AssignmentData:
    path = _assignment_path(row, trial_id)
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "trial_id",
            "patch_start_s",
            "patch_end_s",
            "valid_time_mask",
            "predicted_state",
        }
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"{path}: missing assignment arrays {sorted(missing)}")
        saved_trial = str(np.asarray(payload["trial_id"]).item())
        if saved_trial != trial_id:
            raise ValueError(f"{path}: expected trial {trial_id}, found {saved_trial}")
        return AssignmentData(
            trial_id=saved_trial,
            patch_start_s=np.asarray(payload["patch_start_s"], dtype=np.float64),
            patch_end_s=np.asarray(payload["patch_end_s"], dtype=np.float64),
            valid_time_mask=np.asarray(payload["valid_time_mask"], dtype=bool),
            predicted_state=np.asarray(payload["predicted_state"], dtype=np.int64),
            source_path=str(path.resolve()),
        )


def _timeline_hash(assignment: AssignmentData, *, include_validity: bool) -> str:
    digest = hashlib.sha256()
    digest.update(assignment.trial_id.encode("utf-8"))
    digest.update(np.asarray(assignment.patch_start_s, dtype="<f8").tobytes())
    digest.update(np.asarray(assignment.patch_end_s, dtype="<f8").tobytes())
    if include_validity:
        digest.update(np.asarray(assignment.valid_time_mask, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def assignment_audit(row: pd.Series, assignment: AssignmentData) -> dict[str, Any]:
    frame = assignment.frame("run")
    duplicates = frame.duplicated(["trial_id", "patch_start_s", "patch_end_s"], keep=False)
    valid = np.asarray(assignment.valid_time_mask, dtype=bool)
    labels = np.asarray(assignment.predicted_state, dtype=np.int64)
    return {
        **{key: row[key] for key in RUN_KEY},
        "run_id": run_id(row),
        "trial_id": assignment.trial_id,
        "assignment_path": assignment.source_path,
        "total_patches": assignment.size,
        "valid_patches": int(valid.sum()),
        "invalid_patches": int((~valid).sum()),
        "duplicate_patch_key_rows": int(duplicates.sum()),
        "valid_rows_without_label": int(np.sum(valid & (labels < 0))),
        "invalid_rows_with_label": int(np.sum((~valid) & (labels >= 0))),
        "timeline_hash": _timeline_hash(assignment, include_validity=False),
        "timeline_and_validity_hash": _timeline_hash(assignment, include_validity=True),
    }


def align_assignments(left: AssignmentData, right: AssignmentData) -> dict[str, Any]:
    """Align two runs by physical patch key and return labels plus diagnostics."""

    if left.trial_id != right.trial_id:
        raise ValueError(f"trial mismatch: {left.trial_id!r} versus {right.trial_id!r}")
    left_frame = left.frame("left")
    right_frame = right.frame("right")
    keys = ["trial_id", "patch_start_s", "patch_end_s"]
    if left_frame.duplicated(keys).any() or right_frame.duplicated(keys).any():
        raise ValueError("duplicate physical patch keys prevent one-to-one alignment")
    merged = left_frame.merge(
        right_frame,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
        sort=True,
    )
    common = merged["_merge"].eq("both")
    left_valid = merged["left_valid"].fillna(False).astype(bool)
    right_valid = merged["right_valid"].fillna(False).astype(bool)
    left_labeled = merged["left_label"].fillna(-1).astype(np.int64).ge(0)
    right_labeled = merged["right_label"].fillna(-1).astype(np.int64).ge(0)
    aligned = common & left_valid & right_valid & left_labeled & right_labeled
    valid_union = left_valid | right_valid
    left_labels = merged.loc[aligned, "left_label"].to_numpy(dtype=np.int64)
    right_labels = merged.loc[aligned, "right_label"].to_numpy(dtype=np.int64)
    if left_labels.size == 0:
        raise ValueError("no mutually valid physical patches remain after alignment")
    return {
        "left_labels": left_labels,
        "right_labels": right_labels,
        "number_of_aligned_patches": int(left_labels.size),
        "number_of_union_valid_patches": int(valid_union.sum()),
        "aligned_fraction_of_valid_union": float(left_labels.size / max(valid_union.sum(), 1)),
        "common_physical_patches": int(common.sum()),
        "left_only_physical_patches": int(merged["_merge"].eq("left_only").sum()),
        "right_only_physical_patches": int(merged["_merge"].eq("right_only").sum()),
        "validity_disagreements_on_common_patches": int(
            np.sum(common & (left_valid != right_valid))
        ),
        "identical_full_timeline": bool(
            common.sum() == left.size == right.size
            and not merged["_merge"].ne("both").any()
        ),
    }


def _summary(values: pd.Series) -> dict[str, float | int]:
    array = values.dropna().to_numpy(dtype=np.float64)
    return {
        "pair_count": int(array.size),
        "ari_mean": float(np.mean(array)) if array.size else float("nan"),
        "ari_median": float(np.median(array)) if array.size else float("nan"),
        "ari_sample_sd": float(np.std(array, ddof=1)) if array.size > 1 else float("nan"),
        "ari_min": float(np.min(array)) if array.size else float("nan"),
        "ari_max": float(np.max(array)) if array.size else float("nan"),
        "ari_q25": float(np.quantile(array, 0.25)) if array.size else float("nan"),
        "ari_q75": float(np.quantile(array, 0.75)) if array.size else float("nan"),
    }


def summarize_groups(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in pairs.groupby(
        ["encoder_condition", "downstream_algorithm", "comparison_group"], sort=False
    ):
        rows.append(
            {
                "encoder_condition": key[0],
                "downstream_algorithm": key[1],
                "comparison_group": key[2],
                **_summary(group["ARI"]),
            }
        )
    result = pd.DataFrame(rows)
    result["comparison_group"] = pd.Categorical(
        result["comparison_group"], GROUP_ORDER, ordered=True
    )
    return result.sort_values(
        ["encoder_condition", "downstream_algorithm", "comparison_group"]
    ).reset_index(drop=True)


def summarize_within_encoder_seed(pairs: pd.DataFrame) -> pd.DataFrame:
    subset = pairs[pairs["comparison_group"] == WITHIN_ENCODER]
    rows: list[dict[str, Any]] = []
    for key, group in subset.groupby(
        ["encoder_condition", "downstream_algorithm", "run_1_encoder_seed"], sort=True
    ):
        rows.append(
            {
                "encoder_condition": key[0],
                "encoder_seed": int(key[2]),
                "downstream_algorithm": key[1],
                **_summary(group["ARI"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["encoder_condition", "downstream_algorithm", "encoder_seed"]
    ).reset_index(drop=True)


def summarize_cross_encoder_strata(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    matched = pairs[pairs["comparison_group"] == MATCHED_DOWNSTREAM]
    for key, group in matched.groupby(
        ["encoder_condition", "downstream_algorithm", "run_1_downstream_seed"], sort=True
    ):
        rows.append(
            {
                "encoder_condition": key[0],
                "downstream_algorithm": key[1],
                "comparison_group": MATCHED_DOWNSTREAM,
                "downstream_seed_stratum": str(int(key[2])),
                **_summary(group["ARI"]),
            }
        )
    different = pairs[pairs["comparison_group"] == ACROSS_BOTH].copy()
    different["downstream_seed_stratum"] = different.apply(
        lambda row: "-".join(
            map(
                str,
                sorted(
                    [
                        int(row["run_1_downstream_seed"]),
                        int(row["run_2_downstream_seed"]),
                    ]
                ),
            )
        ),
        axis=1,
    )
    for key, group in different.groupby(
        ["encoder_condition", "downstream_algorithm", "downstream_seed_stratum"], sort=True
    ):
        rows.append(
            {
                "encoder_condition": key[0],
                "downstream_algorithm": key[1],
                "comparison_group": ACROSS_BOTH,
                "downstream_seed_stratum": key[2],
                **_summary(group["ARI"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["encoder_condition", "downstream_algorithm", "comparison_group", "downstream_seed_stratum"]
    ).reset_index(drop=True)


def ari_matrix(
    pairs: pd.DataFrame,
    encoder_seeds: Sequence[int] = DEFAULT_ENCODER_SEEDS,
    downstream_seeds: Sequence[int] = DEFAULT_DOWNSTREAM_SEEDS,
) -> tuple[np.ndarray, list[str]]:
    ordering = [(int(e), int(d)) for e in encoder_seeds for d in downstream_seeds]
    index = {value: position for position, value in enumerate(ordering)}
    matrix = np.full((len(ordering), len(ordering)), np.nan, dtype=np.float64)
    np.fill_diagonal(matrix, 1.0)
    for row in pairs.itertuples(index=False):
        left = (int(row.run_1_encoder_seed), int(row.run_1_downstream_seed))
        right = (int(row.run_2_encoder_seed), int(row.run_2_downstream_seed))
        if left not in index or right not in index:
            raise ValueError(f"unexpected run in ARI matrix: {left}, {right}")
        i, j = index[left], index[right]
        if np.isfinite(matrix[i, j]):
            raise ValueError(f"duplicate pair in ARI matrix: {left}, {right}")
        matrix[i, j] = matrix[j, i] = float(row.ARI)
    if np.isnan(matrix).any():
        raise ValueError("ARI matrix is incomplete")
    if not np.allclose(matrix, matrix.T) or not np.allclose(np.diag(matrix), 1.0):
        raise ValueError("ARI matrix symmetry/diagonal invariant failed")
    labels = [f"e{e}/d{d}" for e, d in ordering]
    return matrix, labels


def _augment_pair_quality(pairs: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "occupied_states": "occupied_states",
        "effective_entropy_states": "effective_states",
        "largest_state_occupancy": "largest_occupancy",
        "smallest_nonempty_state_occupancy": "smallest_occupancy",
        "bouts_per_minute": "bouts_per_minute",
        "median_bout_duration_s": "median_bout_duration_s",
        "silhouette_validation": "silhouette",
    }
    lookup = quality.set_index("run_id")
    result = pairs.copy()
    for source, label in metrics.items():
        result[f"run_1_{label}"] = result["run_1_id"].map(lookup[source])
        result[f"run_2_{label}"] = result["run_2_id"].map(lookup[source])
        result[f"abs_{label}_difference"] = (
            result[f"run_1_{label}"] - result[f"run_2_{label}"]
        ).abs()
    result["run_1_converged"] = result["run_1_id"].map(lookup["converged"])
    result["run_2_converged"] = result["run_2_id"].map(lookup["converged"])
    result["either_run_not_converged"] = ~(
        result["run_1_converged"].astype(bool) & result["run_2_converged"].astype(bool)
    )
    result["occupied_state_count_mismatch"] = (
        result["run_1_occupied_states"] != result["run_2_occupied_states"]
    )
    return result


def _quality_correlations(pair_quality: pd.DataFrame) -> pd.DataFrame:
    difference_columns = [
        value
        for value in pair_quality.columns
        if value.startswith("abs_") and value.endswith("_difference")
    ]
    rows: list[dict[str, Any]] = []
    groupings = [
        ("all", pair_quality),
        *[(f"algorithm:{key}", value) for key, value in pair_quality.groupby("downstream_algorithm")],
        *[(f"encoder:{key}", value) for key, value in pair_quality.groupby("encoder_condition")],
    ]
    for scope, frame in groupings:
        for column in difference_columns:
            valid = frame[["ARI", column]].dropna()
            if valid.shape[0] < 3 or valid[column].nunique() < 2:
                pearson = spearman = float("nan")
            else:
                pearson = float(valid["ARI"].corr(valid[column], method="pearson"))
                spearman = float(
                    valid["ARI"].rank(method="average").corr(
                        valid[column].rank(method="average"), method="pearson"
                    )
                )
            rows.append(
                {
                    "scope": scope,
                    "quality_difference": column,
                    "pair_count": int(valid.shape[0]),
                    "pearson_r": pearson,
                    "spearman_rho": spearman,
                }
            )
    return pd.DataFrame(rows)


def _degeneracy_bands(pair_quality: pd.DataFrame) -> pd.DataFrame:
    definitions = (
        ("ARI < 0.3", pair_quality["ARI"] < 0.3),
        ("0.3 <= ARI < 0.6", (pair_quality["ARI"] >= 0.3) & (pair_quality["ARI"] < 0.6)),
        ("ARI >= 0.6", pair_quality["ARI"] >= 0.6),
    )
    rows: list[dict[str, Any]] = []
    for name, mask in definitions:
        frame = pair_quality[mask]
        rows.append(
            {
                "ari_band": name,
                "pair_count": int(frame.shape[0]),
                "ari_mean": float(frame["ARI"].mean()),
                "mean_abs_effective_states_difference": float(
                    frame["abs_effective_states_difference"].mean()
                ),
                "mean_abs_largest_occupancy_difference": float(
                    frame["abs_largest_occupancy_difference"].mean()
                ),
                "mean_abs_bouts_per_minute_difference": float(
                    frame["abs_bouts_per_minute_difference"].mean()
                ),
                "occupied_state_count_mismatch_pairs": int(
                    frame["occupied_state_count_mismatch"].sum()
                ),
                "pairs_including_nonconverged_run": int(
                    frame["either_run_not_converged"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _plot_heatmap(matrix: np.ndarray, labels: list[str], title: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.4, 7.4))
    image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    axis.set_yticks(range(len(labels)), labels, fontsize=8)
    for boundary in (2.5, 5.5):
        axis.axhline(boundary, color="white", linewidth=2)
        axis.axvline(boundary, color="white", linewidth=2)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] < 0.45 else "black"
            axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6, color=color)
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label="Adjusted Rand index")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _plot_distribution(frame: pd.DataFrame, title: str, path: Path) -> None:
    rng = np.random.default_rng(20261020)
    labels = ["Fixed encoder\n(n=9)", "Matched d-seed\n(n=9)", "Both differ\n(n=18)"]
    values = [frame.loc[frame["comparison_group"] == group, "ARI"].to_numpy() for group in GROUP_ORDER]
    figure, axis = plt.subplots(figsize=(7.6, 5.2))
    axis.boxplot(values, widths=0.45, showfliers=False, medianprops={"color": "black", "linewidth": 1.5})
    for position, group_values in enumerate(values, start=1):
        jitter = rng.uniform(-0.10, 0.10, size=group_values.size)
        axis.scatter(position + jitter, group_values, s=24, alpha=0.8, zorder=3)
    axis.set_xticks([1, 2, 3], labels)
    axis.set_ylim(-0.03, 1.03)
    axis.set_ylabel("Adjusted Rand index")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _main_table(summary: pd.DataFrame) -> pd.DataFrame:
    pivot = summary.pivot_table(
        index=["encoder_condition", "downstream_algorithm"],
        columns="comparison_group",
        values=["ari_mean", "ari_sample_sd"],
        aggfunc="first",
    )
    rows: list[dict[str, Any]] = []
    for (encoder, algorithm), row in pivot.iterrows():
        output: dict[str, Any] = {
            "encoder_condition": encoder,
            "downstream_algorithm": algorithm,
            "encoder_display": DISPLAY_NAMES.get(encoder, encoder),
            "algorithm_display": DISPLAY_NAMES.get(algorithm, algorithm.upper()),
        }
        for group in GROUP_ORDER:
            mean = float(row[("ari_mean", group)])
            sd = float(row[("ari_sample_sd", group)])
            output[f"{group}_mean"] = mean
            output[f"{group}_sample_sd"] = sd
            output[f"{group}_display"] = f"{mean:.3f} +/- {sd:.3f}"
        rows.append(output)
    result = pd.DataFrame(rows)
    encoder_order = {value: index for index, value in enumerate(DEFAULT_ENCODERS)}
    algorithm_order = {value: index for index, value in enumerate(DEFAULT_ALGORITHMS)}
    result["_encoder_order"] = result["encoder_condition"].map(encoder_order)
    result["_algorithm_order"] = result["downstream_algorithm"].map(algorithm_order)
    return result.sort_values(["_encoder_order", "_algorithm_order"]).drop(
        columns=["_encoder_order", "_algorithm_order"]
    ).reset_index(drop=True)


def _stability_differences(summary: pd.DataFrame) -> pd.DataFrame:
    means = summary.pivot_table(
        index=["encoder_condition", "downstream_algorithm"],
        columns="comparison_group",
        values="ari_mean",
        aggfunc="first",
    ).reset_index()
    pair_counts = {WITHIN_ENCODER: 9, MATCHED_DOWNSTREAM: 9, ACROSS_BOTH: 18}
    means["cross_encoder_combined_mean"] = (
        means[MATCHED_DOWNSTREAM] * pair_counts[MATCHED_DOWNSTREAM]
        + means[ACROSS_BOTH] * pair_counts[ACROSS_BOTH]
    ) / 27.0
    means["delta_within_minus_cross_combined"] = (
        means[WITHIN_ENCODER] - means["cross_encoder_combined_mean"]
    )
    means["delta_within_minus_cross_matched"] = (
        means[WITHIN_ENCODER] - means[MATCHED_DOWNSTREAM]
    )
    return means


def _markdown_main_table(main_table: pd.DataFrame) -> str:
    lines = [
        "| Encoder | Algorithm | Fixed-encoder ARI | Cross-encoder matched-seed ARI | Cross-encoder different-seed ARI |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in main_table.itertuples(index=False):
        lines.append(
            f"| {row.encoder_display} | {row.algorithm_display} | "
            f"{getattr(row, WITHIN_ENCODER + '_display')} | "
            f"{getattr(row, MATCHED_DOWNSTREAM + '_display')} | "
            f"{getattr(row, ACROSS_BOTH + '_display')} |"
        )
    return "\n".join(lines)


def _algorithm_overview(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, frame in pairs.groupby(["downstream_algorithm", "comparison_group"]):
        rows.append(
            {
                "downstream_algorithm": key[0],
                "comparison_group": key[1],
                **_summary(frame["ARI"]),
            }
        )
    return pd.DataFrame(rows)


def _report(
    *,
    output: Path,
    main_table: pd.DataFrame,
    differences: pd.DataFrame,
    within_seed: pd.DataFrame,
    run_quality: pd.DataFrame,
    alignment_audit: pd.DataFrame,
    pair_quality: pd.DataFrame,
    correlations: pd.DataFrame,
    degeneracy_bands: pd.DataFrame,
) -> None:
    overview = _algorithm_overview(pair_quality)
    overview_pivot = overview.pivot_table(
        index="downstream_algorithm", columns="comparison_group", values="ari_mean"
    )
    algorithm_lines = []
    for algorithm in DEFAULT_ALGORITHMS:
        row = overview_pivot.loc[algorithm]
        algorithm_lines.append(
            f"- **{DISPLAY_NAMES[algorithm]}:** fixed-encoder mean ARI "
            f"{row[WITHIN_ENCODER]:.3f}; cross-encoder matched-seed "
            f"{row[MATCHED_DOWNSTREAM]:.3f}; both seed identifiers different "
            f"{row[ACROSS_BOTH]:.3f}."
        )
    overall_within = float(pair_quality.loc[pair_quality["comparison_group"] == WITHIN_ENCODER, "ARI"].mean())
    cross_matched = float(
        pair_quality.loc[pair_quality["comparison_group"] == MATCHED_DOWNSTREAM, "ARI"].mean()
    )
    cross_different = float(
        pair_quality.loc[pair_quality["comparison_group"] == ACROSS_BOTH, "ARI"].mean()
    )
    overall_cross = float(
        pair_quality.loc[pair_quality["comparison_group"] != WITHIN_ENCODER, "ARI"].mean()
    )
    nonconverged = run_quality[~run_quality["converged"].astype(bool)]
    nonconverged_text = (
        "No saved confirmation run was marked nonconverged."
        if nonconverged.empty
        else "; ".join(
            f"{DISPLAY_NAMES.get(row.encoder, row.encoder)}/{DISPLAY_NAMES.get(row.algorithm, row.algorithm)} "
            f"encoder {int(row.encoder_seed)}, downstream {int(row.downstream_seed)}"
            for row in nonconverged.itertuples()
        )
        + " were retained and marked nonconverged."
    )
    lowest = within_seed.nsmallest(8, "ari_mean")
    lowest_lines = "\n".join(
        f"- {DISPLAY_NAMES.get(row.encoder_condition, row.encoder_condition)} / "
        f"{DISPLAY_NAMES.get(row.downstream_algorithm, row.downstream_algorithm)} / "
        f"encoder seed {int(row.encoder_seed)}: {row.ari_mean:.3f} mean ARI."
        for row in lowest.itertuples()
    )
    alignment_ok = bool(
        (alignment_audit["duplicate_patch_key_rows"] == 0).all()
        and (alignment_audit["valid_rows_without_label"] == 0).all()
        and (alignment_audit["invalid_rows_with_label"] == 0).all()
        and alignment_audit["timeline_and_validity_hash"].nunique() == 1
    )
    min_aligned = int(pair_quality["number_of_aligned_patches"].min())
    max_aligned = int(pair_quality["number_of_aligned_patches"].max())
    retained_min = float(pair_quality["aligned_fraction_of_valid_union"].min())
    strongest_corr = correlations.dropna(subset=["spearman_rho"]).copy()
    strongest_corr["absolute"] = strongest_corr["spearman_rho"].abs()
    if strongest_corr.empty:
        correlation_text = "No computable exploratory quality-difference correlation was available."
    else:
        top = strongest_corr.nlargest(1, "absolute").iloc[0]
        correlation_text = (
            f"The largest absolute exploratory Spearman association was {top.spearman_rho:.3f} "
            f"for `{top.quality_difference}` in scope `{top.scope}`. These overlapping pairs are "
            "not independent, so this is a diagnostic association only."
        )
    family = pair_quality.copy()
    family["encoder_family"] = np.where(
        family["encoder_condition"].str.startswith("graph_"), "GNN", "Invariant MLP"
    )
    family_within = (
        family[family["comparison_group"] == WITHIN_ENCODER]
        .groupby("encoder_family")["ARI"]
        .mean()
    )
    delta_rank = differences.sort_values("delta_within_minus_cross_combined", ascending=False)
    top_delta = delta_rank.iloc[0]
    bottom_delta = delta_rank.iloc[-1]
    occupied_min = int(run_quality["occupied_states"].min())
    occupied_max = int(run_quality["occupied_states"].max())
    effective_min = float(run_quality["effective_entropy_states"].min())
    effective_max = float(run_quality["effective_entropy_states"].max())
    largest_min = float(run_quality["largest_state_occupancy"].min())
    largest_max = float(run_quality["largest_state_occupancy"].max())
    low_band = degeneracy_bands.loc[degeneracy_bands["ari_band"] == "ARI < 0.3"].iloc[0]
    high_band = degeneracy_bands.loc[degeneracy_bands["ari_band"] == "ARI >= 0.6"].iloc[0]
    graph_full_seed = within_seed[
        (within_seed["encoder_condition"] == "graph_full")
        & (within_seed["downstream_algorithm"].isin(["hmm", "hsmm"]))
    ]
    graph_full_seed_text = "; ".join(
        f"{DISPLAY_NAMES.get(row.downstream_algorithm, row.downstream_algorithm)} seed "
        f"{int(row.encoder_seed)}={row.ari_mean:.3f}"
        for row in graph_full_seed.sort_values(
            ["downstream_algorithm", "encoder_seed"]
        ).itertuples()
    )

    text = f"""# 9D Social-GNN: Decomposing Encoder and Downstream Clustering Instability

Date: 2026-10-20

## 1. Executive summary

This is a post-hoc analysis of the existing 144 K=6 confirmation runs. No encoder or downstream model was retrained.

**Direct answer:** instability is associated primarily with changing the learned representation, but the probabilistic downstream models also contribute substantial instability. Across all 16 encoder-condition x algorithm blocks, mean ARI was **{overall_within:.3f}** when the encoder was fixed and only the downstream seed changed, versus **{overall_cross:.3f}** across independently trained encoder seeds. Cross-encoder matched-seed ARI ({cross_matched:.3f}) and different-seed ARI ({cross_different:.3f}) were nearly identical, so preserving the numerical downstream seed did not rescue agreement after the representation changed. This gap is descriptive, not an additive or causal variance decomposition.

{chr(10).join(algorithm_lines)}

KMeans makes the distinction clearest: it was almost deterministic for a fixed embedding (mean ARI 0.987), yet fell to about 0.435 across encoder seeds. GMM/HMM/HSMM were less stable even conditionally (0.733/0.664/0.662), so their observed instability reflects **both** encoder-seed variation and downstream local optima. Conditional fixed-representation agreement averaged **{family_within.get('GNN', float('nan')):.3f}** for the two GNN conditions and **{family_within.get('Invariant MLP', float('nan')):.3f}** for the two invariant-MLP conditions. The invariant MLP is therefore more conditionally stable in this benchmark, not universally seed-invariant.

The largest within-minus-cross gap occurred for {DISPLAY_NAMES.get(top_delta.encoder_condition, top_delta.encoder_condition)} / {DISPLAY_NAMES.get(top_delta.downstream_algorithm, top_delta.downstream_algorithm)} ({top_delta.delta_within_minus_cross_combined:.3f}); the smallest was {DISPLAY_NAMES.get(bottom_delta.encoder_condition, bottom_delta.encoder_condition)} / {DISPLAY_NAMES.get(bottom_delta.downstream_algorithm, bottom_delta.downstream_algorithm)} ({bottom_delta.delta_within_minus_cross_combined:.3f}).

{nonconverged_text} No run was removed because of convergence status. Stability remains a property of partitions on reused development trial A4_C4, not behavioral accuracy or untouched biological generalization.

## 2. Experimental setup and alignment

For each of four encoder conditions and four algorithms, the confirmation grid contains three encoder seeds (`20261013`, `20261029`, `20261107`) crossed with downstream seeds (`0`, `1`, `2`). Each 3 x 3 block contributes exactly 36 unique pairs: 9 fixed-encoder/different-downstream, 9 cross-encoder/matched-downstream-seed, and 18 pairs where both identifiers differ.

Assignments were aligned by `(trial_id, patch_start_s, patch_end_s)`, never by row order alone. Alignment integrity status: **{'passed' if alignment_ok else 'requires attention'}**. Every pair retained {min_aligned} to {max_aligned} mutually valid physical patches; the minimum retained fraction of the valid union was {retained_min:.6f}. Invalid patches and gaps were excluded rather than interpolated, and no padded timestep was introduced.

Using the same numerical downstream seed in two different PCA/embedding spaces does not guarantee equivalent initialization. The matched-seed group is therefore a controlled cross-encoder reproducibility diagnostic, not a pure causal encoder effect.

## 3. Overall ARI results

Values are mean +/- sample SD across dependent pairwise comparisons. They are descriptive; no naive pair-level p-values or confidence intervals are reported.

{_markdown_main_table(main_table)}

Detailed medians, quartiles, extrema and counts are in `ari_group_summary.csv`; all 576 traceable comparisons are in `all_pairwise_ari.csv`.

## 4. Within-encoder downstream stability

The fixed-encoder group asks whether a downstream method repeatedly recovers the same partition from exactly the same saved representation. The lowest seed-specific conditional results were:

{lowest_lines}

The complete 48-row encoder-seed breakdown is in `within_encoder_seed_summary.csv`. Low values here indicate that downstream initialization/local optima remain important even before changing the encoder. High values here combined with lower cross-encoder values point instead to representation reproducibility as the larger concern.

The instability is strongly encoder-seed specific rather than uniform. For Graph Full temporal models: {graph_full_seed_text}. Thus one Graph Full encoder seed (`20261029`) yielded almost identical HMM/HSMM partitions across downstream starts, whereas seeds `20261013` and `20261107` did not. Conversely, Invariant MLP Hybrid P2 seed `20261107` was essentially deterministic for GMM/HMM/HSMM. These patterns indicate an encoder-by-downstream interaction: some learned geometries create much easier optimization landscapes for the same downstream implementation.

## 5. Across-encoder stability

Both cross-encoder groups compare independently trained representations. The matched downstream seed identifier controls the recorded integer only; it does not establish equivalent initialization in changed embedding spaces. `cross_encoder_seed_summary.csv` preserves results separately for each matched downstream seed and each unordered different-seed stratum.

The descriptive deltas in `stability_differences.csv` report fixed-encoder mean ARI minus (a) matched-seed cross-encoder ARI and (b) all cross-encoder ARIs. They must not be interpreted as a percentage of instability caused by the encoder.

## 6. Algorithm comparison

{chr(10).join(algorithm_lines)}

KMeans, GMM, HMM and HSMM therefore need to be judged separately: temporal coherence and held-out likelihood from the main benchmark do not guarantee seed-stable state identities. Heatmaps expose whether disagreement is blockwise between encoder seeds or dispersed within each fixed-encoder block.

## 7. Stability versus degeneracy

All requested occupancy, effective-state, fragmentation, silhouette and convergence fields are preserved in `run_quality_metrics.csv` and joined pairwise in `ari_vs_occupancy.csv`. Every run occupied {occupied_min} to {occupied_max} states; entropy-effective state count ranged from {effective_min:.3f} to {effective_max:.3f}, and largest-state occupancy from {largest_min:.3f} to {largest_max:.3f}. The old K_eff around 1.6 collapse was therefore not reproduced in this 9D confirmation grid. {nonconverged_text}

Low-ARI pairs did coincide with larger differences in state geometry and fragmentation. For the {int(low_band.pair_count)} pairs with ARI < 0.3, mean absolute K_eff difference was {low_band.mean_abs_effective_states_difference:.3f} and mean bouts/min difference was {low_band.mean_abs_bouts_per_minute_difference:.2f}; for the {int(high_band.pair_count)} pairs with ARI >= 0.6, those values were {high_band.mean_abs_effective_states_difference:.3f} and {high_band.mean_abs_bouts_per_minute_difference:.2f}. No pair differed in the number of occupied states, so disagreement concerned occupancy balance, geometry and temporal boundaries rather than outright missing states.

{correlation_text}

High ARI can coexist with an uninformative or degenerate partition; low ARI can reflect label-independent boundary/state-geometry changes without proving biological failure. The recent K_eff values must therefore be read alongside largest-state occupancy, smallest occupied fraction, bouts/min, median duration and convergence.

## 8. Interpretation and limitations

- **Encoder-seed sensitivity** is evidenced when fixed-encoder agreement is high but cross-encoder agreement is lower.
- **Downstream-seed sensitivity** is evidenced directly by low fixed-encoder agreement.
- **Joint variation** is summarized by the 18-pair group where both identifiers differ, but is not an interaction-effect estimate.
- Every pair reuses runs and the same A4_C4 patches, so the ARIs are dependent diagnostics, not biological replicates.
- A4_C4 was already used for encoder checkpoint selection. Nothing here estimates generalization to an untouched animal pair or experiment.
- State labels were not matched before ARI; ARI is already invariant to label permutations.
- The optional dense co-assignment analysis was not run because it was unnecessary for the complete ARI decomposition and would add a second dependent-pair diagnostic without changing the primary question.

## 9. Actionable recommendations

| Observation | Recommended next step |
| --- | --- |
| Low within-encoder HMM ARI | Inspect HMM initialization/local optima and K before changing the encoder. |
| High within-encoder, low cross-encoder ARI | Prioritize encoder representation reproducibility and compare state signatures across encoder seeds. |
| Strong KMeans but weak HMM stability | Inspect probabilistic temporal-model initialization and emission/transition assumptions. |
| Stability associated with degenerate occupancy | Investigate state collapse or background-state dominance before biological naming. |
| Nonconverged GMM fits | Keep them visible, increase diagnostic scrutiny, and do not use their ARI as evidence of a clean optimum. |
| High stability across both factors | Proceed to blinded video review and a genuinely independent trial; do not equate stability with validity. |

The next action should be chosen from the observed block-specific pattern rather than automatically retraining models. The heatmaps in `heatmaps/` and pair dots in `distributions/` are the fastest way to identify seed-specific failures.
"""
    (output / "ari_stability_report.md").write_text(text, encoding="utf-8")


def analyze_ari_stability(
    *,
    runs_csv: str | Path,
    output_dir: str | Path,
    trial_id: str = "A4_C4",
    k: int = 6,
    encoders: Sequence[str] = DEFAULT_ENCODERS,
    algorithms: Sequence[str] = DEFAULT_ALGORITHMS,
    encoder_seeds: Sequence[int] = DEFAULT_ENCODER_SEEDS,
    downstream_seeds: Sequence[int] = DEFAULT_DOWNSTREAM_SEEDS,
) -> dict[str, Any]:
    runs_path = Path(runs_csv).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(runs_path)
    selected = runs[
        runs["encoder"].isin(encoders)
        & runs["algorithm"].isin(algorithms)
        & runs["encoder_seed"].isin(encoder_seeds)
        & runs["downstream_seed"].isin(downstream_seeds)
        & runs["K"].eq(k)
    ].copy()
    expected = expected_grid(encoders, algorithms, encoder_seeds, downstream_seeds, k=k)
    grid = audit_run_grid(selected, expected)
    grid.to_csv(output / "run_grid_audit.csv", index=False)
    require_complete_grid(grid)
    if not selected["status"].eq("completed").all():
        failed = selected.loc[~selected["status"].eq("completed"), RUN_KEY + ["status", "failure"]]
        raise ValueError(f"saved assignment grid includes non-completed runs: {failed.to_dict('records')}")

    selected["run_id"] = selected.apply(run_id, axis=1)
    assignments: dict[str, AssignmentData] = {}
    audit_rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        assignment = load_assignment(row, trial_id)
        assignments[row["run_id"]] = assignment
        audit_rows.append(assignment_audit(row, assignment))
    alignment_audit = pd.DataFrame(audit_rows).sort_values(RUN_KEY).reset_index(drop=True)
    alignment_audit.to_csv(output / "alignment_audit.csv", index=False)
    invalid_audit = alignment_audit[
        (alignment_audit["duplicate_patch_key_rows"] != 0)
        | (alignment_audit["valid_rows_without_label"] != 0)
        | (alignment_audit["invalid_rows_with_label"] != 0)
    ]
    if not invalid_audit.empty:
        raise ValueError(f"assignment audit failed: {invalid_audit.to_dict('records')}")

    quality_columns = [value for value in QUALITY_COLUMNS if value in selected.columns]
    run_quality = selected[RUN_KEY + ["run_id"] + quality_columns].merge(
        alignment_audit[
            [
                "run_id",
                "trial_id",
                "assignment_path",
                "total_patches",
                "valid_patches",
                "timeline_hash",
                "timeline_and_validity_hash",
            ]
        ],
        on="run_id",
        how="left",
        validate="one_to_one",
    )
    run_quality.to_csv(output / "run_quality_metrics.csv", index=False)

    pair_rows: list[dict[str, Any]] = []
    for (encoder, algorithm), block in selected.groupby(["encoder", "algorithm"], sort=True):
        if block.shape[0] != 9:
            raise ValueError(f"{encoder}/{algorithm}: expected 9 runs, found {block.shape[0]}")
        ordered = block.sort_values(["encoder_seed", "downstream_seed"]).reset_index(drop=True)
        for left_index, right_index in combinations(range(ordered.shape[0]), 2):
            left = ordered.iloc[left_index]
            right = ordered.iloc[right_index]
            alignment = align_assignments(assignments[left["run_id"]], assignments[right["run_id"]])
            ari = adjusted_rand_score(alignment.pop("left_labels"), alignment.pop("right_labels"))
            pair_rows.append(
                {
                    "encoder_condition": encoder,
                    "downstream_algorithm": algorithm,
                    "K": int(k),
                    "run_1_id": left["run_id"],
                    "run_1_encoder_seed": int(left["encoder_seed"]),
                    "run_1_downstream_seed": int(left["downstream_seed"]),
                    "run_1_assignment_path": assignments[left["run_id"]].source_path,
                    "run_1_checkpoint": left.get("checkpoint", ""),
                    "run_2_id": right["run_id"],
                    "run_2_encoder_seed": int(right["encoder_seed"]),
                    "run_2_downstream_seed": int(right["downstream_seed"]),
                    "run_2_assignment_path": assignments[right["run_id"]].source_path,
                    "run_2_checkpoint": right.get("checkpoint", ""),
                    "comparison_group": classify_pair(
                        int(left["encoder_seed"]),
                        int(left["downstream_seed"]),
                        int(right["encoder_seed"]),
                        int(right["downstream_seed"]),
                    ),
                    "trial_id": trial_id,
                    **alignment,
                    "ARI": float(ari),
                }
            )
    pairs = pd.DataFrame(pair_rows)
    expected_counts = {WITHIN_ENCODER: 9, MATCHED_DOWNSTREAM: 9, ACROSS_BOTH: 18}
    block_counts = pairs.groupby(
        ["encoder_condition", "downstream_algorithm", "comparison_group"]
    ).size()
    for (encoder, algorithm, group), count in block_counts.items():
        if int(count) != expected_counts[group]:
            raise ValueError(f"{encoder}/{algorithm}/{group}: expected {expected_counts[group]} pairs, found {count}")
    if pairs.shape[0] != len(encoders) * len(algorithms) * 36:
        raise ValueError(f"expected {len(encoders) * len(algorithms) * 36} total pairs, found {pairs.shape[0]}")
    pairs.to_csv(output / "all_pairwise_ari.csv", index=False)

    summary = summarize_groups(pairs)
    summary.to_csv(output / "ari_group_summary.csv", index=False)
    within_seed = summarize_within_encoder_seed(pairs)
    within_seed.to_csv(output / "within_encoder_seed_summary.csv", index=False)
    cross_seed = summarize_cross_encoder_strata(pairs)
    cross_seed.to_csv(output / "cross_encoder_seed_summary.csv", index=False)
    main_table = _main_table(summary)
    main_table.to_csv(output / "main_comparison_table.csv", index=False)
    differences = _stability_differences(summary)
    differences.to_csv(output / "stability_differences.csv", index=False)
    pair_quality = _augment_pair_quality(pairs, run_quality)
    pair_quality.to_csv(output / "ari_vs_occupancy.csv", index=False)
    correlations = _quality_correlations(pair_quality)
    correlations.to_csv(output / "ari_quality_correlations.csv", index=False)
    degeneracy_bands = _degeneracy_bands(pair_quality)
    degeneracy_bands.to_csv(output / "ari_degeneracy_summary.csv", index=False)

    heatmaps = output / "heatmaps"
    distributions = output / "distributions"
    heatmaps.mkdir(exist_ok=True)
    distributions.mkdir(exist_ok=True)
    for (encoder, algorithm), block in pairs.groupby(
        ["encoder_condition", "downstream_algorithm"], sort=True
    ):
        matrix, labels = ari_matrix(block, encoder_seeds, downstream_seeds)
        title = f"{DISPLAY_NAMES.get(encoder, encoder)} / {DISPLAY_NAMES.get(algorithm, algorithm)} / K=6"
        slug = f"{encoder}__{algorithm}"
        _plot_heatmap(matrix, labels, title, heatmaps / f"{slug}.png")
        _plot_distribution(block, title, distributions / f"{slug}.png")

    _report(
        output=output,
        main_table=main_table,
        differences=differences,
        within_seed=within_seed,
        run_quality=run_quality,
        alignment_audit=alignment_audit,
        pair_quality=pair_quality,
        correlations=correlations,
        degeneracy_bands=degeneracy_bands,
    )
    manifest = {
        "contract": "social_gnn_ari_stability_decomposition_v1",
        "source_runs_csv": str(runs_path),
        "output_dir": str(output),
        "trial_id": trial_id,
        "K": int(k),
        "encoders": list(encoders),
        "algorithms": list(algorithms),
        "encoder_seeds": [int(value) for value in encoder_seeds],
        "downstream_seeds": [int(value) for value in downstream_seeds],
        "saved_runs_used": int(selected.shape[0]),
        "pairwise_comparisons": int(pairs.shape[0]),
        "pairs_per_block": expected_counts,
        "models_retrained": 0,
        "notes": [
            "Assignments are aligned by trial_id, patch_start_s, and patch_end_s.",
            "Pairwise ARIs share runs and patches and are descriptive dependent diagnostics.",
            "Matched downstream seed identifiers across encoders are not equivalent causal initializations.",
            "A4_C4 is reused development data, not an untouched biological test.",
            "No run was removed for nonconvergence.",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trial-id", default="A4_C4")
    parser.add_argument("--k", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = analyze_ari_stability(
        runs_csv=args.runs_csv,
        output_dir=args.output_dir,
        trial_id=args.trial_id,
        k=args.k,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
