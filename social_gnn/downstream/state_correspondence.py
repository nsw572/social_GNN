"""State-level correspondence audit for saved Social-GNN HMM assignments.

The analysis in this module is deliberately post-hoc: it never fits an
encoder or a downstream model.  Run-local state labels are preserved.  A
Hungarian permutation is used only as a diagnostic summary and is accompanied
by the complete overlap matrix and match-quality metrics.
"""

from __future__ import annotations

import json
import math
import hashlib
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .ari_stability import (
    DEFAULT_DOWNSTREAM_SEEDS,
    DEFAULT_ENCODER_SEEDS,
    DEFAULT_ENCODERS,
    AssignmentData,
    load_assignment,
    run_id,
)


SAME_CHECKPOINT = "same_encoder_checkpoint_different_downstream_seed"
ACROSS_ENCODER = "different_encoder_seed_same_condition"
GRAPH_VS_MLP = "graph_full_vs_invariant_mlp_full"

DISPLAY_NAMES = {
    "graph_full": "Graph Full",
    "graph_hybrid_p2": "Graph Hybrid P2",
    "invariant_mlp_full": "Invariant MLP Full",
    "invariant_mlp_hybrid_p2": "Invariant MLP Hybrid P2",
}


@dataclass(frozen=True)
class CorrespondenceThresholds:
    """Transparent heuristic thresholds used only for interpretation flags."""

    component_fraction: float = 0.20
    ambiguity_margin: float = 0.10
    weak_f1: float = 0.40
    stable_f1: float = 0.70
    defensible_jaccard: float = 0.30

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")


def contingency_matrix(
    left_labels: np.ndarray, right_labels: np.ndarray, k: int
) -> np.ndarray:
    """Count co-assignments without assuming that numeric labels correspond."""

    left = np.asarray(left_labels, dtype=np.int64)
    right = np.asarray(right_labels, dtype=np.int64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("state arrays must be equal-length one-dimensional arrays")
    if left.size == 0:
        raise ValueError("state arrays must not be empty")
    if np.any((left < 0) | (left >= k)) or np.any((right < 0) | (right >= k)):
        raise ValueError(f"labels must be in [0, {k - 1}]")
    matrix = np.zeros((k, k), dtype=np.int64)
    np.add.at(matrix, (left, right), 1)
    return matrix


def overlap_metrics(matrix: np.ndarray) -> pd.DataFrame:
    """Return all directed/symmetric overlap measures for every matrix cell."""

    counts = np.asarray(matrix, dtype=np.int64)
    if counts.ndim != 2:
        raise ValueError("contingency matrix must be two-dimensional")
    left_total = counts.sum(axis=1)
    right_total = counts.sum(axis=0)
    rows: list[dict[str, Any]] = []
    for left_state in range(counts.shape[0]):
        for right_state in range(counts.shape[1]):
            intersection = int(counts[left_state, right_state])
            left_n = int(left_total[left_state])
            right_n = int(right_total[right_state])
            recall_left = intersection / left_n if left_n else 0.0
            precision_left = intersection / right_n if right_n else 0.0
            denominator = left_n + right_n - intersection
            jaccard = intersection / denominator if denominator else 0.0
            f1 = (
                2.0 * precision_left * recall_left / (precision_left + recall_left)
                if precision_left + recall_left
                else 0.0
            )
            rows.append(
                {
                    "left_state": left_state,
                    "right_state": right_state,
                    "intersection_patches": intersection,
                    "left_state_patches": left_n,
                    "right_state_patches": right_n,
                    "p_right_given_left": recall_left,
                    "p_left_given_right": precision_left,
                    "recall_of_left_state": recall_left,
                    "precision_of_right_state_for_left": precision_left,
                    "jaccard": jaccard,
                    "f1": f1,
                    "left_occupancy": left_n / max(int(counts.sum()), 1),
                    "right_occupancy": right_n / max(int(counts.sum()), 1),
                }
            )
    return pd.DataFrame(rows)


def hungarian_state_matches(
    matrix: np.ndarray,
    *,
    thresholds: CorrespondenceThresholds | None = None,
) -> pd.DataFrame:
    """Maximize matched patch count and expose quality for each forced match."""

    thresholds = thresholds or CorrespondenceThresholds()
    counts = np.asarray(matrix, dtype=np.int64)
    metrics = overlap_metrics(counts).set_index(["left_state", "right_state"])
    left_indices, right_indices = linear_sum_assignment(-counts.astype(np.float64))
    left_best = counts.argmax(axis=1)
    right_best = counts.argmax(axis=0)
    rows: list[dict[str, Any]] = []
    for left_state, right_state in zip(left_indices, right_indices):
        values = metrics.loc[(int(left_state), int(right_state))].to_dict()
        reciprocal = bool(
            left_best[left_state] == right_state and right_best[right_state] == left_state
        )
        f1 = float(values["f1"])
        jaccard = float(values["jaccard"])
        if reciprocal and f1 >= thresholds.stable_f1:
            status = "stable_reciprocal"
        elif f1 < thresholds.weak_f1:
            status = "weak_forced_match"
        elif reciprocal:
            status = "partial_reciprocal"
        else:
            status = "partial_nonreciprocal"
        rows.append(
            {
                "left_state": int(left_state),
                "right_state": int(right_state),
                **values,
                "reciprocal_best": reciprocal,
                "match_status": status,
                "defensible_for_consensus": bool(
                    f1 >= thresholds.weak_f1
                    and (reciprocal or jaccard >= thresholds.defensible_jaccard)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("left_state").reset_index(drop=True)


def split_merge_diagnostics(
    matrix: np.ndarray,
    *,
    thresholds: CorrespondenceThresholds | None = None,
) -> pd.DataFrame:
    """Flag one-to-many split and many-to-one merge patterns transparently."""

    thresholds = thresholds or CorrespondenceThresholds()
    counts = np.asarray(matrix, dtype=np.int64)
    rows: list[dict[str, Any]] = []

    def summarize(fractions: np.ndarray, state: int, direction: str) -> dict[str, Any]:
        ordering = np.argsort(fractions)[::-1]
        best = float(fractions[ordering[0]]) if ordering.size else 0.0
        second = float(fractions[ordering[1]]) if ordering.size > 1 else 0.0
        components = np.flatnonzero(fractions >= thresholds.component_fraction)
        return {
            "direction": direction,
            "source_state": int(state),
            "best_target_state": int(ordering[0]) if ordering.size else -1,
            "second_target_state": int(ordering[1]) if ordering.size > 1 else -1,
            "best_overlap_fraction": best,
            "second_overlap_fraction": second,
            "ambiguity_margin": best - second,
            "ambiguous": bool(best - second < thresholds.ambiguity_margin),
            "substantial_target_count": int(components.size),
            "substantial_targets": json.dumps([int(value) for value in components]),
            "split_candidate": bool(direction == "left_to_right" and components.size >= 2),
            "merge_candidate": bool(direction == "right_to_left" and components.size >= 2),
        }

    row_total = counts.sum(axis=1, keepdims=True)
    row_fraction = np.divide(
        counts, row_total, out=np.zeros_like(counts, dtype=float), where=row_total > 0
    )
    for state in range(counts.shape[0]):
        rows.append(summarize(row_fraction[state], state, "left_to_right"))

    column_total = counts.sum(axis=0, keepdims=True)
    column_fraction = np.divide(
        counts,
        column_total,
        out=np.zeros_like(counts, dtype=float),
        where=column_total > 0,
    )
    for state in range(counts.shape[1]):
        rows.append(summarize(column_fraction[:, state], state, "right_to_left"))
    return pd.DataFrame(rows)


def boundary_agreement(
    left_labels: np.ndarray,
    right_labels: np.ndarray,
    valid_mask: np.ndarray,
    patch_start_s: np.ndarray,
    patch_end_s: np.ndarray,
    *,
    tolerance_patches: int = 1,
) -> dict[str, float | int]:
    """Compare state-change boundaries without crossing invalid clock gaps."""

    left = np.asarray(left_labels, dtype=np.int64)
    right = np.asarray(right_labels, dtype=np.int64)
    valid = np.asarray(valid_mask, dtype=bool)
    start = np.asarray(patch_start_s, dtype=np.float64)
    end = np.asarray(patch_end_s, dtype=np.float64)
    if not (left.shape == right.shape == valid.shape == start.shape == end.shape):
        raise ValueError("boundary inputs must have identical shapes")
    duration = np.median(end - start)
    continuous = (
        valid[:-1]
        & valid[1:]
        & (left[:-1] >= 0)
        & (left[1:] >= 0)
        & (right[:-1] >= 0)
        & (right[1:] >= 0)
        & (np.abs(start[1:] - end[:-1]) <= max(1e-8, abs(duration) * 1e-4))
    )
    left_boundary = continuous & (left[:-1] != left[1:])
    right_boundary = continuous & (right[:-1] != right[1:])
    exact = int(np.sum(left_boundary & right_boundary))
    left_n = int(left_boundary.sum())
    right_n = int(right_boundary.sum())
    precision = exact / right_n if right_n else float(left_n == 0)
    recall = exact / left_n if left_n else float(right_n == 0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = int(np.sum(left_boundary | right_boundary))

    left_positions = np.flatnonzero(left_boundary)
    right_positions = np.flatnonzero(right_boundary)
    used: set[int] = set()
    tolerant_matches = 0
    for position in left_positions:
        candidates = [
            int(value)
            for value in right_positions
            if int(value) not in used and abs(int(value) - int(position)) <= tolerance_patches
        ]
        if candidates:
            selected = min(candidates, key=lambda value: (abs(value - int(position)), value))
            used.add(selected)
            tolerant_matches += 1
    tolerant_precision = tolerant_matches / right_n if right_n else float(left_n == 0)
    tolerant_recall = tolerant_matches / left_n if left_n else float(right_n == 0)
    tolerant_f1 = (
        2 * tolerant_precision * tolerant_recall / (tolerant_precision + tolerant_recall)
        if tolerant_precision + tolerant_recall
        else 0.0
    )
    return {
        "eligible_adjacent_patch_pairs": int(continuous.sum()),
        "left_boundary_count": left_n,
        "right_boundary_count": right_n,
        "exact_boundary_matches": exact,
        "boundary_precision_exact": precision,
        "boundary_recall_exact": recall,
        "boundary_f1_exact": f1,
        "boundary_jaccard_exact": exact / union if union else 1.0,
        "boundary_tolerance_patches": int(tolerance_patches),
        "tolerant_boundary_matches": tolerant_matches,
        "boundary_f1_tolerant": tolerant_f1,
    }


def _pair_id(left_id: str, right_id: str, scope: str) -> str:
    clean = lambda value: value.replace("|", "_").replace(":", "_")
    return f"{scope}__{clean(left_id)}__vs__{clean(right_id)}"


def _artifact_slug(pair_id: str) -> str:
    scope = pair_id.split("__", 1)[0]
    prefix = {
        SAME_CHECKPOINT: "fixed",
        ACROSS_ENCODER: "encoder",
        GRAPH_VS_MLP: "graph_mlp",
    }.get(scope, "pair")
    return f"{prefix}_{hashlib.sha1(pair_id.encode('utf-8')).hexdigest()[:16]}"


def _enumerate_pairs(selected: pd.DataFrame) -> list[tuple[str, pd.Series, pd.Series]]:
    pairs: list[tuple[str, pd.Series, pd.Series]] = []
    for _, block in selected.groupby(["encoder", "encoder_seed"], sort=True):
        ordered = block.sort_values("downstream_seed")
        for left_i, right_i in combinations(range(len(ordered)), 2):
            pairs.append((SAME_CHECKPOINT, ordered.iloc[left_i], ordered.iloc[right_i]))
    for _, block in selected.groupby("encoder", sort=True):
        ordered = block.sort_values(["encoder_seed", "downstream_seed"])
        for left_i, right_i in combinations(range(len(ordered)), 2):
            left, right = ordered.iloc[left_i], ordered.iloc[right_i]
            if int(left.encoder_seed) != int(right.encoder_seed):
                pairs.append((ACROSS_ENCODER, left, right))
    graph = selected[selected["encoder"] == "graph_full"].sort_values(
        ["encoder_seed", "downstream_seed"]
    )
    mlp = selected[selected["encoder"] == "invariant_mlp_full"].sort_values(
        ["encoder_seed", "downstream_seed"]
    )
    for (_, left), (_, right) in product(graph.iterrows(), mlp.iterrows()):
        pairs.append((GRAPH_VS_MLP, left, right))
    return pairs


def _assignment_arrays_equal(left: AssignmentData, right: AssignmentData) -> bool:
    return bool(
        left.trial_id == right.trial_id
        and np.array_equal(left.patch_start_s, right.patch_start_s)
        and np.array_equal(left.patch_end_s, right.patch_end_s)
        and np.array_equal(left.valid_time_mask, right.valid_time_mask)
    )


def _annotate_pair(
    frame: pd.DataFrame,
    *,
    pair_id: str,
    scope: str,
    left: pd.Series,
    right: pd.Series,
    left_assignment: AssignmentData,
    right_assignment: AssignmentData,
) -> pd.DataFrame:
    result = frame.copy()
    metadata = {
        "pair_id": pair_id,
        "comparison_scope": scope,
        "trial_id": left_assignment.trial_id,
        "left_run_id": left.run_id,
        "left_encoder": left.encoder,
        "left_encoder_seed": int(left.encoder_seed),
        "left_downstream_seed": int(left.downstream_seed),
        "left_assignment_path": left_assignment.source_path,
        "right_run_id": right.run_id,
        "right_encoder": right.encoder,
        "right_encoder_seed": int(right.encoder_seed),
        "right_downstream_seed": int(right.downstream_seed),
        "right_assignment_path": right_assignment.source_path,
    }
    for key, value in reversed(list(metadata.items())):
        result.insert(0, key, value)
    return result


def _plot_matrix(matrix: np.ndarray, title: str, path: Path, mode: str) -> None:
    values = np.asarray(matrix, dtype=float)
    if mode == "row":
        denominator = values.sum(axis=1, keepdims=True)
        values = np.divide(values, denominator, out=np.zeros_like(values), where=denominator > 0)
        color_label = "P(right state | left state)"
        vmax = 1.0
    elif mode == "column":
        denominator = values.sum(axis=0, keepdims=True)
        values = np.divide(values, denominator, out=np.zeros_like(values), where=denominator > 0)
        color_label = "P(left state | right state)"
        vmax = 1.0
    elif mode == "count":
        color_label = "Patch count"
        vmax = max(float(values.max()), 1.0)
    else:
        raise ValueError(mode)
    figure, axis = plt.subplots(figsize=(6.5, 5.6))
    image = axis.imshow(values, cmap="viridis", vmin=0.0, vmax=vmax)
    axis.set_xlabel("Right run-local state")
    axis.set_ylabel("Left run-local state")
    axis.set_xticks(range(values.shape[1]))
    axis.set_yticks(range(values.shape[0]))
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            label = f"{int(values[i, j])}" if mode == "count" else f"{values[i, j]:.2f}"
            axis.text(j, i, label, ha="center", va="center", fontsize=7,
                      color="white" if values[i, j] < 0.45 * vmax else "black")
    axis.set_title(title, fontsize=10)
    figure.colorbar(image, ax=axis, label=color_label)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _select_plot_pairs(pair_summary: pd.DataFrame) -> set[str]:
    """Deterministic visualization subset selected without using outcomes."""

    selected: set[str] = set()
    same = pair_summary[
        (pair_summary["comparison_scope"] == SAME_CHECKPOINT)
        & (pair_summary["left_downstream_seed"] == 0)
        & (pair_summary["right_downstream_seed"] == 1)
    ]
    selected.update(same["pair_id"])
    cross = pair_summary[
        (pair_summary["comparison_scope"] == ACROSS_ENCODER)
        & (pair_summary["left_downstream_seed"] == 0)
        & (pair_summary["right_downstream_seed"] == 0)
    ]
    selected.update(cross["pair_id"])
    architecture = pair_summary[
        (pair_summary["comparison_scope"] == GRAPH_VS_MLP)
        & (pair_summary["left_encoder_seed"] == pair_summary["right_encoder_seed"])
        & (pair_summary["left_downstream_seed"] == 0)
        & (pair_summary["right_downstream_seed"] == 0)
    ]
    selected.update(architecture["pair_id"])
    return selected


def _per_state_stability(
    overlap: pd.DataFrame,
    matches: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    observations: list[dict[str, Any]] = []
    diagnostic_lookup = diagnostics.set_index(["pair_id", "direction", "source_state"])
    for pair_id, pair_matches in matches.groupby("pair_id", sort=False):
        pair_overlap = overlap[overlap["pair_id"] == pair_id]
        meta = pair_matches.iloc[0]
        for side in ("left", "right"):
            for state in range(6):
                if side == "left":
                    candidates = pair_overlap[pair_overlap["left_state"] == state]
                    best = candidates.sort_values(
                        ["f1", "jaccard", "right_state"], ascending=[False, False, True]
                    ).iloc[0]
                    match = pair_matches[pair_matches["left_state"] == state].iloc[0]
                    diag_key = (pair_id, "left_to_right", state)
                    run_id_value = meta.left_run_id
                    encoder = meta.left_encoder
                    encoder_seed = int(meta.left_encoder_seed)
                    downstream_seed = int(meta.left_downstream_seed)
                    source_assignment_path = meta.left_assignment_path
                    counterpart = int(best.right_state)
                    occupancy = float(best.left_occupancy)
                    event = "split"
                else:
                    candidates = pair_overlap[pair_overlap["right_state"] == state]
                    best = candidates.sort_values(
                        ["f1", "jaccard", "left_state"], ascending=[False, False, True]
                    ).iloc[0]
                    match = pair_matches[pair_matches["right_state"] == state].iloc[0]
                    diag_key = (pair_id, "right_to_left", state)
                    run_id_value = meta.right_run_id
                    encoder = meta.right_encoder
                    encoder_seed = int(meta.right_encoder_seed)
                    downstream_seed = int(meta.right_downstream_seed)
                    source_assignment_path = meta.right_assignment_path
                    counterpart = int(best.left_state)
                    occupancy = float(best.right_occupancy)
                    event = "merge"
                diag = diagnostic_lookup.loc[diag_key]
                observations.append(
                    {
                        "run_id": run_id_value,
                        "trial_id": meta.trial_id,
                        "encoder": encoder,
                        "encoder_seed": encoder_seed,
                        "downstream_seed": downstream_seed,
                        "state": state,
                        "source_assignment_path": source_assignment_path,
                        "comparison_scope": meta.comparison_scope,
                        "pair_id": pair_id,
                        "best_counterpart_state": counterpart,
                        "best_jaccard": float(best.jaccard),
                        "best_f1": float(best.f1),
                        "hungarian_jaccard": float(match.jaccard),
                        "hungarian_f1": float(match.f1),
                        "hungarian_reciprocal_best": bool(match.reciprocal_best),
                        "ambiguous": bool(diag.ambiguous),
                        "split_or_merge": bool(diag[f"{event}_candidate"]),
                        "occupancy": occupancy,
                    }
                )
    detail = pd.DataFrame(observations)
    rows: list[dict[str, Any]] = []
    for key, group in detail.groupby(
        [
            "run_id",
            "trial_id",
            "encoder",
            "encoder_seed",
            "downstream_seed",
            "state",
            "comparison_scope",
        ],
        sort=True,
    ):
        rows.append(
            {
                "run_id": key[0],
                "trial_id": key[1],
                "encoder": key[2],
                "encoder_seed": int(key[3]),
                "downstream_seed": int(key[4]),
                "state": int(key[5]),
                "comparison_scope": key[6],
                "source_assignment_path": str(group.source_assignment_path.iloc[0]),
                "comparison_count": int(group.shape[0]),
                "mean_best_match_jaccard": float(group.best_jaccard.mean()),
                "mean_best_match_f1": float(group.best_f1.mean()),
                "mean_hungarian_jaccard": float(group.hungarian_jaccard.mean()),
                "mean_hungarian_f1": float(group.hungarian_f1.mean()),
                "reciprocal_best_fraction": float(group.hungarian_reciprocal_best.mean()),
                "ambiguity_fraction": float(group.ambiguous.mean()),
                "split_or_merge_fraction": float(group.split_or_merge.mean()),
                "mean_occupancy": float(group.occupancy.mean()),
                "occupancy_sd": float(group.occupancy.std(ddof=1)) if len(group) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def within_encoder_consensus(
    block: pd.DataFrame,
    assignments: dict[str, AssignmentData],
    *,
    k: int,
    thresholds: CorrespondenceThresholds,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a confidence-qualified three-seed diagnostic consensus.

    Downstream seed 0 is always the reference.  This deterministic rule is
    chosen before inspecting correspondence quality and does not change any
    saved HMM assignment.
    """

    ordered = block.sort_values("downstream_seed")
    if ordered["downstream_seed"].tolist() != [0, 1, 2]:
        raise ValueError("consensus requires downstream seeds 0, 1, and 2")
    reference = ordered.iloc[0]
    reference_assignment = assignments[reference.run_id]
    if not all(
        _assignment_arrays_equal(reference_assignment, assignments[row.run_id])
        for row in ordered.iloc[1:].itertuples()
    ):
        raise ValueError("consensus assignments do not share an identical physical timeline")

    mapped = np.full((3, reference_assignment.size), -1, dtype=np.int64)
    mapping_ok = np.zeros((3, k), dtype=bool)
    mapped[0] = reference_assignment.predicted_state
    mapping_ok[0, :] = True
    mapping_rows: list[dict[str, Any]] = []
    for run_position in (1, 2):
        row = ordered.iloc[run_position]
        assignment = assignments[row.run_id]
        valid = (
            reference_assignment.valid_time_mask
            & assignment.valid_time_mask
            & (reference_assignment.predicted_state >= 0)
            & (assignment.predicted_state >= 0)
        )
        counts = contingency_matrix(
            reference_assignment.predicted_state[valid], assignment.predicted_state[valid], k
        )
        matches = hungarian_state_matches(counts, thresholds=thresholds)
        for match in matches.itertuples(index=False):
            mapping_rows.append(
                {
                    "encoder": reference.encoder,
                    "encoder_seed": int(reference.encoder_seed),
                    "trial_id": reference_assignment.trial_id,
                    "reference_run_id": reference.run_id,
                    "reference_assignment_path": reference_assignment.source_path,
                    "mapped_run_id": row.run_id,
                    "mapped_downstream_seed": int(row.downstream_seed),
                    "mapped_assignment_path": assignment.source_path,
                    "reference_state": int(match.left_state),
                    "mapped_run_local_state": int(match.right_state),
                    "f1": float(match.f1),
                    "jaccard": float(match.jaccard),
                    "reciprocal_best": bool(match.reciprocal_best),
                    "defensible_for_consensus": bool(match.defensible_for_consensus),
                }
            )
            if bool(match.defensible_for_consensus):
                mapping_ok[run_position, int(match.right_state)] = True
                locations = assignment.predicted_state == int(match.right_state)
                mapped[run_position, locations] = int(match.left_state)

    rows: list[dict[str, Any]] = []
    for patch in range(reference_assignment.size):
        valid_patch = bool(reference_assignment.valid_time_mask[patch])
        votes = mapped[:, patch] if valid_patch else np.asarray([], dtype=np.int64)
        defensible = votes[votes >= 0]
        consensus_state = -1
        status = "invalid_patch"
        agreeing_votes = 0
        if valid_patch:
            if defensible.size < 3:
                status = "insufficient_defensible_mappings"
            else:
                counts = np.bincount(defensible, minlength=k)
                consensus_state = int(counts.argmax())
                agreeing_votes = int(counts.max())
                status = "unanimous_3_of_3" if agreeing_votes == 3 else "majority_2_of_3"
        rows.append(
            {
                "trial_id": reference_assignment.trial_id,
                "encoder": reference.encoder,
                "encoder_seed": int(reference.encoder_seed),
                "reference_downstream_seed": 0,
                "reference_run_id": reference.run_id,
                "reference_assignment_path": reference_assignment.source_path,
                "downstream_seed_1_assignment_path": assignments[
                    ordered.iloc[1].run_id
                ].source_path,
                "downstream_seed_2_assignment_path": assignments[
                    ordered.iloc[2].run_id
                ].source_path,
                "patch_id": patch,
                "patch_start_s": float(reference_assignment.patch_start_s[patch]),
                "patch_end_s": float(reference_assignment.patch_end_s[patch]),
                "valid_patch": valid_patch,
                "mapped_state_downstream_seed_0": int(mapped[0, patch]),
                "mapped_state_downstream_seed_1": int(mapped[1, patch]),
                "mapped_state_downstream_seed_2": int(mapped[2, patch]),
                "defensible_vote_count": int(defensible.size),
                "agreeing_vote_count": agreeing_votes,
                "consensus_state_in_reference_label_space": consensus_state,
                "consensus_status": status,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(mapping_rows)


def _measurement_comparison(
    matches: pd.DataFrame, measurement_csv: Path, runs: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    measurements = pd.read_csv(measurement_csv)
    measurements = measurements[
        measurements["algorithm"].eq("hmm") & measurements["K"].eq(6)
    ].copy()
    measurements["run_id"] = measurements.apply(run_id, axis=1)
    bout_rows: list[dict[str, Any]] = []
    for run in runs.itertuples(index=False):
        bouts = pd.read_csv(Path(str(run.bout_table)))
        total_minutes = float(bouts.bout_duration_s.sum() / 60.0)
        for state, group in bouts.groupby("state", sort=True):
            bout_rows.append(
                {
                    "run_id": run.run_id,
                    "state": int(state),
                    "state_bout_count": int(group.shape[0]),
                    "state_bouts_per_observed_minute": float(
                        group.shape[0] / max(total_minutes, 1e-12)
                    ),
                    "state_mean_bout_duration_s": float(group.bout_duration_s.mean()),
                    "state_median_bout_duration_s": float(group.bout_duration_s.median()),
                    "state_total_bout_duration_s": float(group.bout_duration_s.sum()),
                }
            )
    measurements = measurements.merge(
        pd.DataFrame(bout_rows), on=["run_id", "state"], how="left", validate="one_to_one"
    )
    metadata = {
        "encoder",
        "encoder_seed",
        "algorithm",
        "downstream_seed",
        "K",
        "max_duration_patches",
        "pca_dim",
        "status",
        "state",
        "source_table",
        "run_id",
    }
    numeric_columns = []
    for column in measurements.columns:
        if column in metadata:
            continue
        converted = pd.to_numeric(measurements[column], errors="coerce")
        if converted.notna().any():
            measurements[column] = converted
            numeric_columns.append(column)
    scales = measurements[numeric_columns].std(axis=0, ddof=1).replace(0.0, np.nan)
    lookup = measurements.set_index(["run_id", "state"])
    long_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for match in matches.itertuples(index=False):
        left = lookup.loc[(match.left_run_id, int(match.left_state))]
        right = lookup.loc[(match.right_run_id, int(match.right_state))]
        standardized: list[float] = []
        family_values: dict[str, list[float]] = {}
        for column in numeric_columns:
            left_value = float(left[column])
            right_value = float(right[column])
            scale = float(scales[column]) if np.isfinite(scales[column]) else float("nan")
            z_difference = (left_value - right_value) / scale if scale > 0 else float("nan")
            if "bout" in column:
                family = "bout"
            elif column.startswith("node_"):
                family = "node"
            elif "confidence" in column or "coverage" in column:
                family = "measurement_confidence_or_coverage"
            elif column.startswith("edge_"):
                family = "edge"
            elif column == "patches":
                family = "occupancy"
            else:
                family = "other"
            if np.isfinite(z_difference):
                standardized.append(z_difference)
                family_values.setdefault(family, []).append(z_difference)
            long_rows.append(
                {
                    "pair_id": match.pair_id,
                    "comparison_scope": match.comparison_scope,
                    "trial_id": match.trial_id,
                    "left_run_id": match.left_run_id,
                    "left_encoder": match.left_encoder,
                    "left_encoder_seed": int(match.left_encoder_seed),
                    "left_downstream_seed": int(match.left_downstream_seed),
                    "left_state": int(match.left_state),
                    "right_run_id": match.right_run_id,
                    "right_encoder": match.right_encoder,
                    "right_encoder_seed": int(match.right_encoder_seed),
                    "right_downstream_seed": int(match.right_downstream_seed),
                    "right_state": int(match.right_state),
                    "left_assignment_path": match.left_assignment_path,
                    "right_assignment_path": match.right_assignment_path,
                    "hungarian_f1": float(match.f1),
                    "measurement": column,
                    "measurement_family": family,
                    "left_value": left_value,
                    "right_value": right_value,
                    "difference": left_value - right_value,
                    "standardized_difference": z_difference,
                }
            )
        summary = {
            "pair_id": match.pair_id,
            "comparison_scope": match.comparison_scope,
            "trial_id": match.trial_id,
            "left_run_id": match.left_run_id,
            "left_encoder": match.left_encoder,
            "left_encoder_seed": int(match.left_encoder_seed),
            "left_downstream_seed": int(match.left_downstream_seed),
            "left_state": int(match.left_state),
            "right_run_id": match.right_run_id,
            "right_encoder": match.right_encoder,
            "right_encoder_seed": int(match.right_encoder_seed),
            "right_downstream_seed": int(match.right_downstream_seed),
            "right_state": int(match.right_state),
            "left_assignment_path": match.left_assignment_path,
            "right_assignment_path": match.right_assignment_path,
            "hungarian_f1": float(match.f1),
            "hungarian_jaccard": float(match.jaccard),
            "standardized_measurement_rms": float(
                np.sqrt(np.mean(np.square(standardized))) if standardized else np.nan
            ),
        }
        for family, values in family_values.items():
            summary[f"{family}_standardized_rms"] = float(
                np.sqrt(np.mean(np.square(values)))
            )
        summary_rows.append(summary)
    return pd.DataFrame(summary_rows), pd.DataFrame(long_rows)


def _plot_summaries(
    pair_summary: pd.DataFrame, per_state: pd.DataFrame, output: Path
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    order = [SAME_CHECKPOINT, ACROSS_ENCODER, GRAPH_VS_MLP]
    labels = ["Fixed encoder\nHMM seed", "Encoder seeds", "Graph vs MLP"]
    values = [
        pair_summary.loc[pair_summary.comparison_scope.eq(scope), "mean_hungarian_f1"].to_numpy()
        for scope in order
    ]
    figure, axis = plt.subplots(figsize=(7.5, 5.2))
    axis.boxplot(values, showfliers=False)
    rng = np.random.default_rng(20261020)
    for position, group in enumerate(values, 1):
        axis.scatter(position + rng.uniform(-0.08, 0.08, len(group)), group, s=16, alpha=0.65)
    axis.set_xticks([1, 2, 3], labels)
    axis.set_ylabel("Mean Hungarian matched-state F1")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "matched_state_f1_by_scope.png", dpi=200)
    plt.close(figure)

    summary = per_state.groupby(["encoder", "comparison_scope"], sort=True)[
        "mean_best_match_jaccard"
    ].mean().unstack()
    axis = summary.plot(kind="bar", figsize=(10.5, 5.4), ylim=(0, 1))
    axis.set_ylabel("Mean per-state best-match Jaccard")
    axis.set_xlabel("")
    axis.legend(title="Comparison scope", fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    axis.figure.tight_layout()
    axis.figure.savefig(output / "per_state_jaccard_by_encoder_and_scope.png", dpi=200)
    plt.close(axis.figure)

    figure, axis = plt.subplots(figsize=(7.5, 5.2))
    for scope, group in pair_summary.groupby("comparison_scope"):
        axis.scatter(
            group.ARI,
            group.boundary_f1_exact,
            s=24,
            alpha=0.65,
            label=scope,
        )
    axis.set_xlabel("Patch partition ARI")
    axis.set_ylabel("Exact bout-boundary F1")
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(-0.02, 1.02)
    axis.legend(fontsize=7)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "patch_ari_vs_boundary_f1.png", dpi=200)
    plt.close(figure)


def _state_report(
    output: Path,
    pair_summary: pd.DataFrame,
    per_state: pd.DataFrame,
    consensus: pd.DataFrame,
) -> None:
    group = pair_summary.groupby("comparison_scope").agg(
        pair_count=("pair_id", "size"),
        ari_mean=("ARI", "mean"),
        nmi_mean=("NMI", "mean"),
        matched_f1_mean=("mean_hungarian_f1", "mean"),
        boundary_f1_mean=("boundary_f1_exact", "mean"),
        split_mean=("left_split_candidate_count", "mean"),
        merge_mean=("right_merge_candidate_count", "mean"),
    )
    lines = [
        "# HMM K=6 cross-seed state correspondence",
        "",
        "Date: 2026-10-20",
        "",
        "This is a post-hoc audit of 36 saved A4_C4 HMM assignments. No model was retrained. State IDs remain run-local; Hungarian matches maximize overlapping patch count and are never treated as biological identities.",
        "",
        "## Scope summary",
        "",
        "| Scope | Pairs | Mean ARI | Mean NMI | Mean matched F1 | Exact boundary F1 | Mean splits | Mean merges |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scope, row in group.iterrows():
        lines.append(
            f"| `{scope}` | {int(row.pair_count)} | {row.ari_mean:.3f} | {row.nmi_mean:.3f} | "
            f"{row.matched_f1_mean:.3f} | {row.boundary_f1_mean:.3f} | {row.split_mean:.2f} | {row.merge_mean:.2f} |"
        )

    graph_seed = pair_summary[
        pair_summary.comparison_scope.eq(ACROSS_ENCODER)
        & pair_summary.left_encoder.eq("graph_full")
        & (
            pair_summary.left_encoder_seed.eq(20261029)
            | pair_summary.right_encoder_seed.eq(20261029)
        )
    ]
    graph_seed_f1 = float(graph_seed.mean_hungarian_f1.mean())
    graph_seed_ari = float(graph_seed.ARI.mean())
    consensus_valid = consensus[consensus.valid_patch]
    consensus_summary = (
        consensus_valid.groupby("encoder")["consensus_status"]
        .value_counts(normalize=True)
        .rename("fraction")
        .reset_index()
    )
    unanimous = consensus_summary[
        consensus_summary.consensus_status.eq("unanimous_3_of_3")
    ].set_index("encoder")["fraction"].to_dict()

    unstable = per_state[per_state.comparison_scope.eq(ACROSS_ENCODER)].nsmallest(
        10, "mean_best_match_jaccard"
    )
    stable_within = per_state[
        per_state.comparison_scope.eq(SAME_CHECKPOINT)
    ].nlargest(10, "mean_best_match_jaccard")
    stable_cross = per_state[
        per_state.comparison_scope.eq(ACROSS_ENCODER)
    ].nlargest(10, "mean_best_match_jaccard")
    stable_within_lines = "\n".join(
        f"- `{row.run_id}` state {int(row.state)}: best-match Jaccard "
        f"{row.mean_best_match_jaccard:.3f}, matched F1 {row.mean_hungarian_f1:.3f}."
        for row in stable_within.itertuples()
    )
    stable_cross_lines = "\n".join(
        f"- `{row.run_id}` state {int(row.state)}: best-match Jaccard "
        f"{row.mean_best_match_jaccard:.3f}, matched F1 {row.mean_hungarian_f1:.3f}."
        for row in stable_cross.itertuples()
    )
    unstable_lines = "\n".join(
        f"- `{row.run_id}` state {int(row.state)}: best-match Jaccard {row.mean_best_match_jaccard:.3f}, "
        f"ambiguity fraction {row.ambiguity_fraction:.2f}, split/merge fraction {row.split_or_merge_fraction:.2f}."
        for row in unstable.itertuples()
    )
    lines.extend(
        [
            "",
            "## Main findings",
            "",
            f"Graph Full encoder seed `20261029` is highly stable across its three HMM starts, but its comparisons with the other Graph Full encoder seeds average ARI **{graph_seed_ari:.3f}** and matched-state F1 **{graph_seed_f1:.3f}**. Easy downstream optimization therefore does not imply cross-encoder state identity.",
            "",
            "The deterministic consensus uses downstream seed 0 as the reference for each fixed encoder checkpoint. A mapping contributes only when its Hungarian match has F1 >= 0.40 and is reciprocal-best or has Jaccard >= 0.30. Valid-patch unanimous fractions are: "
            + "; ".join(
                f"{DISPLAY_NAMES.get(encoder, encoder)}={fraction:.3f}"
                for encoder, fraction in sorted(unanimous.items())
            )
            + ". Numeric state IDs are never averaged.",
            "",
            "Most reproducible run-local states across downstream starts of a fixed encoder checkpoint:",
            "",
            stable_within_lines,
            "",
            "Most reproducible run-local states across independently trained encoder seeds:",
            "",
            stable_cross_lines,
            "",
            f"Cross-encoder comparisons average {group.loc[ACROSS_ENCODER, 'split_mean']:.2f} left-state split candidates and {group.loc[ACROSS_ENCODER, 'merge_mean']:.2f} right-state merge candidates per six-state pair. The disagreement is therefore structured one-to-many/many-to-one overlap as well as weak alternatives, not merely a numeric label permutation.",
            "",
            f"Graph Full versus Invariant MLP Full averages ARI {group.loc[GRAPH_VS_MLP, 'ari_mean']:.3f}, matched-state F1 {group.loc[GRAPH_VS_MLP, 'matched_f1_mean']:.3f}, and exact boundary F1 {group.loc[GRAPH_VS_MLP, 'boundary_f1_mean']:.3f}. Their state systems overlap partially but should not be forced into one taxonomy.",
            "",
            "Lowest cross-encoder run-local state stabilities:",
            "",
            unstable_lines,
            "",
            "## Interpretation limits",
            "",
            "- Pairwise comparisons reuse the same A4_C4 patches and runs; rows are dependent diagnostics, not biological replicates.",
            "- Split, merge, ambiguity and weak-match labels use configurable heuristics documented in `analysis_manifest.json`.",
            "- Matching and stability do not establish biological validity or behavioral meaning.",
            "- Boundary F1 is computed only between adjacent valid physical patches and never across invalid gaps.",
        ]
    )
    (output / "state_correspondence_report.md").write_text("\n".join(lines), encoding="utf-8")


def analyze_state_correspondence(
    *,
    runs_csv: str | Path,
    measurement_csv: str | Path,
    output_dir: str | Path,
    trial_id: str = "A4_C4",
    k: int = 6,
    encoders: Sequence[str] = DEFAULT_ENCODERS,
    encoder_seeds: Sequence[int] = DEFAULT_ENCODER_SEEDS,
    downstream_seeds: Sequence[int] = DEFAULT_DOWNSTREAM_SEEDS,
    thresholds: CorrespondenceThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or CorrespondenceThresholds()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    matrix_dir = output / "contingency_matrices"
    plot_dir = output / "plots"
    matrix_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    runs = pd.read_csv(Path(runs_csv).resolve())
    selected = runs[
        runs.encoder.isin(encoders)
        & runs.encoder_seed.isin(encoder_seeds)
        & runs.downstream_seed.isin(downstream_seeds)
        & runs.algorithm.eq("hmm")
        & runs.K.eq(k)
    ].copy()
    expected = len(encoders) * len(encoder_seeds) * len(downstream_seeds)
    if selected.shape[0] != expected:
        raise ValueError(f"expected {expected} completed HMM runs, found {selected.shape[0]}")
    if not selected.status.eq("completed").all():
        raise ValueError("state correspondence input includes non-completed runs")
    selected["run_id"] = selected.apply(run_id, axis=1)
    if selected.run_id.duplicated().any():
        raise ValueError("duplicate HMM run IDs")

    assignments = {
        row.run_id: load_assignment(row, trial_id)
        for _, row in selected.iterrows()
    }
    reference_assignment = next(iter(assignments.values()))
    inconsistent = [
        identifier
        for identifier, assignment in assignments.items()
        if not _assignment_arrays_equal(reference_assignment, assignment)
    ]
    if inconsistent:
        raise ValueError(f"HMM assignments do not share one physical timeline: {inconsistent}")

    pair_specs = _enumerate_pairs(selected)
    expected_pairs = 36 + 108 + 81
    if len(pair_specs) != expected_pairs:
        raise ValueError(f"expected {expected_pairs} requested pair comparisons, found {len(pair_specs)}")

    pair_rows: list[dict[str, Any]] = []
    overlap_frames: list[pd.DataFrame] = []
    match_frames: list[pd.DataFrame] = []
    diagnostic_frames: list[pd.DataFrame] = []
    matrices: dict[str, np.ndarray] = {}
    artifact_slugs: dict[str, str] = {}
    for scope, left, right in pair_specs:
        left_assignment = assignments[left.run_id]
        right_assignment = assignments[right.run_id]
        valid = (
            left_assignment.valid_time_mask
            & right_assignment.valid_time_mask
            & (left_assignment.predicted_state >= 0)
            & (right_assignment.predicted_state >= 0)
        )
        left_labels = left_assignment.predicted_state[valid]
        right_labels = right_assignment.predicted_state[valid]
        matrix = contingency_matrix(left_labels, right_labels, k)
        identifier = _pair_id(left.run_id, right.run_id, scope)
        artifact_slug = _artifact_slug(identifier)
        artifact_slugs[identifier] = artifact_slug
        matrices[identifier] = matrix
        np.savez_compressed(
            matrix_dir / f"{artifact_slug}.npz",
            pair_id=np.asarray(identifier),
            comparison_scope=np.asarray(scope),
            trial_id=np.asarray(trial_id),
            left_run_id=np.asarray(left.run_id),
            right_run_id=np.asarray(right.run_id),
            left_assignment_path=np.asarray(left_assignment.source_path),
            right_assignment_path=np.asarray(right_assignment.source_path),
            contingency_count=matrix,
            row_normalized=np.divide(
                matrix,
                matrix.sum(axis=1, keepdims=True),
                out=np.zeros_like(matrix, dtype=float),
                where=matrix.sum(axis=1, keepdims=True) > 0,
            ),
            column_normalized=np.divide(
                matrix,
                matrix.sum(axis=0, keepdims=True),
                out=np.zeros_like(matrix, dtype=float),
                where=matrix.sum(axis=0, keepdims=True) > 0,
            ),
        )
        overlap = _annotate_pair(
            overlap_metrics(matrix), pair_id=identifier, scope=scope,
            left=left, right=right, left_assignment=left_assignment,
            right_assignment=right_assignment,
        )
        matches = _annotate_pair(
            hungarian_state_matches(matrix, thresholds=thresholds),
            pair_id=identifier, scope=scope, left=left, right=right,
            left_assignment=left_assignment, right_assignment=right_assignment,
        )
        diagnostics = _annotate_pair(
            split_merge_diagnostics(matrix, thresholds=thresholds),
            pair_id=identifier, scope=scope, left=left, right=right,
            left_assignment=left_assignment, right_assignment=right_assignment,
        )
        overlap["hungarian_match"] = False
        match_keys = set(zip(matches.left_state, matches.right_state))
        overlap["hungarian_match"] = [
            (int(row.left_state), int(row.right_state)) in match_keys
            for row in overlap.itertuples()
        ]
        boundaries = boundary_agreement(
            left_assignment.predicted_state,
            right_assignment.predicted_state,
            valid,
            left_assignment.patch_start_s,
            left_assignment.patch_end_s,
        )
        pair_rows.append(
            {
                "pair_id": identifier,
                "matrix_artifact": str(matrix_dir / f"{artifact_slug}.npz"),
                "comparison_scope": scope,
                "trial_id": trial_id,
                "left_run_id": left.run_id,
                "left_encoder": left.encoder,
                "left_encoder_seed": int(left.encoder_seed),
                "left_downstream_seed": int(left.downstream_seed),
                "left_assignment_path": left_assignment.source_path,
                "right_run_id": right.run_id,
                "right_encoder": right.encoder,
                "right_encoder_seed": int(right.encoder_seed),
                "right_downstream_seed": int(right.downstream_seed),
                "right_assignment_path": right_assignment.source_path,
                "aligned_valid_patches": int(valid.sum()),
                "ARI": float(adjusted_rand_score(left_labels, right_labels)),
                "NMI": float(normalized_mutual_info_score(left_labels, right_labels)),
                "hungarian_matched_patch_fraction": float(
                    matches.intersection_patches.sum() / max(valid.sum(), 1)
                ),
                "mean_hungarian_jaccard": float(matches.jaccard.mean()),
                "mean_hungarian_f1": float(matches.f1.mean()),
                "weak_forced_match_count": int(matches.match_status.eq("weak_forced_match").sum()),
                "reciprocal_match_count": int(matches.reciprocal_best.sum()),
                "left_split_candidate_count": int(diagnostics.split_candidate.sum()),
                "right_merge_candidate_count": int(diagnostics.merge_candidate.sum()),
                "ambiguous_state_direction_count": int(diagnostics.ambiguous.sum()),
                **boundaries,
            }
        )
        overlap_frames.append(overlap)
        match_frames.append(matches)
        diagnostic_frames.append(diagnostics)

    pair_summary = pd.DataFrame(pair_rows)
    overlap = pd.concat(overlap_frames, ignore_index=True)
    matches = pd.concat(match_frames, ignore_index=True)
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    pair_summary.to_csv(output / "pairwise_correspondence_summary.csv", index=False)
    overlap.to_csv(output / "pairwise_state_overlap.csv", index=False)
    matches.to_csv(output / "hungarian_matches.csv", index=False)
    diagnostics.to_csv(output / "split_merge_diagnostics.csv", index=False)
    pair_summary[pair_summary.comparison_scope.eq(ACROSS_ENCODER)].to_csv(
        output / "cross_encoder_correspondence.csv", index=False
    )
    pair_summary[pair_summary.comparison_scope.eq(GRAPH_VS_MLP)].to_csv(
        output / "gnn_vs_mlp_correspondence.csv", index=False
    )

    per_state = _per_state_stability(overlap, matches, diagnostics)
    per_state.to_csv(output / "per_state_stability.csv", index=False)

    consensus_frames: list[pd.DataFrame] = []
    consensus_mapping_frames: list[pd.DataFrame] = []
    for _, block in selected.groupby(["encoder", "encoder_seed"], sort=True):
        frame, mapping = within_encoder_consensus(
            block, assignments, k=k, thresholds=thresholds
        )
        consensus_frames.append(frame)
        consensus_mapping_frames.append(mapping)
    consensus = pd.concat(consensus_frames, ignore_index=True)
    consensus_mapping = pd.concat(consensus_mapping_frames, ignore_index=True)
    consensus.to_csv(output / "within_encoder_consensus.csv", index=False)
    consensus_mapping.to_csv(output / "within_encoder_consensus_mappings.csv", index=False)
    consensus_summary = (
        consensus.groupby(
            [
                "encoder",
                "encoder_seed",
                "trial_id",
                "reference_run_id",
                "reference_assignment_path",
                "valid_patch",
                "consensus_status",
            ]
        )
        .size()
        .rename("patches")
        .reset_index()
    )
    consensus_summary.to_csv(output / "within_encoder_consensus_summary.csv", index=False)

    measurement_summary, measurement_long = _measurement_comparison(
        matches, Path(measurement_csv).resolve(), selected
    )
    measurement_summary.to_csv(
        output / "matched_state_measurement_comparison.csv", index=False
    )
    measurement_long.to_csv(
        output / "matched_state_measurement_differences_long.csv", index=False
    )

    selected_plot_pairs = _select_plot_pairs(pair_summary)
    for pair_id in sorted(selected_plot_pairs):
        row = pair_summary[pair_summary.pair_id.eq(pair_id)].iloc[0]
        title = (
            f"{DISPLAY_NAMES.get(row.left_encoder, row.left_encoder)} e{row.left_encoder_seed}/d{row.left_downstream_seed} "
            f"vs {DISPLAY_NAMES.get(row.right_encoder, row.right_encoder)} e{row.right_encoder_seed}/d{row.right_downstream_seed}"
        )
        for mode in ("count", "row", "column"):
            _plot_matrix(
                matrices[pair_id], title,
                plot_dir / f"{artifact_slugs[pair_id]}_{mode}.png", mode,
            )
    _plot_summaries(pair_summary, per_state, plot_dir)
    _state_report(output, pair_summary, per_state, consensus)

    manifest = {
        "contract": "social_gnn_hmm_state_correspondence_v1",
        "source_runs_csv": str(Path(runs_csv).resolve()),
        "source_measurement_csv": str(Path(measurement_csv).resolve()),
        "trial_id": trial_id,
        "K": int(k),
        "saved_hmm_runs_used": int(selected.shape[0]),
        "pairwise_comparisons": int(pair_summary.shape[0]),
        "pair_counts": pair_summary.comparison_scope.value_counts().to_dict(),
        "matching_objective": "maximum matched patch count",
        "consensus_reference_rule": "downstream_seed_0_for_each_fixed_encoder_checkpoint",
        "thresholds": thresholds.__dict__,
        "models_retrained": 0,
        "notes": [
            "Run-local state IDs are preserved.",
            "Hungarian matches are forced one-to-one diagnostics, not biological state identities.",
            "All comparisons use identical A4_C4 physical patch keys and mutually valid observations.",
            "Pairwise rows are dependent because runs and patches are reused.",
        ],
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return {
        "pair_summary": pair_summary,
        "overlap": overlap,
        "matches": matches,
        "diagnostics": diagnostics,
        "per_state": per_state,
        "consensus": consensus,
        "manifest": manifest,
    }
