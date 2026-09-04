"""Frame-wise MouseGPT to idtracker.ai identity matching.

idtracker.ai identities are treated as authoritative. MouseGPT ``track_id`` is
retained for diagnostics only and never introduces temporal inertia. Matching
is a one-to-one linear assignment performed independently on every frame.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


BODY_ANCHOR_PARTS: tuple[tuple[str, float], ...] = (
    ("SpineF", 0.4),
    ("SpineG", 0.3),
    ("SpineH", 0.2),
    ("Hip", 0.1),
)

MATCH_COLUMNS = [
    "mousegpt_row_id",
    "frame_id",
    "mousegpt_track_id",
    "anchor_x_px",
    "anchor_y_px",
    "anchor_method",
    "anchor_confidence",
    "proposed_idtracker_identity",
    "idtracker_identity",
    "association_valid",
    "association_status",
    "association_distance_px",
    "alternative_identity_distance_px",
    "assignment_margin_px",
    "idtracker_centroid_confidence",
    "distance_confidence",
    "margin_confidence",
    "association_confidence",
]


def add_mousegpt_body_anchors(mousegpt: pd.DataFrame) -> pd.DataFrame:
    """Add a robust body anchor without changing the original track labels."""
    required = {"frame_id", "track_id", "bbox_score"}
    for name, _weight in BODY_ANCHOR_PARTS:
        required.update({f"{name}_x", f"{name}_y", f"{name}_score"})
    missing = sorted(required - set(mousegpt.columns))
    if missing:
        raise ValueError(f"MouseGPT CSV lacks required columns: {missing}")

    result = mousegpt.copy().reset_index(drop=True)
    result.insert(0, "mousegpt_row_id", np.arange(len(result), dtype=np.int64))
    weight_sum = np.zeros(len(result), dtype=float)
    anchor_x_sum = np.zeros(len(result), dtype=float)
    anchor_y_sum = np.zeros(len(result), dtype=float)
    score_arrays: list[np.ndarray] = []
    used_count = np.zeros(len(result), dtype=np.int16)

    for name, weight in BODY_ANCHOR_PARTS:
        x = pd.to_numeric(result[f"{name}_x"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(result[f"{name}_y"], errors="coerce").to_numpy(float)
        score = pd.to_numeric(
            result[f"{name}_score"], errors="coerce"
        ).to_numpy(float)
        valid = np.isfinite(x) & np.isfinite(y)
        weight_sum += weight * valid
        anchor_x_sum += weight * np.where(valid, x, 0.0)
        anchor_y_sum += weight * np.where(valid, y, 0.0)
        score_arrays.append(np.where(valid & np.isfinite(score), score, np.inf))
        used_count += valid.astype(np.int16)

    has_body_anchor = weight_sum > 0
    anchor_x = np.full(len(result), np.nan, dtype=float)
    anchor_y = np.full(len(result), np.nan, dtype=float)
    anchor_x[has_body_anchor] = (
        anchor_x_sum[has_body_anchor] / weight_sum[has_body_anchor]
    )
    anchor_y[has_body_anchor] = (
        anchor_y_sum[has_body_anchor] / weight_sum[has_body_anchor]
    )

    body_confidence = np.min(np.stack(score_arrays, axis=1), axis=1)
    body_confidence[~np.isfinite(body_confidence)] = np.nan

    bbox_columns = {"bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"}
    if bbox_columns.issubset(result.columns):
        bbox_x1 = pd.to_numeric(result["bbox_x1"], errors="coerce").to_numpy(float)
        bbox_y1 = pd.to_numeric(result["bbox_y1"], errors="coerce").to_numpy(float)
        bbox_x2 = pd.to_numeric(result["bbox_x2"], errors="coerce").to_numpy(float)
        bbox_y2 = pd.to_numeric(result["bbox_y2"], errors="coerce").to_numpy(float)
        bbox_valid = (
            np.isfinite(bbox_x1)
            & np.isfinite(bbox_y1)
            & np.isfinite(bbox_x2)
            & np.isfinite(bbox_y2)
        )
        use_bbox = ~has_body_anchor & bbox_valid
        anchor_x[use_bbox] = (bbox_x1[use_bbox] + bbox_x2[use_bbox]) / 2.0
        anchor_y[use_bbox] = (bbox_y1[use_bbox] + bbox_y2[use_bbox]) / 2.0
    else:
        use_bbox = np.zeros(len(result), dtype=bool)

    bbox_score = pd.to_numeric(result["bbox_score"], errors="coerce").to_numpy(float)
    anchor_confidence = np.where(has_body_anchor, body_confidence, np.nan)
    anchor_confidence[use_bbox] = bbox_score[use_bbox]
    anchor_method = np.full(len(result), "missing", dtype=object)
    anchor_method[has_body_anchor] = "body_keypoints"
    anchor_method[use_bbox] = "bbox_center"

    result["mousegpt_track_id"] = pd.to_numeric(
        result["track_id"], errors="coerce"
    ).astype("Int64")
    result["anchor_x_px"] = anchor_x
    result["anchor_y_px"] = anchor_y
    result["anchor_method"] = anchor_method
    result["anchor_keypoint_count"] = used_count
    result["anchor_confidence"] = anchor_confidence
    return result


def _prepare_idtracker_centroids(idtracker: pd.DataFrame) -> pd.DataFrame:
    required = {
        "frame_id",
        "identity",
        "centroid_x_px",
        "centroid_y_px",
        "centroid_valid",
    }
    missing = sorted(required - set(idtracker.columns))
    if missing:
        raise ValueError(f"idtracker CSV lacks required columns: {missing}")
    result = idtracker.copy().reset_index(drop=True)
    if result.duplicated(["frame_id", "identity"]).any():
        raise ValueError("idtracker CSV contains duplicate (frame_id, identity) rows")
    for column in ("frame_id", "identity"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype(int)
    for column in ("centroid_x_px", "centroid_y_px"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["centroid_valid"] = (
        pd.to_numeric(result["centroid_valid"], errors="coerce").fillna(0).astype(bool)
        & np.isfinite(result["centroid_x_px"])
        & np.isfinite(result["centroid_y_px"])
    )
    if "centroid_confidence_v0" in result:
        result["centroid_confidence_v0"] = np.clip(
            pd.to_numeric(result["centroid_confidence_v0"], errors="coerce")
            .fillna(0.0)
            .to_numpy(float),
            0.0,
            1.0,
        )
    else:
        result["centroid_confidence_v0"] = result["centroid_valid"].astype(float)
    return result


def _second_best_assignment_margin(
    costs: np.ndarray, rows: np.ndarray, cols: np.ndarray
) -> tuple[float, float, float]:
    """Return best cost, second-best cost, and their non-negative margin."""
    best = float(costs[rows, cols].sum())
    candidates: list[float] = []
    expected_pairs = min(costs.shape)
    for row, col in zip(rows, cols):
        alternative = costs.copy()
        alternative[row, col] = np.inf
        try:
            alt_rows, alt_cols = linear_sum_assignment(alternative)
        except ValueError:
            continue
        if len(alt_rows) != expected_pairs:
            continue
        selected = alternative[alt_rows, alt_cols]
        if np.isfinite(selected).all():
            candidates.append(float(selected.sum()))
    if not candidates:
        return best, float("nan"), float("nan")
    second = min(candidates)
    return best, second, max(0.0, second - best)


def _confidence_from_distance(distance: float, scale: float) -> float:
    if not np.isfinite(distance):
        return 0.0
    return float(math.exp(-0.5 * (distance / scale) ** 2))


def _confidence_from_margin(margin: float, scale: float) -> float:
    if not np.isfinite(margin):
        return 1.0
    return float(np.clip(margin / scale, 0.0, 1.0))


def match_mousegpt_to_idtracker(
    mousegpt: pd.DataFrame,
    idtracker: pd.DataFrame,
    *,
    max_distance_px: float = 200.0,
    min_assignment_margin_px: float = 100.0,
    distance_confidence_scale_px: float = 100.0,
    margin_confidence_scale_px: float = 300.0,
    min_anchor_confidence: float = 0.30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Match independently at each frame and return matches, enriched rows, frames."""
    if max_distance_px <= 0 or distance_confidence_scale_px <= 0:
        raise ValueError("distance thresholds must be positive")
    if min_assignment_margin_px < 0 or margin_confidence_scale_px <= 0:
        raise ValueError("margin thresholds must be non-negative/positive")

    enriched = add_mousegpt_body_anchors(mousegpt)
    id_data = _prepare_idtracker_centroids(idtracker)
    enriched["frame_id"] = pd.to_numeric(
        enriched["frame_id"], errors="raise"
    ).astype(int)

    n_rows = len(enriched)
    proposed_identity = np.full(n_rows, np.nan, dtype=float)
    final_identity = np.full(n_rows, np.nan, dtype=float)
    association_valid = np.zeros(n_rows, dtype=bool)
    status = np.full(n_rows, "unprocessed", dtype=object)
    assigned_distance = np.full(n_rows, np.nan, dtype=float)
    alternative_distance = np.full(n_rows, np.nan, dtype=float)
    assignment_margin = np.full(n_rows, np.nan, dtype=float)
    idtracker_centroid_confidence = np.zeros(n_rows, dtype=float)
    distance_confidence = np.zeros(n_rows, dtype=float)
    margin_confidence = np.zeros(n_rows, dtype=float)
    association_confidence = np.zeros(n_rows, dtype=float)

    mouse_groups = enriched.groupby("frame_id", sort=False).indices
    id_groups = id_data.groupby("frame_id", sort=False).indices
    all_frames = sorted(set(mouse_groups) | set(id_groups))
    frame_records: list[dict[str, Any]] = []

    for frame_id in all_frames:
        mouse_indices = np.asarray(mouse_groups.get(frame_id, []), dtype=int)
        id_indices = np.asarray(id_groups.get(frame_id, []), dtype=int)
        valid_mouse_indices = mouse_indices[
            np.isfinite(enriched.loc[mouse_indices, "anchor_x_px"].to_numpy(float))
            & np.isfinite(enriched.loc[mouse_indices, "anchor_y_px"].to_numpy(float))
        ]
        valid_id_rows = id_data.loc[id_indices]
        valid_id_rows = valid_id_rows[valid_id_rows["centroid_valid"]]

        if len(mouse_indices):
            missing_anchor = np.setdiff1d(
                mouse_indices, valid_mouse_indices, assume_unique=False
            )
            status[missing_anchor] = "anchor_missing"
        if len(valid_mouse_indices) and valid_id_rows.empty:
            status[valid_mouse_indices] = "no_valid_idtracker_centroid"

        best_cost = second_cost = global_margin = float("nan")
        n_proposed = n_valid = 0
        if len(valid_mouse_indices) and not valid_id_rows.empty:
            anchors = enriched.loc[
                valid_mouse_indices, ["anchor_x_px", "anchor_y_px"]
            ].to_numpy(float)
            centroids = valid_id_rows[
                ["centroid_x_px", "centroid_y_px"]
            ].to_numpy(float)
            identities = valid_id_rows["identity"].to_numpy(int)
            idtracker_confidences = valid_id_rows[
                "centroid_confidence_v0"
            ].to_numpy(float)
            costs = np.linalg.norm(
                anchors[:, None, :] - centroids[None, :, :], axis=-1
            )
            assignment_rows, assignment_cols = linear_sum_assignment(costs)
            best_cost, second_cost, global_margin = _second_best_assignment_margin(
                costs, assignment_rows, assignment_cols
            )
            assigned_local_rows = set(assignment_rows.tolist())

            for local_row, local_col in zip(assignment_rows, assignment_cols):
                row_id = int(valid_mouse_indices[local_row])
                identity = int(identities[local_col])
                distance = float(costs[local_row, local_col])
                other_distances = np.delete(costs[local_row], local_col)
                alternative = (
                    float(np.min(other_distances))
                    if len(other_distances)
                    else float("nan")
                )
                anchor_conf = float(enriched.loc[row_id, "anchor_confidence"])
                problems: list[str] = []
                if distance > max_distance_px:
                    problems.append("distance")
                if np.isfinite(global_margin) and global_margin < min_assignment_margin_px:
                    problems.append("ambiguous")
                if not np.isfinite(anchor_conf) or anchor_conf < min_anchor_confidence:
                    problems.append("anchor_confidence")

                q_distance = _confidence_from_distance(
                    distance, distance_confidence_scale_px
                )
                q_margin = _confidence_from_margin(
                    global_margin, margin_confidence_scale_px
                )
                q_anchor = float(np.clip(anchor_conf, 0.0, 1.0)) if np.isfinite(anchor_conf) else 0.0
                q_idtracker = float(idtracker_confidences[local_col])
                valid = not problems

                proposed_identity[row_id] = identity
                assigned_distance[row_id] = distance
                alternative_distance[row_id] = alternative
                assignment_margin[row_id] = global_margin
                idtracker_centroid_confidence[row_id] = q_idtracker
                distance_confidence[row_id] = q_distance
                margin_confidence[row_id] = q_margin
                association_valid[row_id] = valid
                status[row_id] = "matched" if valid else "rejected:" + "+".join(problems)
                association_confidence[row_id] = (
                    min(q_anchor, q_idtracker, q_distance, q_margin)
                    if valid
                    else 0.0
                )
                if valid:
                    final_identity[row_id] = identity
                    n_valid += 1
                n_proposed += 1

            unassigned_local = sorted(
                set(range(len(valid_mouse_indices))) - assigned_local_rows
            )
            if unassigned_local:
                unassigned_rows = valid_mouse_indices[unassigned_local]
                status[unassigned_rows] = "unassigned_competition"

        frame_records.append(
            {
                "frame_id": frame_id,
                "mousegpt_detection_count": int(len(mouse_indices)),
                "mousegpt_valid_anchor_count": int(len(valid_mouse_indices)),
                "idtracker_valid_centroid_count": int(len(valid_id_rows)),
                "proposed_match_count": int(n_proposed),
                "valid_match_count": int(n_valid),
                "best_assignment_cost_px": best_cost,
                "second_assignment_cost_px": second_cost,
                "assignment_margin_px": global_margin,
            }
        )

    enriched["proposed_idtracker_identity"] = pd.array(
        proposed_identity, dtype="Int64"
    )
    enriched["idtracker_identity"] = pd.array(final_identity, dtype="Int64")
    enriched["association_valid"] = association_valid
    enriched["association_status"] = status
    enriched["association_distance_px"] = assigned_distance
    enriched["alternative_identity_distance_px"] = alternative_distance
    enriched["assignment_margin_px"] = assignment_margin
    enriched["idtracker_centroid_confidence"] = idtracker_centroid_confidence
    enriched["distance_confidence"] = distance_confidence
    enriched["margin_confidence"] = margin_confidence
    enriched["association_confidence"] = association_confidence
    matches = enriched[MATCH_COLUMNS].copy()
    frame_summary = pd.DataFrame(frame_records)
    return matches, enriched, frame_summary


def _quantiles(series: pd.Series) -> dict[str, float]:
    finite = pd.to_numeric(series, errors="coerce")
    finite = finite[np.isfinite(finite)]
    if finite.empty:
        return {}
    return {
        str(q): float(value)
        for q, value in finite.quantile([0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0]).items()
    }


def _track_identity_change_diagnostics(
    enriched: pd.DataFrame,
) -> dict[str, dict[str, int]]:
    """Describe how MouseGPT's diagnostic track labels map to idtracker IDs.

    A change across a frame gap is kept separate from a true consecutive-frame
    flip. MouseGPT track IDs never influence the assignment itself.
    """
    valid = enriched[
        enriched["association_valid"] & enriched["mousegpt_track_id"].notna()
    ].copy()
    diagnostics: dict[str, dict[str, int]] = {}
    for track_id, rows in valid.groupby("mousegpt_track_id"):
        ordered = rows.sort_values("frame_id")
        frames = pd.to_numeric(ordered["frame_id"], errors="coerce")
        identities = ordered["idtracker_identity"]
        changed = identities.ne(identities.shift())
        if len(changed):
            changed.iloc[0] = False
        consecutive = frames.diff().eq(1)
        diagnostics[str(int(track_id))] = {
            "all_adjacent_valid_observation_changes": int(changed.sum()),
            "consecutive_frame_changes": int((changed & consecutive).sum()),
            "changes_across_frame_gaps": int((changed & ~consecutive).sum()),
        }
    return diagnostics


def run_matching(
    *,
    mousegpt_csv: Path,
    idtracker_csv: Path,
    output_dir: Path,
    output_stem: str,
    max_distance_px: float = 200.0,
    min_assignment_margin_px: float = 100.0,
    distance_confidence_scale_px: float = 100.0,
    margin_confidence_scale_px: float = 300.0,
    min_anchor_confidence: float = 0.30,
) -> dict[str, Any]:
    mousegpt_csv = mousegpt_csv.resolve()
    idtracker_csv = idtracker_csv.resolve()
    if not mousegpt_csv.is_file():
        raise FileNotFoundError(mousegpt_csv)
    if not idtracker_csv.is_file():
        raise FileNotFoundError(idtracker_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    mousegpt = pd.read_csv(mousegpt_csv)
    idtracker = pd.read_csv(idtracker_csv)
    matches, enriched, frames = match_mousegpt_to_idtracker(
        mousegpt,
        idtracker,
        max_distance_px=max_distance_px,
        min_assignment_margin_px=min_assignment_margin_px,
        distance_confidence_scale_px=distance_confidence_scale_px,
        margin_confidence_scale_px=margin_confidence_scale_px,
        min_anchor_confidence=min_anchor_confidence,
    )

    matches_path = output_dir / f"{output_stem}_identity_matches.csv"
    enriched_path = output_dir / f"{output_stem}_mousegpt_with_idtracker_identity.csv"
    frames_path = output_dir / f"{output_stem}_identity_match_frames.csv"
    summary_path = output_dir / f"{output_stem}_identity_match_summary.json"
    matches.to_csv(matches_path, index=False, encoding="utf-8-sig")
    enriched.to_csv(enriched_path, index=False, encoding="utf-8-sig")
    frames.to_csv(frames_path, index=False, encoding="utf-8-sig")

    status_counts = {
        str(key): int(value)
        for key, value in matches["association_status"].value_counts().items()
    }
    summary: dict[str, Any] = {
        "contract_version": "identity_matching_v0",
        "method": "independent_per_frame_one_to_one_linear_assignment",
        "identity_authority": "idtracker.ai",
        "mousegpt_track_usage": "diagnostic_only",
        "mousegpt_csv": str(mousegpt_csv),
        "idtracker_csv": str(idtracker_csv),
        "thresholds": {
            "max_distance_px": max_distance_px,
            "min_assignment_margin_px": min_assignment_margin_px,
            "distance_confidence_scale_px": distance_confidence_scale_px,
            "margin_confidence_scale_px": margin_confidence_scale_px,
            "min_anchor_confidence": min_anchor_confidence,
        },
        "mousegpt_rows": int(len(enriched)),
        "valid_matches": int(enriched["association_valid"].sum()),
        "invalid_matches": int((~enriched["association_valid"]).sum()),
        "status_counts": status_counts,
        "distance_quantiles_px": _quantiles(matches["association_distance_px"]),
        "assignment_margin_quantiles_px": _quantiles(
            matches["assignment_margin_px"]
        ),
        "association_confidence_quantiles": _quantiles(
            matches.loc[matches["association_valid"], "association_confidence"]
        ),
        "mousegpt_track_identity_changes": _track_identity_change_diagnostics(
            enriched
        ),
        "outputs": {
            "matches_csv": str(matches_path),
            "enriched_mousegpt_csv": str(enriched_path),
            "frame_summary_csv": str(frames_path),
            "summary_json": str(summary_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match MouseGPT detections to idtracker.ai identities per frame."
    )
    parser.add_argument("--mousegpt-csv", required=True, type=Path)
    parser.add_argument("--idtracker-csv", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--output-stem", required=True)
    parser.add_argument("--max-distance-px", type=float, default=200.0)
    parser.add_argument("--min-assignment-margin-px", type=float, default=100.0)
    parser.add_argument("--distance-confidence-scale-px", type=float, default=100.0)
    parser.add_argument("--margin-confidence-scale-px", type=float, default=300.0)
    parser.add_argument("--min-anchor-confidence", type=float, default=0.30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_matching(
        mousegpt_csv=args.mousegpt_csv,
        idtracker_csv=args.idtracker_csv,
        output_dir=args.outdir,
        output_stem=args.output_stem,
        max_distance_px=args.max_distance_px,
        min_assignment_margin_px=args.min_assignment_margin_px,
        distance_confidence_scale_px=args.distance_confidence_scale_px,
        margin_confidence_scale_px=args.margin_confidence_scale_px,
        min_anchor_confidence=args.min_anchor_confidence,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
