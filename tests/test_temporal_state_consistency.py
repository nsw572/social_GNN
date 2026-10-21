from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from social_gnn.downstream.ari_stability import AssignmentData
from social_gnn.downstream.state_correspondence import (
    CorrespondenceThresholds,
    boundary_agreement,
    contingency_matrix,
    hungarian_state_matches,
    overlap_metrics,
    split_merge_diagnostics,
    within_encoder_consensus,
)
from social_gnn.downstream.temporal_alignment import (
    _mapping_frame,
    contiguous_valid_runs,
    map_patch_intervals_to_frames,
    probe_video_pts,
    recompute_bouts,
    validate_patch_clock,
    validate_saved_bouts,
)


def _assignment(labels: np.ndarray, *, valid: np.ndarray | None = None) -> AssignmentData:
    labels = np.asarray(labels, dtype=np.int64)
    start = 3.25 + np.arange(labels.size, dtype=float) * 0.16
    if valid is None:
        valid = np.ones(labels.size, dtype=bool)
    saved = labels.copy()
    saved[~valid] = -1
    return AssignmentData(
        trial_id="synthetic",
        patch_start_s=start,
        patch_end_s=start + 0.16,
        valid_time_mask=np.asarray(valid, dtype=bool),
        predicted_state=saved,
        source_path="synthetic.npz",
    )


def test_pts_mapping_handles_nonzero_origin_offset_and_vfr() -> None:
    pts = np.asarray([10.0, 10.04, 10.09, 10.13, 10.18])
    mapped = map_patch_intervals_to_frames(
        np.asarray([2.00, 2.08]),
        np.asarray([2.08, 2.16]),
        pts,
        video_time_offset_s=8.0,
    )
    assert mapped["video_frame_start"].tolist() == [0, 2]
    assert mapped["video_frame_end"].tolist() == [1, 3]
    assert mapped["video_frame_center"].tolist() == [1, 3]
    assert mapped["alignment_error_s"] == pytest.approx([0.0, 0.01])


def test_patch_clock_validation_detects_nonuniform_but_valid_clock() -> None:
    result = validate_patch_clock(
        np.asarray([5.0, 5.16, 5.35]), np.asarray([5.16, 5.32, 5.51])
    )
    assert result["clock_valid"]
    assert result["inter_patch_gap_max_s"] == pytest.approx(0.03)


def test_contiguous_runs_and_bouts_do_not_cross_invalid_gap() -> None:
    valid = np.asarray([True, True, False, True, True])
    assignment = _assignment(np.asarray([0, 0, 1, 0, 0]), valid=valid)
    runs = contiguous_valid_runs(valid, assignment.patch_start_s, assignment.patch_end_s)
    assert [values.tolist() for values in runs] == [[0, 1], [3, 4]]
    bouts = recompute_bouts(assignment)
    assert len(bouts) == 2
    assert bouts["run_id"].tolist() == [0, 1]
    assert bouts["bout_duration_s"].tolist() == pytest.approx([0.32, 0.32])
    audit = validate_saved_bouts(assignment, bouts)
    assert audit["bout_rows_exact"]
    assert audit["partition_nonoverlap"]


def test_label_correspondence_is_permutation_invariant() -> None:
    left = np.repeat(np.arange(6), 10)
    permutation = np.asarray([4, 2, 5, 1, 0, 3])
    right = permutation[left]
    matrix = contingency_matrix(left, right, 6)
    matches = hungarian_state_matches(matrix)
    assert adjusted_rand_score(left, right) == pytest.approx(1.0)
    assert matches["f1"].tolist() == pytest.approx([1.0] * 6)
    assert matches["reciprocal_best"].all()
    assert matches.sort_values("left_state")["right_state"].tolist() == permutation.tolist()


def test_known_split_merge_and_ambiguity_are_flagged() -> None:
    matrix = np.asarray(
        [
            [40, 35, 0],
            [0, 30, 30],
            [0, 0, 50],
        ]
    )
    diagnostics = split_merge_diagnostics(
        matrix,
        thresholds=CorrespondenceThresholds(component_fraction=0.20, ambiguity_margin=0.10),
    )
    left_zero = diagnostics[
        diagnostics.direction.eq("left_to_right") & diagnostics.source_state.eq(0)
    ].iloc[0]
    right_one = diagnostics[
        diagnostics.direction.eq("right_to_left") & diagnostics.source_state.eq(1)
    ].iloc[0]
    assert left_zero.split_candidate
    assert left_zero.ambiguous
    assert right_one.merge_candidate


def test_overlap_metrics_keep_both_conditionals_visible() -> None:
    metrics = overlap_metrics(np.asarray([[30, 10], [20, 40]])).set_index(
        ["left_state", "right_state"]
    )
    cell = metrics.loc[(0, 0)]
    assert cell.p_right_given_left == pytest.approx(0.75)
    assert cell.p_left_given_right == pytest.approx(0.60)
    assert cell.jaccard == pytest.approx(0.50)


def test_boundary_agreement_excludes_invalid_gap() -> None:
    left = np.asarray([0, 0, -1, 1, 1])
    right = np.asarray([0, 1, -1, 0, 0])
    valid = np.asarray([True, True, False, True, True])
    start = np.arange(5) * 0.16
    result = boundary_agreement(left, right, valid, start, start + 0.16)
    assert result["eligible_adjacent_patch_pairs"] == 2
    assert result["left_boundary_count"] == 0
    assert result["right_boundary_count"] == 1
    assert result["boundary_f1_exact"] == 0.0


def test_consensus_uses_defensible_permutations_not_numeric_averages() -> None:
    base = np.repeat(np.arange(6), 3)
    perm1 = np.asarray([2, 4, 1, 5, 0, 3])
    perm2 = np.asarray([5, 3, 4, 0, 2, 1])
    assignments = {
        "enc|e1|hmm|k6|d0": _assignment(base),
        "enc|e1|hmm|k6|d1": _assignment(perm1[base]),
        "enc|e1|hmm|k6|d2": _assignment(perm2[base]),
    }
    block = pd.DataFrame(
        [
            {"encoder": "enc", "encoder_seed": 1, "downstream_seed": seed,
             "run_id": f"enc|e1|hmm|k6|d{seed}"}
            for seed in range(3)
        ]
    )
    consensus, mappings = within_encoder_consensus(
        block, assignments, k=6, thresholds=CorrespondenceThresholds()
    )
    assert consensus.consensus_status.eq("unanimous_3_of_3").all()
    assert np.array_equal(
        consensus.consensus_state_in_reference_label_space.to_numpy(), base
    )
    assert mappings.defensible_for_consensus.all()


def test_absent_video_is_reported_without_fabricated_frames() -> None:
    missing = Path("definitely_missing_social_gnn_video.mp4").resolve()
    with pytest.raises(Exception):
        probe_video_pts(missing, sys.executable)
    frame = _mapping_frame(
        trial_id="missing",
        patch_start_s=np.asarray([1.0]),
        patch_end_s=np.asarray([1.16]),
        valid_patch=np.asarray([True]),
        source_video=str(missing),
        source_node_npz="node.npz",
        source_edge_npz="edge.npz",
        timeline=None,
        offset_s=None,
        provenance="unresolved",
    )
    assert frame.loc[0, "alignment_status"] == "missing_or_unreadable_source_video"
    assert pd.isna(frame.loc[0, "video_frame_center"])
