from __future__ import annotations

import pandas as pd

from social_gnn.downstream.final_k_selection import (
    _localize_artifact_paths,
    parameter_count,
    select_working_k,
)


def test_expected_hmm_parameter_counts() -> None:
    assert parameter_count(12) == 527
    assert parameter_count(13) == 584
    assert parameter_count(14) == 643


def test_selection_uses_parsimony_for_reproducibility_tie() -> None:
    summary = pd.DataFrame(
        {
            "K": [12, 13, 14],
            "cross_matched_ari_mean": [0.50, 0.51, 0.49],
            "within_ari_mean": [0.80, 0.81, 0.79],
        }
    )
    refinement = pd.DataFrame(
        {
            "left_K": [12, 13, 12],
            "right_K": [13, 14, 14],
            "additional_state_support": [
                "ambiguous_or_inconsistent",
                "ambiguous_or_inconsistent",
                "ambiguous_or_inconsistent",
            ],
        }
    )
    selected, eligible = select_working_k(summary, refinement)
    assert selected == 12
    assert eligible == [12, 13, 14]


def test_supported_refinement_can_justify_higher_k() -> None:
    summary = pd.DataFrame(
        {
            "K": [12, 13, 14],
            "cross_matched_ari_mean": [0.50, 0.51, 0.30],
            "within_ari_mean": [0.80, 0.81, 0.90],
        }
    )
    refinement = pd.DataFrame(
        {
            "left_K": [12, 13, 12],
            "right_K": [13, 14, 14],
            "additional_state_support": [
                "supported",
                "ambiguous_or_inconsistent",
                "ambiguous_or_inconsistent",
            ],
        }
    )
    selected, eligible = select_working_k(summary, refinement)
    assert selected == 13
    assert eligible == [12, 13]


def test_windows_artifact_paths_are_localized_for_wsl(tmp_path) -> None:
    frame = pd.DataFrame(
        {"assignment": [r"C:\Users\DELL\Desktop\deepOF\evaluation_runs\x.npz"]}
    )
    localized = _localize_artifact_paths(frame, tmp_path)
    assert localized.assignment.iloc[0] == str(tmp_path / "evaluation_runs/x.npz")
