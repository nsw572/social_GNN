from __future__ import annotations

import numpy as np

import argparse

import pytest

from social_gnn.downstream.rslds_completion import GRID, _parse_cell, _run_id


def test_completed_pilot_grid_has_exactly_fifteen_unique_cells() -> None:
    assert len(GRID) == 15
    assert len(set(GRID)) == 15
    assert set((k, m) for k, m, _ in GRID) == {
        (4, 4),
        (6, 4),
        (8, 4),
        (6, 8),
        (8, 8),
    }
    assert all(set(seed for kk, mm, seed in GRID if (kk, mm) == (k, m)) == {0, 1, 2} for k, m in set((k, m) for k, m, _ in GRID))


def test_run_ids_encode_dimensions_and_seed() -> None:
    assert _run_id(8, 4, 2) == "k8_m4_seed2"


def test_selected_cell_parser_accepts_declared_k8_cells() -> None:
    assert _parse_cell("8,4,0") == (8, 4, 0)
    assert _parse_cell("8,8,0") == (8, 8, 0)


def test_selected_cell_parser_rejects_cells_outside_grid() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="outside"):
        _parse_cell("8,6,0")


def test_ari_is_invariant_to_state_label_permutation() -> None:
    from sklearn.metrics import adjusted_rand_score

    left = np.array([0, 0, 1, 1, 2, 2])
    right = np.array([2, 2, 0, 0, 1, 1])
    assert adjusted_rand_score(left, right) == 1.0
