from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from social_gnn.downstream.k_selection import prior_strength_for_k
from social_gnn.downstream.k_selection_k30 import (
    EXTENSION_K_GRID,
    FULL_K_GRID,
    PRIOR_CONTROL_K_GRID,
    REUSED_K_GRID,
    _add_k12_deltas,
    _drop_stale_reference_columns,
    _restore_initial_fit_cost,
    _marginal_gains,
)


def _row(k: int, ll: float, parameters: int) -> dict:
    return {
        "architecture": "graph_full",
        "encoder_seed": 1,
        "hmm_seed": 0,
        "K": k,
        "status": "completed",
        "dev_log_likelihood_total": ll,
        "dev_log_likelihood_per_point": ll / 100.0,
        "aic_train": -2.0 * ll + 2.0 * parameters,
        "bic_train": -2.0 * ll + np.log(100.0) * parameters,
        "parameter_count": parameters,
    }


def test_declared_k30_grid_is_exact() -> None:
    assert REUSED_K_GRID == (4, 5, 6, 7, 8, 10, 12)
    assert EXTENSION_K_GRID == (14, 16, 18, 20, 24, 27, 30)
    assert FULL_K_GRID == REUSED_K_GRID + EXTENSION_K_GRID
    assert PRIOR_CONTROL_K_GRID == (16, 20, 24, 30)


def test_normalized_prior_at_k30() -> None:
    assert prior_strength_for_k("normalized_6_over_k", 30) == 10000.0
    assert prior_strength_for_k("fixed", 30) == 50000.0


def test_k12_deltas_and_marginal_gains_are_seed_local() -> None:
    runs = pd.DataFrame([_row(6, -1000.0, 227), _row(12, -900.0, 527), _row(14, -880.0, 643)])
    values = _add_k12_deltas(runs)
    assert values.loc[values.K.eq(12), "dev_delta_log_likelihood_vs_k12"].item() == 0.0
    assert values.loc[values.K.eq(14), "dev_delta_log_likelihood_vs_k12"].item() == 20.0
    marginal = _marginal_gains(values)
    k14 = marginal[marginal.K.eq(14)].iloc[0]
    assert k14.previous_K == 12
    assert k14.added_parameters == 116
    assert np.isclose(k14.marginal_dev_ll_per_added_parameter, 20.0 / 116.0)


def test_marginal_aggregation_is_deterministic_under_row_shuffle() -> None:
    runs = pd.DataFrame([_row(12, -900.0, 527), _row(6, -1000.0, 227), _row(14, -880.0, 643)])
    first = _marginal_gains(_add_k12_deltas(runs)).sort_values("K").reset_index(drop=True)
    second = _marginal_gains(_add_k12_deltas(runs.sample(frac=1.0, random_state=3))).sort_values("K").reset_index(drop=True)
    pd.testing.assert_frame_equal(first, second)


def test_stale_reused_reference_columns_are_removed_before_recompute() -> None:
    frame = pd.DataFrame(
        {
            "K": [6],
            "reference_dev_log_likelihood_total": [-1.0],
            "dev_delta_log_likelihood_vs_k6": [0.0],
            "unrelated": [3],
        }
    )
    cleaned = _drop_stale_reference_columns(frame)
    assert list(cleaned.columns) == ["K", "unrelated"]


def test_initial_fit_cost_replaces_resumption_cost() -> None:
    current = pd.DataFrame(
        [{"architecture": "graph_full", "encoder_seed": 1, "hmm_seed": 0, "K": 14, "elapsed_s": 1.0, "peak_rss_mb": 10.0}]
    )
    original = current.copy()
    original["elapsed_s"] = 22.0
    original["peak_rss_mb"] = 333.0
    path = Path(__file__).with_name("_tmp_k30_initial_fit.csv")
    try:
        original.to_csv(path, index=False)
        restored = _restore_initial_fit_cost(current, path)
        assert restored.elapsed_s.item() == 22.0
        assert restored.peak_rss_mb.item() == 333.0
    finally:
        path.unlink(missing_ok=True)
