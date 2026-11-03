import pytest

from social_gnn.holdout import (
    HOLDOUT_RELEASE_TOKEN,
    LockedHoldoutError,
    assert_9d_confirmatory_split,
    assert_9d_holdout_access,
    default_embedding_trial_ids,
)


def test_9d_a5_is_locked_without_explicit_release() -> None:
    with pytest.raises(LockedHoldoutError, match="locked 9D confirmatory holdout"):
        assert_9d_holdout_access(
            ["A5_C5"], node_dim=9, release_token=None, context="test"
        )


def test_9d_a5_accepts_exact_release_token() -> None:
    assert assert_9d_holdout_access(
        ["A5_C5"],
        node_dim=9,
        release_token=HOLDOUT_RELEASE_TOKEN,
        context="test",
    ) == ("A5_C5",)


def test_historical_6d_a5_is_not_retroactively_blocked() -> None:
    assert assert_9d_holdout_access(
        ["A5_C5"], node_dim=6, release_token=None, context="historical"
    ) == ("A5_C5",)


def test_default_9d_embedding_export_excludes_a5() -> None:
    assert default_embedding_trial_ids(
        ["A1_C1", "A4_C4", "A5_C5"],
        node_dim=9,
        excluded_trial_ids=[],
    ) == ("A1_C1", "A4_C4")


def test_wrong_release_token_does_not_unlock_a5() -> None:
    with pytest.raises(LockedHoldoutError, match="locked 9D confirmatory holdout"):
        assert_9d_holdout_access(
            ["A5_C5"],
            node_dim=9,
            release_token="yes-please",
            context="test",
        )


def test_release_allows_a5_only_as_test_data() -> None:
    assert assert_9d_confirmatory_split(
        training_trial_ids=["A1_C1"],
        validation_trial_ids=["A4_C4"],
        test_trial_ids=["A5_C5"],
        node_dim=9,
        release_token=HOLDOUT_RELEASE_TOKEN,
        context="test",
    ) == ("A1_C1", "A4_C4", "A5_C5")


def test_release_does_not_allow_a5_in_training() -> None:
    with pytest.raises(LockedHoldoutError, match="only as test data"):
        assert_9d_confirmatory_split(
            training_trial_ids=["A5_C5"],
            validation_trial_ids=["A4_C4"],
            test_trial_ids=[],
            node_dim=9,
            release_token=HOLDOUT_RELEASE_TOKEN,
            context="test",
        )
