"""Procedural lock for the 9D confirmatory holdout trial.

This is intentionally a guardrail rather than a security boundary.  Its goal
is to make any release of A5_C5 into 9D evaluation explicit and reviewable.
Historical 6D experiments remain reproducible and are not covered by this
lock because their upstream representation is different.
"""

from __future__ import annotations

from collections.abc import Iterable


LOCKED_9D_CONFIRMATORY_TRIALS = frozenset({"A5_C5"})
HOLDOUT_RELEASE_TOKEN = "RELEASE_A5_C5_9D_CONFIRMATORY_HOLDOUT"


class LockedHoldoutError(ValueError):
    """Raised when locked 9D holdout data are requested prematurely."""


def assert_9d_holdout_access(
    trial_ids: Iterable[str],
    *,
    node_dim: int,
    release_token: str | None,
    context: str,
) -> tuple[str, ...]:
    """Reject access to a locked 9D holdout unless explicitly released."""

    requested = tuple(str(value) for value in trial_ids)
    locked = sorted(set(requested) & LOCKED_9D_CONFIRMATORY_TRIALS)
    if int(node_dim) == 9 and locked and release_token != HOLDOUT_RELEASE_TOKEN:
        raise LockedHoldoutError(
            f"{context}: locked 9D confirmatory holdout requested: {locked}. "
            "Finish model/K/M/seed selection on A4_C4, freeze hashes and the "
            "evaluation protocol, then provide the explicit holdout release token."
        )
    return requested


def assert_9d_confirmatory_split(
    *,
    training_trial_ids: Iterable[str],
    validation_trial_ids: Iterable[str],
    test_trial_ids: Iterable[str],
    node_dim: int,
    release_token: str | None,
    context: str,
) -> tuple[str, ...]:
    """Enforce that a released 9D holdout appears only in the test split."""

    training = tuple(str(value) for value in training_trial_ids)
    validation = tuple(str(value) for value in validation_trial_ids)
    test = tuple(str(value) for value in test_trial_ids)
    active = (*training, *validation, *test)
    assert_9d_holdout_access(
        active,
        node_dim=node_dim,
        release_token=release_token,
        context=context,
    )
    if int(node_dim) != 9 or release_token != HOLDOUT_RELEASE_TOKEN:
        return active
    invalid_roles = (
        set(training) | set(validation)
    ) & LOCKED_9D_CONFIRMATORY_TRIALS
    if invalid_roles:
        raise LockedHoldoutError(
            f"{context}: a released 9D confirmatory holdout may be used only "
            "as test data, not train/validation: "
            f"{sorted(invalid_roles)}"
        )
    return active


def default_embedding_trial_ids(
    all_trial_ids: Iterable[str],
    *,
    node_dim: int,
    excluded_trial_ids: Iterable[str],
) -> tuple[str, ...]:
    """Return safe default exports without silently releasing a 9D holdout."""

    excluded = {str(value) for value in excluded_trial_ids}
    if int(node_dim) == 9:
        excluded |= set(LOCKED_9D_CONFIRMATORY_TRIALS)
    return tuple(str(value) for value in all_trial_ids if str(value) not in excluded)
