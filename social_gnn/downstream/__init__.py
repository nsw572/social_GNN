"""Probabilistic temporal models for frozen Social-GNN embeddings.

Imports are lazy so the optional rSLDS environment does not need the HMM/HSMM
runtime (notably numba), and the ordinary downstream environment does not need
the optional Linderman ``ssm`` backend.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "EmbeddingPreprocessor": (".preprocessing", "EmbeddingPreprocessor"),
    "FrozenEmbeddingTrial": (".embeddings", "FrozenEmbeddingTrial"),
    "GaussianHMM": (".gaussian_hmm", "GaussianHMM"),
    "GaussianHSMM": (".gaussian_hsmm", "GaussianHSMM"),
    "GaussianMixtureStateModel": (".independent", "GaussianMixtureStateModel"),
    "KMeansStateModel": (".independent", "KMeansStateModel"),
    "RSLDSPosterior": (".rslds", "RSLDSPosterior"),
    "RSLDSSequence": (".rslds", "RSLDSSequence"),
    "SSMBackendUnavailable": (".rslds", "SSMBackendUnavailable"),
    "SSMRecurrentSLDS": (".rslds", "SSMRecurrentSLDS"),
    "build_rslds_sequences": (".rslds", "build_rslds_sequences"),
    "extract_frozen_embeddings": (".embeddings", "extract_frozen_embeddings"),
    "recurrent_transition_probabilities": (
        ".rslds",
        "recurrent_transition_probabilities",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
