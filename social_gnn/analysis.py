"""Downstream clustering helpers for Social-V0 embeddings."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def flatten_valid_embeddings(embeddings: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Flatten [B,T,D] embeddings, optionally excluding padded windows."""
    values = np.asarray(embeddings)
    if values.ndim != 3:
        raise ValueError("embeddings must have shape [B,T,D]")
    flat = values.reshape(-1, values.shape[-1])
    if valid_mask is None:
        return flat
    mask = np.asarray(valid_mask, dtype=bool)
    if mask.shape != values.shape[:2]:
        raise ValueError("valid_mask must have shape [B,T]")
    return flat[mask.reshape(-1)]


def kmeans_embeddings(embeddings: np.ndarray, n_clusters: int, random_state: int = 0) -> tuple[np.ndarray, KMeans]:
    values = flatten_valid_embeddings(embeddings)
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    return model.fit_predict(values), model


def gmm_embeddings(embeddings: np.ndarray, n_components: int, random_state: int = 0) -> tuple[np.ndarray, GaussianMixture]:
    values = flatten_valid_embeddings(embeddings)
    model = GaussianMixture(n_components=n_components, random_state=random_state)
    return model.fit_predict(values), model
