"""Train-only standardization and optional PCA for temporal models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.decomposition import PCA

from .embeddings import FrozenEmbeddingTrial


@dataclass
class EmbeddingPreprocessor:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    pca_components_: np.ndarray | None = None
    pca_mean_: np.ndarray | None = None
    explained_variance_ratio_: np.ndarray | None = None
    epsilon: float = 1e-6

    @property
    def output_dim(self) -> int:
        if self.mean_ is None:
            raise RuntimeError("preprocessor is not fitted")
        if self.pca_components_ is None:
            return int(self.mean_.shape[0])
        return int(self.pca_components_.shape[0])

    def fit(
        self,
        trials: Iterable[FrozenEmbeddingTrial],
        *,
        pca_dim: int | None = None,
        seed: int = 0,
    ) -> "EmbeddingPreprocessor":
        arrays = [
            trial.graph_embeddings[trial.valid_time_mask].astype(np.float64)
            for trial in trials
        ]
        if not arrays or any(value.shape[0] == 0 for value in arrays):
            raise ValueError("every training trial must contain valid embeddings")
        joined = np.concatenate(arrays, axis=0)
        self.mean_ = joined.mean(axis=0)
        scale = joined.std(axis=0)
        self.scale_ = np.maximum(scale, self.epsilon)
        standardized = (joined - self.mean_) / self.scale_
        if pca_dim is None or pca_dim == standardized.shape[1]:
            self.pca_components_ = None
            self.pca_mean_ = None
            self.explained_variance_ratio_ = None
        else:
            if not 1 <= pca_dim < standardized.shape[1]:
                raise ValueError(
                    f"pca_dim must be in [1,{standardized.shape[1] - 1}] or omitted"
                )
            pca = PCA(n_components=pca_dim, svd_solver="randomized", random_state=seed)
            pca.fit(standardized)
            self.pca_components_ = np.asarray(pca.components_, dtype=np.float64)
            self.pca_mean_ = np.asarray(pca.mean_, dtype=np.float64)
            self.explained_variance_ratio_ = np.asarray(
                pca.explained_variance_ratio_, dtype=np.float64
            )
        return self

    def transform_array(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("preprocessor is not fitted")
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.mean_.shape[0]:
            raise ValueError(
                f"expected [T,{self.mean_.shape[0]}], got {array.shape}"
            )
        transformed = (array - self.mean_) / self.scale_
        if self.pca_components_ is not None:
            transformed = (transformed - self.pca_mean_) @ self.pca_components_.T
        if not np.isfinite(transformed).all():
            raise ValueError("preprocessing produced nonfinite values")
        return transformed

    def transform_trial_sequences(
        self, trial: FrozenEmbeddingTrial
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        raw, indices = trial.model_sequences()
        return [self.transform_array(value) for value in raw], indices

    def to_arrays(self) -> dict[str, np.ndarray]:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("preprocessor is not fitted")
        return {
            "preprocess_mean": self.mean_,
            "preprocess_scale": self.scale_,
            "pca_components": np.empty((0, self.mean_.size))
            if self.pca_components_ is None
            else self.pca_components_,
            "pca_mean": np.empty((0,))
            if self.pca_mean_ is None
            else self.pca_mean_,
            "explained_variance_ratio": np.empty((0,))
            if self.explained_variance_ratio_ is None
            else self.explained_variance_ratio_,
        }

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **self.to_arrays())
        return output

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray]) -> "EmbeddingPreprocessor":
        components = np.asarray(arrays["pca_components"], dtype=np.float64)
        pca_mean = np.asarray(arrays["pca_mean"], dtype=np.float64)
        ratio = np.asarray(arrays["explained_variance_ratio"], dtype=np.float64)
        return cls(
            mean_=np.asarray(arrays["preprocess_mean"], dtype=np.float64),
            scale_=np.asarray(arrays["preprocess_scale"], dtype=np.float64),
            pca_components_=None if components.shape[0] == 0 else components,
            pca_mean_=None if pca_mean.size == 0 else pca_mean,
            explained_variance_ratio_=None if ratio.size == 0 else ratio,
        )

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingPreprocessor":
        with np.load(path, allow_pickle=False) as data:
            return cls.from_arrays({key: np.asarray(data[key]) for key in data.files})
