"""Independent-patch state models used as controls for temporal models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def _joined(sequences: Iterable[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(value, dtype=np.float64) for value in sequences]
    if not arrays or any(value.ndim != 2 or value.shape[0] == 0 for value in arrays):
        raise ValueError("state-model sequences must be nonempty [T,D] arrays")
    dimensions = {value.shape[1] for value in arrays}
    if len(dimensions) != 1:
        raise ValueError("all state-model sequences must have the same feature dimension")
    joined = np.concatenate(arrays, axis=0)
    if not np.isfinite(joined).all():
        raise ValueError("state-model inputs contain nonfinite values")
    return joined


class KMeansStateModel:
    """KMeans adapter with the same decode interface as HMM/HSMM.

    ``posterior`` returns hard one-hot assignments only. These are an interface
    convenience, not calibrated probabilities; benchmark outputs mark the
    absence of posterior uncertainty explicitly.
    """

    def __init__(self, n_states: int, *, seed: int = 0, n_init: int = 20) -> None:
        self.n_states = int(n_states)
        self.seed = int(seed)
        self.n_init = int(n_init)
        self.model_: KMeans | None = None
        self.converged_ = False
        self.n_iter_ = 0

    def fit(self, sequences: Iterable[np.ndarray]) -> "KMeansStateModel":
        values = _joined(sequences)
        self.model_ = KMeans(
            n_clusters=self.n_states,
            random_state=self.seed,
            n_init=self.n_init,
            algorithm="lloyd",
        ).fit(values)
        self.converged_ = True
        self.n_iter_ = int(self.model_.n_iter_)
        return self

    def _require_model(self) -> KMeans:
        if self.model_ is None:
            raise RuntimeError("KMeans model is not fitted")
        return self.model_

    def predict(self, sequence: np.ndarray) -> np.ndarray:
        return np.asarray(self._require_model().predict(sequence), dtype=np.int64)

    def posterior(self, sequence: np.ndarray) -> np.ndarray:
        state = self.predict(sequence)
        result = np.zeros((state.size, self.n_states), dtype=np.float64)
        result[np.arange(state.size), state] = 1.0
        return result

    def score(self, sequences: Iterable[np.ndarray]) -> float:
        model = self._require_model()
        return float(sum(model.score(np.asarray(value)) for value in sequences))

    def inertia(self, sequences: Iterable[np.ndarray]) -> float:
        return -self.score(sequences)

    def parameter_count(self) -> int:
        model = self._require_model()
        return int(model.cluster_centers_.size)

    def save(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> Path:
        model = self._require_model()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            model_type=np.asarray("kmeans_state_v1"),
            n_states=np.asarray(self.n_states),
            seed=np.asarray(self.seed),
            n_init=np.asarray(self.n_init),
            n_iter=np.asarray(self.n_iter_),
            cluster_centers=np.asarray(model.cluster_centers_, dtype=np.float64),
            metadata_json=np.asarray(json.dumps(metadata or {}, sort_keys=True)),
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> tuple["KMeansStateModel", dict[str, Any]]:
        with np.load(path, allow_pickle=False) as data:
            if str(data["model_type"].item()) != "kmeans_state_v1":
                raise ValueError("not a KMeansStateModel checkpoint")
            result = cls(
                int(data["n_states"].item()),
                seed=int(data["seed"].item()),
                n_init=int(data["n_init"].item()),
            )
            centers = np.asarray(data["cluster_centers"], dtype=np.float64)
            model = KMeans(n_clusters=result.n_states, random_state=result.seed)
            model.cluster_centers_ = centers
            model.n_features_in_ = centers.shape[1]
            model._n_threads = 1
            result.model_ = model
            result.n_iter_ = int(data["n_iter"].item())
            result.converged_ = True
            metadata = json.loads(str(data["metadata_json"].item()))
        return result, metadata


class GaussianMixtureStateModel:
    """Diagonal Gaussian-mixture adapter for independent patch clustering."""

    def __init__(
        self,
        n_states: int,
        *,
        seed: int = 0,
        covariance_floor: float = 1e-3,
        max_iter: int = 100,
        tolerance: float = 1e-4,
        n_init: int = 1,
    ) -> None:
        self.n_states = int(n_states)
        self.seed = int(seed)
        self.covariance_floor = float(covariance_floor)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.n_init = int(n_init)
        self.model_: GaussianMixture | None = None
        self.converged_ = False
        self.n_iter_ = 0

    def fit(self, sequences: Iterable[np.ndarray]) -> "GaussianMixtureStateModel":
        values = _joined(sequences)
        self.model_ = GaussianMixture(
            n_components=self.n_states,
            covariance_type="diag",
            reg_covar=self.covariance_floor,
            max_iter=self.max_iter,
            tol=self.tolerance,
            n_init=self.n_init,
            random_state=self.seed,
            init_params="kmeans",
        ).fit(values)
        self.converged_ = bool(self.model_.converged_)
        self.n_iter_ = int(self.model_.n_iter_)
        return self

    def _require_model(self) -> GaussianMixture:
        if self.model_ is None:
            raise RuntimeError("Gaussian mixture is not fitted")
        return self.model_

    def predict(self, sequence: np.ndarray) -> np.ndarray:
        return np.asarray(self._require_model().predict(sequence), dtype=np.int64)

    def posterior(self, sequence: np.ndarray) -> np.ndarray:
        return np.asarray(self._require_model().predict_proba(sequence), dtype=np.float64)

    def score(self, sequences: Iterable[np.ndarray]) -> float:
        model = self._require_model()
        return float(
            sum(model.score(np.asarray(value)) * len(value) for value in sequences)
        )

    def parameter_count(self) -> int:
        model = self._require_model()
        dimension = int(model.means_.shape[1])
        return int((self.n_states - 1) + 2 * self.n_states * dimension)

    def aic(self, sequences: Iterable[np.ndarray]) -> float:
        values = _joined(sequences)
        return float(self._require_model().aic(values))

    def bic(self, sequences: Iterable[np.ndarray]) -> float:
        values = _joined(sequences)
        return float(self._require_model().bic(values))

    def save(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> Path:
        model = self._require_model()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            model_type=np.asarray("diagonal_gmm_state_v1"),
            n_states=np.asarray(self.n_states),
            seed=np.asarray(self.seed),
            covariance_floor=np.asarray(self.covariance_floor),
            max_iter=np.asarray(self.max_iter),
            tolerance=np.asarray(self.tolerance),
            n_init=np.asarray(self.n_init),
            n_iter=np.asarray(self.n_iter_),
            converged=np.asarray(self.converged_),
            weights=np.asarray(model.weights_, dtype=np.float64),
            means=np.asarray(model.means_, dtype=np.float64),
            covariances=np.asarray(model.covariances_, dtype=np.float64),
            lower_bound=np.asarray(model.lower_bound_),
            metadata_json=np.asarray(json.dumps(metadata or {}, sort_keys=True)),
        )
        return output

    @classmethod
    def load(
        cls, path: str | Path
    ) -> tuple["GaussianMixtureStateModel", dict[str, Any]]:
        with np.load(path, allow_pickle=False) as data:
            if str(data["model_type"].item()) != "diagonal_gmm_state_v1":
                raise ValueError("not a GaussianMixtureStateModel checkpoint")
            result = cls(
                int(data["n_states"].item()),
                seed=int(data["seed"].item()),
                covariance_floor=float(data["covariance_floor"].item()),
                max_iter=int(data["max_iter"].item()),
                tolerance=float(data["tolerance"].item()),
                n_init=int(data["n_init"].item()),
            )
            model = GaussianMixture(
                n_components=result.n_states,
                covariance_type="diag",
                reg_covar=result.covariance_floor,
            )
            model.weights_ = np.asarray(data["weights"], dtype=np.float64)
            model.means_ = np.asarray(data["means"], dtype=np.float64)
            model.covariances_ = np.asarray(data["covariances"], dtype=np.float64)
            model.precisions_ = 1.0 / model.covariances_
            model.precisions_cholesky_ = np.sqrt(model.precisions_)
            model.n_features_in_ = model.means_.shape[1]
            model.converged_ = bool(data["converged"].item())
            model.n_iter_ = int(data["n_iter"].item())
            model.lower_bound_ = float(data["lower_bound"].item())
            result.model_ = model
            result.converged_ = model.converged_
            result.n_iter_ = model.n_iter_
            metadata = json.loads(str(data["metadata_json"].item()))
        return result, metadata
