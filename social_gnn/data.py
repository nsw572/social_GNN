"""Validated full-trial packages and variable-length PyTorch batching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SocialTrialValidationError(ValueError):
    """Raised when one trial violates the Social-GNN data contract."""


def _fail(trial_id: str, message: str) -> None:
    raise SocialTrialValidationError(f"Trial {trial_id!r}: {message}")


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _freeze_array(value: Any) -> np.ndarray:
    array = np.array(value, copy=True, order="C")
    array.setflags(write=False)
    return array


def _validate_finite(trial_id: str, name: str, array: np.ndarray) -> None:
    if not np.issubdtype(array.dtype, np.number):
        _fail(trial_id, f"{name} must be numeric, got dtype {array.dtype}")
    invalid = np.argwhere(~np.isfinite(array))
    if invalid.size:
        _fail(
            trial_id,
            f"{name} contains a non-finite value at index {tuple(invalid[0])}",
        )


def _normalise_node_mask(trial_id: str, value: Any) -> np.ndarray:
    mask = np.asarray(value)
    if mask.dtype == np.bool_:
        return _freeze_array(mask)
    if not np.issubdtype(mask.dtype, np.number):
        _fail(trial_id, f"node_mask must be boolean or binary, got {mask.dtype}")
    _validate_finite(trial_id, "node_mask", mask)
    if not np.all((mask == 0) | (mask == 1)):
        bad = tuple(np.argwhere((mask != 0) & (mask != 1))[0])
        _fail(trial_id, f"node_mask must contain only 0/1; invalid value at {bad}")
    return _freeze_array(mask.astype(bool, copy=False))


def _identity_values(identity: np.ndarray) -> tuple[Any, ...]:
    return tuple(_python_scalar(value) for value in identity.tolist())


@dataclass(frozen=True)
class SocialTrialPackage:
    """One immutable, continuous and fully aligned Social-GNN trial."""

    trial_id: str
    node_features: np.ndarray
    node_mask: np.ndarray
    edge_values: np.ndarray
    edge_confidence: np.ndarray
    patch_start_s: np.ndarray
    patch_end_s: np.ndarray
    identity: np.ndarray

    def __post_init__(self) -> None:
        trial_id = str(self.trial_id).strip()
        if not trial_id:
            raise SocialTrialValidationError("Trial ID must be a non-empty string")
        object.__setattr__(self, "trial_id", trial_id)

        node_features = _freeze_array(self.node_features)
        node_mask = _normalise_node_mask(trial_id, self.node_mask)
        edge_values = _freeze_array(self.edge_values)
        edge_confidence = _freeze_array(self.edge_confidence)
        patch_start_s = _freeze_array(self.patch_start_s)
        patch_end_s = _freeze_array(self.patch_end_s)
        identity = _freeze_array(self.identity)

        if node_features.ndim != 3:
            _fail(
                trial_id,
                f"node_features must have shape [T,N,D_node], got {node_features.shape}",
            )
        timesteps, animals, node_dim = node_features.shape
        if timesteps < 1 or animals < 1 or node_dim < 1:
            _fail(trial_id, "node_features dimensions T, N and D_node must be positive")
        if node_mask.shape != (timesteps, animals):
            _fail(
                trial_id,
                f"node_mask must have shape {(timesteps, animals)}, got {node_mask.shape}",
            )
        expected_edge_shape = (timesteps, animals, animals, 8)
        if edge_values.shape != expected_edge_shape:
            _fail(
                trial_id,
                f"edge_values must have shape {expected_edge_shape}, got {edge_values.shape}",
            )
        if edge_confidence.shape != expected_edge_shape:
            _fail(
                trial_id,
                "edge_confidence must have shape "
                f"{expected_edge_shape}, got {edge_confidence.shape}",
            )
        if patch_start_s.shape != (timesteps,):
            _fail(
                trial_id,
                f"patch_start_s must have shape {(timesteps,)}, got {patch_start_s.shape}",
            )
        if patch_end_s.shape != (timesteps,):
            _fail(
                trial_id,
                f"patch_end_s must have shape {(timesteps,)}, got {patch_end_s.shape}",
            )
        if identity.shape != (animals,):
            _fail(
                trial_id,
                f"identity must have shape {(animals,)}, got {identity.shape}",
            )
        identity_tuple = _identity_values(identity)
        if len(set(identity_tuple)) != animals:
            _fail(trial_id, f"identity values must be unique and ordered, got {identity_tuple}")

        for name, array in (
            ("node_features", node_features),
            ("edge_values", edge_values),
            ("edge_confidence", edge_confidence),
            ("patch_start_s", patch_start_s),
            ("patch_end_s", patch_end_s),
        ):
            _validate_finite(trial_id, name, array)
        invalid_confidence = np.argwhere(
            (edge_confidence < 0.0) | (edge_confidence > 1.0)
        )
        if invalid_confidence.size:
            index = tuple(invalid_confidence[0])
            _fail(
                trial_id,
                "edge_confidence must be in [0,1]; "
                f"found {edge_confidence[index]!r} at index {index}",
            )
        invalid_duration = np.argwhere(patch_end_s <= patch_start_s)
        if invalid_duration.size:
            index = int(invalid_duration[0, 0])
            _fail(
                trial_id,
                "each patch must have positive duration; "
                f"patch {index} is [{patch_start_s[index]}, {patch_end_s[index]})",
            )
        if timesteps > 1 and np.any(np.diff(patch_start_s) <= 0):
            index = int(np.argwhere(np.diff(patch_start_s) <= 0)[0, 0] + 1)
            _fail(trial_id, f"patch_start_s must be strictly increasing at index {index}")
        if timesteps > 1 and np.any(np.diff(patch_end_s) <= 0):
            index = int(np.argwhere(np.diff(patch_end_s) <= 0)[0, 0] + 1)
            _fail(trial_id, f"patch_end_s must be strictly increasing at index {index}")

        object.__setattr__(self, "node_features", node_features)
        object.__setattr__(self, "node_mask", node_mask)
        object.__setattr__(self, "edge_values", edge_values)
        object.__setattr__(self, "edge_confidence", edge_confidence)
        object.__setattr__(self, "patch_start_s", patch_start_s)
        object.__setattr__(self, "patch_end_s", patch_end_s)
        object.__setattr__(self, "identity", identity)

    @property
    def timesteps(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def animals(self) -> int:
        return int(self.node_features.shape[1])

    @property
    def node_dim(self) -> int:
        return int(self.node_features.shape[2])

    def to_torch_sample(self) -> dict[str, Any]:
        """Return fresh tensors so dynamic augmentation cannot mutate storage."""
        return {
            "node_features": torch.tensor(self.node_features, dtype=torch.float32),
            "node_mask": torch.tensor(self.node_mask, dtype=torch.bool),
            "edge_values": torch.tensor(self.edge_values, dtype=torch.float32),
            "edge_confidence": torch.tensor(
                self.edge_confidence, dtype=torch.float32
            ),
            "patch_start_s": torch.tensor(self.patch_start_s, dtype=torch.float64),
            "patch_end_s": torch.tensor(self.patch_end_s, dtype=torch.float64),
            "identity": _identity_values(self.identity),
            "trial_id": self.trial_id,
        }

    @classmethod
    def from_npz(
        cls,
        node_path: str | Path,
        edge_path: str | Path,
        *,
        trial_id: str | None = None,
    ) -> "SocialTrialPackage":
        """Load and strictly align a node NPZ with an edge-extractor NPZ."""
        node_path = Path(node_path)
        edge_path = Path(edge_path)
        provisional_id = str(trial_id or _infer_trial_id(node_path))
        try:
            with np.load(node_path, allow_pickle=False) as node_pack, np.load(
                edge_path, allow_pickle=False
            ) as edge_pack:
                resolved_id = _resolve_trial_id(
                    provisional_id, trial_id, node_pack, edge_pack
                )
                node_identity = _required_array(
                    node_pack, "identity", resolved_id, node_path
                )
                edge_identity = _required_array(
                    edge_pack, "identity", resolved_id, edge_path
                )
                node_start = _required_array(
                    node_pack, "patch_start_s", resolved_id, node_path
                )
                edge_start = _required_array(
                    edge_pack, "patch_start_s", resolved_id, edge_path
                )
                node_end = _required_array(
                    node_pack, "patch_end_s", resolved_id, node_path
                )
                edge_end = _required_array(
                    edge_pack, "patch_end_s", resolved_id, edge_path
                )
                _require_exact_alignment(
                    resolved_id, "identity", node_identity, edge_identity
                )
                _require_exact_alignment(
                    resolved_id, "patch_start_s", node_start, edge_start
                )
                _require_exact_alignment(
                    resolved_id, "patch_end_s", node_end, edge_end
                )
                return cls(
                    trial_id=resolved_id,
                    node_features=_required_array(
                        node_pack, "node_features", resolved_id, node_path
                    ),
                    node_mask=_required_array(
                        node_pack, "node_mask", resolved_id, node_path
                    ),
                    edge_values=_array_alias(
                        edge_pack,
                        ("edge_values", "edge_value_dense"),
                        "edge_values",
                        resolved_id,
                        edge_path,
                    ),
                    edge_confidence=_array_alias(
                        edge_pack,
                        ("edge_confidence", "edge_confidence_dense"),
                        "edge_confidence",
                        resolved_id,
                        edge_path,
                    ),
                    patch_start_s=node_start,
                    patch_end_s=node_end,
                    identity=node_identity,
                )
        except SocialTrialValidationError:
            raise
        except (OSError, ValueError) as exc:
            _fail(
                provisional_id,
                f"could not read node/edge NPZ files: {exc}",
            )
        raise AssertionError("unreachable")


def _infer_trial_id(node_path: Path) -> str:
    stem = node_path.stem
    for suffix in ("_node_features", "_nodes", "_node"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _optional_trial_id(pack: np.lib.npyio.NpzFile) -> str | None:
    if "trial_id" not in pack.files:
        return None
    value = np.asarray(pack["trial_id"])
    if value.size != 1:
        return None
    scalar = _python_scalar(value.reshape(-1)[0])
    if isinstance(scalar, bytes):
        scalar = scalar.decode("utf-8")
    return str(scalar)


def _resolve_trial_id(
    provisional_id: str,
    explicit_id: str | None,
    node_pack: np.lib.npyio.NpzFile,
    edge_pack: np.lib.npyio.NpzFile,
) -> str:
    node_id = _optional_trial_id(node_pack)
    edge_id = _optional_trial_id(edge_pack)
    resolved = str(explicit_id or node_id or edge_id or provisional_id)
    for source, value in (("node", node_id), ("edge", edge_id)):
        if value is not None and value != resolved:
            _fail(
                resolved,
                f"{source} package trial_id {value!r} does not match {resolved!r}",
            )
    return resolved


def _required_array(
    pack: np.lib.npyio.NpzFile,
    key: str,
    trial_id: str,
    path: Path,
) -> np.ndarray:
    if key not in pack.files:
        _fail(trial_id, f"{path.name} is missing required array {key!r}")
    return np.asarray(pack[key])


def _array_alias(
    pack: np.lib.npyio.NpzFile,
    keys: tuple[str, ...],
    canonical_name: str,
    trial_id: str,
    path: Path,
) -> np.ndarray:
    present = [key for key in keys if key in pack.files]
    if not present:
        _fail(
            trial_id,
            f"{path.name} is missing {canonical_name!r}; accepted keys are {keys}",
        )
    first = np.asarray(pack[present[0]])
    for key in present[1:]:
        other = np.asarray(pack[key])
        _require_exact_alignment(
            trial_id, f"duplicate {canonical_name} arrays", first, other
        )
    return first


def _require_exact_alignment(
    trial_id: str, name: str, node_value: np.ndarray, edge_value: np.ndarray
) -> None:
    left = np.asarray(node_value)
    right = np.asarray(edge_value)
    if left.shape != right.shape:
        _fail(
            trial_id,
            f"node-edge {name} shape mismatch: node {left.shape}, edge {right.shape}",
        )
    if np.array_equal(left, right):
        return
    unequal = np.argwhere(left != right)
    if unequal.size:
        index = tuple(unequal[0])
        _fail(
            trial_id,
            f"node-edge {name} mismatch at index {index}: "
            f"node={left[index]!r}, edge={right[index]!r}",
        )
    _fail(trial_id, f"node-edge {name} mismatch")


@dataclass(frozen=True)
class SocialTrialSource:
    """Paths used to load one full trial lazily inside a Dataset worker."""

    trial_id: str
    node_path: Path | str
    edge_path: Path | str

    def __post_init__(self) -> None:
        trial_id = str(self.trial_id).strip()
        if not trial_id:
            raise SocialTrialValidationError("Trial ID must be a non-empty string")
        object.__setattr__(self, "trial_id", trial_id)
        object.__setattr__(self, "node_path", Path(self.node_path))
        object.__setattr__(self, "edge_path", Path(self.edge_path))

    def load(self) -> SocialTrialPackage:
        return SocialTrialPackage.from_npz(
            self.node_path, self.edge_path, trial_id=self.trial_id
        )


class SocialTrialDataset(Dataset[dict[str, Any]]):
    """Dataset where one item is one complete, never-window-shuffled trial."""

    def __init__(
        self, trials: Sequence[SocialTrialPackage | SocialTrialSource]
    ) -> None:
        self._trials = tuple(trials)
        identifiers = [trial.trial_id for trial in self._trials]
        if len(set(identifiers)) != len(identifiers):
            raise SocialTrialValidationError(
                f"Dataset trial_id values must be unique, got {identifiers}"
            )

    def __len__(self) -> int:
        return len(self._trials)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self._trials[index]
        package = record.load() if isinstance(record, SocialTrialSource) else record
        return package.to_torch_sample()


def collate_social_trials(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pad full trials along T only; never merge or reorder their time axes."""
    if not samples:
        raise ValueError("collate_social_trials requires at least one sample")
    first = samples[0]
    node_dim = int(first["node_features"].shape[2])
    animals = int(first["node_features"].shape[1])
    max_timesteps = max(int(sample["node_features"].shape[0]) for sample in samples)
    batch_size = len(samples)

    node_features = torch.zeros(
        batch_size, max_timesteps, animals, node_dim, dtype=torch.float32
    )
    node_mask = torch.zeros(
        batch_size, max_timesteps, animals, dtype=torch.bool
    )
    edge_values = torch.zeros(
        batch_size, max_timesteps, animals, animals, 8, dtype=torch.float32
    )
    edge_confidence = torch.zeros_like(edge_values)
    time_mask = torch.zeros(batch_size, max_timesteps, dtype=torch.bool)
    patch_start_s = torch.zeros(batch_size, max_timesteps, dtype=torch.float64)
    patch_end_s = torch.zeros_like(patch_start_s)
    sequence_length = torch.zeros(batch_size, dtype=torch.long)
    identities: list[tuple[Any, ...]] = []
    trial_ids: list[str] = []

    for batch_index, sample in enumerate(samples):
        trial_id = str(sample["trial_id"])
        nodes = sample["node_features"]
        length = int(nodes.shape[0])
        expected_node_shape = (length, animals, node_dim)
        expected_edge_shape = (length, animals, animals, 8)
        if tuple(nodes.shape) != expected_node_shape:
            _fail(
                trial_id,
                "all trials in one batch must share N and D_node; "
                f"expected {expected_node_shape}, got {tuple(nodes.shape)}",
            )
        for key, expected in (
            ("node_mask", (length, animals)),
            ("edge_values", expected_edge_shape),
            ("edge_confidence", expected_edge_shape),
            ("patch_start_s", (length,)),
            ("patch_end_s", (length,)),
        ):
            if tuple(sample[key].shape) != expected:
                _fail(
                    trial_id,
                    f"batch field {key} expected {expected}, "
                    f"got {tuple(sample[key].shape)}",
                )

        node_features[batch_index, :length] = nodes
        node_mask[batch_index, :length] = sample["node_mask"]
        edge_values[batch_index, :length] = sample["edge_values"]
        edge_confidence[batch_index, :length] = sample["edge_confidence"]
        patch_start_s[batch_index, :length] = sample["patch_start_s"]
        patch_end_s[batch_index, :length] = sample["patch_end_s"]
        time_mask[batch_index, :length] = True
        sequence_length[batch_index] = length
        identities.append(tuple(sample["identity"]))
        trial_ids.append(trial_id)

    return {
        "node_features": node_features,
        "node_mask": node_mask,
        "edge_values": edge_values,
        "edge_confidence": edge_confidence,
        "time_mask": time_mask,
        "patch_start_s": patch_start_s,
        "patch_end_s": patch_end_s,
        "identity": identities,
        "trial_id": trial_ids,
        "sequence_length": sequence_length,
    }


def build_social_dataloader(
    dataset: SocialTrialDataset,
    *,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """Create a DataLoader whose shuffle unit is always one complete trial."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_social_trials,
        persistent_workers=num_workers > 0,
    )
