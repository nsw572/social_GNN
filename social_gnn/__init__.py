"""Animal-level Social-GNN pipeline.

The first version intentionally has no PyTorch Geometric runtime dependency.
Its tensor contract is compatible with a later PyG-backed implementation.
"""

from .data import (
    SocialTrialDataset,
    SocialTrialPackage,
    SocialTrialSource,
    SocialTrialValidationError,
    build_social_dataloader,
    collate_social_trials,
)
from .graph_builder import complete_directed_edge_index, compose_edge_inputs
from .models import SocialGNNWithTCN, SocialV0
from .temporal import TemporalConvNet

__all__ = [
    "SocialGNNWithTCN",
    "SocialTrialDataset",
    "SocialTrialPackage",
    "SocialTrialSource",
    "SocialTrialValidationError",
    "SocialV0",
    "TemporalConvNet",
    "build_social_dataloader",
    "collate_social_trials",
    "complete_directed_edge_index",
    "compose_edge_inputs",
]
