"""Animal-level Social-GNN pipeline.

The first version intentionally has no PyTorch Geometric runtime dependency.
Its tensor contract is compatible with a later PyG-backed implementation.
"""

from .graph_builder import complete_directed_edge_index, compose_edge_inputs
from .models import SocialV0

__all__ = ["SocialV0", "complete_directed_edge_index", "compose_edge_inputs"]
