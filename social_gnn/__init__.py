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
from .h_zip_node_converter import HZipConversionError, convert_h_zip_to_node_npz
from .latent_node_converter import (
    LatentAnimalSource,
    LatentNodeConversionError,
    convert_aligned_latents_to_node_npz,
)
from .models import SocialV0
from .masked_social_ssl import (
    MaskedSocialSSL,
    MaskedSocialInputs,
    SocialMaskConfig,
    apply_social_masks,
    masked_social_reconstruction_loss,
)
from .vae import SocialPatchVAE, social_patch_vae_loss
from .training import train_masked_social_ssl_epoch, train_social_patch_vae_epoch
from .imu_video_identity import (
    EntryOrderEvidence,
    HorizontalAcceleration,
    MotionIdentityEvidence,
    detect_impulse_event_times,
    estimate_horizontal_acceleration,
    first_consecutive_valid_time,
    match_entry_order,
    score_motion_identity_permutations,
    trajectory_motion_signature,
)
from .sequential_entry_identity import (
    IMUAnimalSource,
    IMUEntryEvent,
    SequentialEntryIdentityError,
    SequentialEntryMapping,
    apply_sequential_entry_mapping,
    detect_entry_event_from_imu,
    infer_sequential_entry_mapping,
    resolve_entry_events,
    run_sequential_entry_identity_mapping,
)

__all__ = [
    "HZipConversionError",
    "LatentAnimalSource",
    "LatentNodeConversionError",
    "SocialTrialDataset",
    "SocialTrialPackage",
    "SocialTrialSource",
    "SocialTrialValidationError",
    "SocialV0",
    "MaskedSocialSSL",
    "MaskedSocialInputs",
    "SocialMaskConfig",
    "apply_social_masks",
    "masked_social_reconstruction_loss",
    "SocialPatchVAE",
    "social_patch_vae_loss",
    "train_masked_social_ssl_epoch",
    "train_social_patch_vae_epoch",
    "EntryOrderEvidence",
    "HorizontalAcceleration",
    "MotionIdentityEvidence",
    "detect_impulse_event_times",
    "estimate_horizontal_acceleration",
    "first_consecutive_valid_time",
    "match_entry_order",
    "score_motion_identity_permutations",
    "trajectory_motion_signature",
    "IMUAnimalSource",
    "IMUEntryEvent",
    "SequentialEntryIdentityError",
    "SequentialEntryMapping",
    "apply_sequential_entry_mapping",
    "detect_entry_event_from_imu",
    "infer_sequential_entry_mapping",
    "resolve_entry_events",
    "run_sequential_entry_identity_mapping",
    "build_social_dataloader",
    "collate_social_trials",
    "complete_directed_edge_index",
    "compose_edge_inputs",
    "convert_h_zip_to_node_npz",
    "convert_aligned_latents_to_node_npz",
]
