"""Animal-level Social-GNN pipeline.

Public symbols are loaded lazily.  This keeps lightweight downstream analysis
of already-frozen embeddings independent from the PyTorch runtime while
preserving ``from social_gnn import ...``.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "SocialTrialDataset": (".data", "SocialTrialDataset"),
    "SocialTrialPackage": (".data", "SocialTrialPackage"),
    "SocialTrialSource": (".data", "SocialTrialSource"),
    "SocialTrialValidationError": (".data", "SocialTrialValidationError"),
    "build_social_dataloader": (".data", "build_social_dataloader"),
    "collate_social_trials": (".data", "collate_social_trials"),
    "complete_directed_edge_index": (".graph_builder", "complete_directed_edge_index"),
    "compose_edge_inputs": (".graph_builder", "compose_edge_inputs"),
    "HZipConversionError": (".h_zip_node_converter", "HZipConversionError"),
    "convert_h_zip_to_node_npz": (".h_zip_node_converter", "convert_h_zip_to_node_npz"),
    "LatentAnimalSource": (".latent_node_converter", "LatentAnimalSource"),
    "LatentNodeConversionError": (".latent_node_converter", "LatentNodeConversionError"),
    "convert_aligned_latents_to_node_npz": (
        ".latent_node_converter",
        "convert_aligned_latents_to_node_npz",
    ),
    "SocialV0": (".models", "SocialV0"),
    "MaskedSocialSSL": (".masked_social_ssl", "MaskedSocialSSL"),
    "MaskedSocialInputs": (".masked_social_ssl", "MaskedSocialInputs"),
    "SocialMaskConfig": (".masked_social_ssl", "SocialMaskConfig"),
    "apply_social_masks": (".masked_social_ssl", "apply_social_masks"),
    "masked_social_reconstruction_loss": (
        ".masked_social_ssl",
        "masked_social_reconstruction_loss",
    ),
    "GraphLevelSocialSSL": (".graph_level_ssl", "GraphLevelSocialSSL"),
    "GraphLevelTargets": (".graph_level_ssl", "GraphLevelTargets"),
    "MLPFusionEncoder": (".graph_level_ssl", "MLPFusionEncoder"),
    "SymmetrizedMLPFusionEncoder": (
        ".graph_level_ssl",
        "SymmetrizedMLPFusionEncoder",
    ),
    "swap_two_animal_graph": (".graph_level_ssl", "swap_two_animal_graph"),
    "graph_node_target_dim": (".graph_level_ssl", "graph_node_target_dim"),
    "graph_level_reconstruction_loss": (
        ".graph_level_ssl",
        "graph_level_reconstruction_loss",
    ),
    "graph_level_targets": (".graph_level_ssl", "graph_level_targets"),
    "DualBranchSocialEncoder": (".node_preservation_ssl", "DualBranchSocialEncoder"),
    "NodePreservingGraphSSL": (".node_preservation_ssl", "NodePreservingGraphSSL"),
    "NodeSetEncoder": (".node_preservation_ssl", "NodeSetEncoder"),
    "hybrid_reconstruction_loss": (
        ".node_preservation_ssl",
        "hybrid_reconstruction_loss",
    ),
    "node_preserving_reconstruction_loss": (
        ".node_preservation_ssl",
        "node_preserving_reconstruction_loss",
    ),
    "train_masked_social_ssl_epoch": (".training", "train_masked_social_ssl_epoch"),
    "EntryOrderEvidence": (".imu_video_identity", "EntryOrderEvidence"),
    "HorizontalAcceleration": (".imu_video_identity", "HorizontalAcceleration"),
    "MotionIdentityEvidence": (".imu_video_identity", "MotionIdentityEvidence"),
    "detect_impulse_event_times": (".imu_video_identity", "detect_impulse_event_times"),
    "estimate_horizontal_acceleration": (
        ".imu_video_identity",
        "estimate_horizontal_acceleration",
    ),
    "first_consecutive_valid_time": (
        ".imu_video_identity",
        "first_consecutive_valid_time",
    ),
    "match_entry_order": (".imu_video_identity", "match_entry_order"),
    "score_motion_identity_permutations": (
        ".imu_video_identity",
        "score_motion_identity_permutations",
    ),
    "trajectory_motion_signature": (
        ".imu_video_identity",
        "trajectory_motion_signature",
    ),
    "IMUAnimalSource": (".sequential_entry_identity", "IMUAnimalSource"),
    "IMUEntryEvent": (".sequential_entry_identity", "IMUEntryEvent"),
    "SequentialEntryIdentityError": (
        ".sequential_entry_identity",
        "SequentialEntryIdentityError",
    ),
    "SequentialEntryMapping": (".sequential_entry_identity", "SequentialEntryMapping"),
    "apply_sequential_entry_mapping": (
        ".sequential_entry_identity",
        "apply_sequential_entry_mapping",
    ),
    "detect_entry_event_from_imu": (
        ".sequential_entry_identity",
        "detect_entry_event_from_imu",
    ),
    "infer_sequential_entry_mapping": (
        ".sequential_entry_identity",
        "infer_sequential_entry_mapping",
    ),
    "resolve_entry_events": (".sequential_entry_identity", "resolve_entry_events"),
    "run_sequential_entry_identity_mapping": (
        ".sequential_entry_identity",
        "run_sequential_entry_identity_mapping",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
