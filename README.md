# Social-GNN

Animal-level Social-GNN code for combining continuous per-animal IMU latents
with directed, video-derived social relations.

The implementation includes parallel MouseGPT/idtracker.ai orchestration,
frame-wise identity matching, optional idtracker.ai trajectory repair,
confidence-aware fixed-clock edge extraction, and a minimal edge-aware GNN.

See [social_gnn/README.md](social_gnn/README.md) for the data contracts, pipeline
configuration, and usage notes.
