"""Small training-loop helpers for the two patch-level SSL experiments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch

from .masked_social_ssl import (
    MaskedSocialSSL,
    SocialMaskConfig,
    masked_social_reconstruction_loss,
)
from .vae import SocialPatchVAE, social_patch_vae_loss


def _to_device(batch: Mapping[str, Any], device: torch.device | str) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _average_metrics(total: dict[str, float], count: int) -> dict[str, float]:
    if count == 0:
        raise ValueError("the dataloader produced no batches")
    return {key: value / count for key, value in total.items()}


def train_masked_social_ssl_epoch(
    model: MaskedSocialSSL,
    dataloader: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str = "cpu",
    mask_config: SocialMaskConfig | None = None,
) -> dict[str, float]:
    """Train one epoch of patch-local masked multimodal reconstruction."""

    model.train()
    totals = {"total": 0.0, "node": 0.0, "edge": 0.0}
    count = 0
    for raw_batch in dataloader:
        batch = _to_device(raw_batch, device)
        output = model(batch, mask_config=mask_config)
        losses = masked_social_reconstruction_loss(output, batch)
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        optimizer.step()
        for key in totals:
            totals[key] += float(losses[key].detach().cpu())
        count += 1
    return _average_metrics(totals, count)


def train_social_patch_vae_epoch(
    model: SocialPatchVAE,
    dataloader: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str = "cpu",
    kl_beta: float = 1e-3,
) -> dict[str, float]:
    """Train one epoch of the independent patch-level VAE baseline."""

    model.train()
    totals = {"total": 0.0, "reconstruction": 0.0, "node": 0.0, "edge": 0.0, "kl": 0.0}
    count = 0
    for raw_batch in dataloader:
        batch = _to_device(raw_batch, device)
        output = model(batch)
        losses = social_patch_vae_loss(output, batch, kl_beta=kl_beta)
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        optimizer.step()
        for key in totals:
            totals[key] += float(losses[key].detach().cpu())
        count += 1
    return _average_metrics(totals, count)
