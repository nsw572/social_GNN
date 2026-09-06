"""Causal temporal convolution for social graph embeddings."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _masked_channels(sequence: torch.Tensor, time_mask: torch.Tensor | None) -> torch.Tensor:
    """Zero invalid timesteps in a channel-first ``[B,C,T]`` tensor."""
    if time_mask is None:
        return sequence
    return sequence * time_mask[:, None, :].to(dtype=sequence.dtype)


class CausalConv1d(nn.Conv1d):
    """One-dimensional convolution padded only on the temporal left side."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
    ) -> None:
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        if dilation < 1:
            raise ValueError("dilation must be >= 1")
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            dilation=dilation,
        )
        self.left_padding = dilation * (kernel_size - 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(sequence, (self.left_padding, 0)))


class CausalTemporalBlock(nn.Module):
    """Two-convolution residual TCN block at one dilation level."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.conv1 = CausalConv1d(
            channels, channels, kernel_size, dilation=dilation
        )
        self.conv2 = CausalConv1d(
            channels, channels, kernel_size, dilation=dilation
        )
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _normalise(sequence: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
        return norm(sequence.transpose(1, 2)).transpose(1, 2)

    def forward(
        self, sequence: torch.Tensor, time_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        residual = _masked_channels(sequence, time_mask)
        hidden = self.conv1(residual)
        hidden = self.dropout(self.activation(self._normalise(hidden, self.norm1)))
        hidden = _masked_channels(hidden, time_mask)
        hidden = self.conv2(hidden)
        hidden = self.dropout(self.activation(self._normalise(hidden, self.norm2)))
        hidden = _masked_channels(hidden, time_mask)
        return _masked_channels(self.activation(residual + hidden), time_mask)


class TemporalConvNet(nn.Module):
    """Map independent graph embeddings to causal social states.

    Input and output layouts are ``[batch, social_time, channels]``. Invalid
    padded timesteps are zeroed after every block so that they cannot influence
    later valid states when a mask contains gaps.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        levels: int = 4,
        kernel_size: int = 3,
        dilation_base: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1:
            raise ValueError("input_dim and hidden_dim must be >= 1")
        if levels < 1:
            raise ValueError("levels must be >= 1")
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        if dilation_base < 1:
            raise ValueError("dilation_base must be >= 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.levels = levels
        self.kernel_size = kernel_size
        self.dilations = tuple(dilation_base**level for level in range(levels))
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList(
            CausalTemporalBlock(
                hidden_dim,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
            )
            for dilation in self.dilations
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    @property
    def receptive_field(self) -> int:
        """Maximum number of input patches visible to one output state."""
        return 1 + 2 * (self.kernel_size - 1) * sum(self.dilations)

    @staticmethod
    def _validate_time_mask(
        time_mask: torch.Tensor | None,
        *,
        batch_size: int,
        timesteps: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if time_mask is None:
            return None
        if time_mask.shape != (batch_size, timesteps):
            raise ValueError("time_mask must have shape [B,T]")
        return time_mask.to(device=device, dtype=torch.bool)

    def forward(
        self, graph_embeddings: torch.Tensor, time_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if graph_embeddings.ndim != 3:
            raise ValueError("graph_embeddings must have shape [B,T,D]")
        batch_size, timesteps, channels = graph_embeddings.shape
        if timesteps < 1:
            raise ValueError("graph_embeddings must contain at least one timestep")
        if channels != self.input_dim:
            raise ValueError(
                f"expected graph embedding dimension {self.input_dim}, got {channels}"
            )
        mask = self._validate_time_mask(
            time_mask,
            batch_size=batch_size,
            timesteps=timesteps,
            device=graph_embeddings.device,
        )

        hidden = graph_embeddings
        if mask is not None:
            hidden = hidden * mask.unsqueeze(-1).to(dtype=hidden.dtype)
        hidden = self.input_norm(self.input_projection(hidden))
        hidden = hidden.transpose(1, 2)
        hidden = _masked_channels(hidden, mask)
        for block in self.blocks:
            hidden = block(hidden, mask)
        social_state = self.output_norm(hidden.transpose(1, 2))
        if mask is not None:
            social_state = social_state * mask.unsqueeze(-1).to(
                dtype=social_state.dtype
            )
        return social_state
