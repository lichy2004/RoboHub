# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Positional encodings for the DETR transformer."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn


class PositionEmbeddingSine(nn.Module):
    """Two-dimensional sine and cosine positional encoding."""

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: int = 10000,
        normalize: bool = False,
        scale: float | None = None,
    ) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and not normalize:
            raise ValueError("normalize must be true when scale is provided")
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, tensor: Tensor) -> Tensor:
        """Encode spatial positions for a BCHW feature map."""
        not_mask = torch.ones_like(tensor[:, 0])
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale

        dimensions = torch.arange(
            self.num_pos_feats,
            dtype=torch.float32,
            device=tensor.device,
        )
        dimensions = self.temperature ** (
            2 * torch.div(dimensions, 2, rounding_mode="floor") / self.num_pos_feats
        )
        pos_x = x_embed[:, :, :, None] / dimensions
        pos_y = y_embed[:, :, :, None] / dimensions
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4,
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4,
        ).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class PositionEmbeddingLearned(nn.Module):
    """Learned absolute positional encoding."""

    def __init__(self, num_pos_feats: int = 256) -> None:
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize embedding tables uniformly."""
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor: Tensor) -> Tensor:
        """Encode spatial positions for a BCHW feature map."""
        height, width = tensor.shape[-2:]
        x_embedding = self.col_embed(torch.arange(width, device=tensor.device))
        y_embedding = self.row_embed(torch.arange(height, device=tensor.device))
        position = torch.cat(
            [
                x_embedding.unsqueeze(0).repeat(height, 1, 1),
                y_embedding.unsqueeze(1).repeat(1, width, 1),
            ],
            dim=-1,
        )
        return (
            position.permute(2, 0, 1)
            .unsqueeze(0)
            .repeat(tensor.shape[0], 1, 1, 1)
        )


def build_position_encoding(args: Any) -> nn.Module:
    """Build the requested image position encoder."""
    features = args.hidden_dim // 2
    if args.position_embedding in {"v2", "sine"}:
        return PositionEmbeddingSine(features, normalize=True)
    if args.position_embedding in {"v3", "learned"}:
        return PositionEmbeddingLearned(features)
    raise ValueError(f"unsupported position embedding: {args.position_embedding}")
