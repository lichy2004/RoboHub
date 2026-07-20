# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""ResNet backbone modules used by DETR-VAE."""

from __future__ import annotations

from typing import Any

import torch
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(nn.Module):
    """Batch normalization with fixed statistics and affine parameters."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.register_buffer("weight", torch.ones(channels))
        self.register_buffer("bias", torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        state_dict.pop(prefix + "num_batches_tracked", None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        running_variance = self.running_var.reshape(1, -1, 1, 1)
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        scale = weight * (running_variance + 1e-5).rsqrt()
        return inputs * scale + (bias - running_mean * scale)


class BackboneBase(nn.Module):
    """Expose selected intermediate layers from an image backbone."""

    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        num_channels: int,
        return_interm_layers: bool,
    ) -> None:
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or not any(
                layer in name for layer in ("layer2", "layer3", "layer4")
            ):
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {
                "layer1": "0",
                "layer2": "1",
                "layer3": "2",
                "layer4": "3",
            }
        else:
            return_layers = {"layer4": "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor: Tensor) -> dict[str, Tensor]:
        """Extract the configured ResNet feature maps."""
        return self.body(tensor)


class Backbone(BackboneBase):
    """ResNet backbone with frozen batch normalization."""

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        return_interm_layers: bool,
        dilation: bool,
        pretrained: bool,
    ) -> None:
        constructor = getattr(torchvision.models, name)
        backbone = constructor(
            replace_stride_with_dilation=[False, False, dilation],
            weights="DEFAULT" if pretrained else None,
            norm_layer=FrozenBatchNorm2d,
        )
        num_channels = 512 if name in {"resnet18", "resnet34"} else 2048
        super().__init__(
            backbone,
            train_backbone,
            num_channels,
            return_interm_layers,
        )


class Joiner(nn.Sequential):
    """Combine an image backbone with positional encoding."""

    num_channels: int

    def __init__(self, backbone: Backbone, position_embedding: nn.Module) -> None:
        super().__init__(backbone, position_embedding)

    def forward(self, tensor: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        """Return image features and matching positional encodings."""
        feature_maps = self[0](tensor)
        features = []
        positions = []
        for feature_map in feature_maps.values():
            features.append(feature_map)
            positions.append(self[1](feature_map).to(feature_map.dtype))
        return features, positions


def build_backbone(args: Any) -> Joiner:
    """Build the configured ResNet and position encoder."""
    backbone = Backbone(
        args.backbone,
        args.lr_backbone > 0,
        args.masks,
        args.dilation,
        bool(getattr(args, "pretrained_backbone", True)),
    )
    model = Joiner(backbone, build_position_encoding(args))
    model.num_channels = backbone.num_channels
    return model
