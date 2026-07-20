# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Model and optimizer factories for ACT."""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .detr_vae import build as build_detr_vae

_DEFAULTS: dict[str, Any] = {
    "lr": 4e-5,
    "lr_backbone": 1e-5,
    "weight_decay": 1e-4,
    "backbone": "resnet18",
    "dilation": False,
    "position_embedding": "sine",
    "camera_names": ["cam_high"],
    "enc_layers": 4,
    "dec_layers": 7,
    "dim_feedforward": 3200,
    "hidden_dim": 512,
    "dropout": 0.1,
    "nheads": 8,
    "pre_norm": False,
    "masks": False,
    "chunk_size": 30,
    "state_dim": 25,
    "action_dim": 25,
    "kl_weight": 10.0,
    "loss_function": "l1",
}


def _config_values(config: object | None) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "__dict__"):
        return vars(config).copy()
    raise TypeError("config must be a mapping or an object with attributes")


def resolve_model_config(
    args_override: Mapping[str, Any] | None = None,
    config: object | None = None,
) -> Namespace:
    """Resolve model settings with overrides taking precedence over config."""
    config_values = _config_values(config)
    override_values = dict(args_override or {})
    values = _DEFAULTS.copy()
    values.update(config_values)
    values.update(override_values)

    values["chunk_size"] = int(values["chunk_size"])
    values["state_dim"] = int(values.get("state_dim", 25))
    if "action_dim" not in config_values and "action_dim" not in override_values:
        values["action_dim"] = values["state_dim"]
    values["action_dim"] = int(values["action_dim"])
    if values["chunk_size"] <= 0:
        raise ValueError("chunk_size must be positive")
    if values["state_dim"] <= 0 or values["action_dim"] <= 0:
        raise ValueError("state_dim and action_dim must be positive")
    return Namespace(**values)


def build_act_model(
    args_override: Mapping[str, Any] | None = None,
    config: object | None = None,
    *,
    device: torch.device | str,
) -> nn.Module:
    """Build DETR-VAE and move it to the explicitly selected device."""
    args = resolve_model_config(args_override, config)
    return build_detr_vae(args).to(torch.device(device))


def build_act_model_and_optimizer(
    args_override: Mapping[str, Any] | None = None,
    config: object | None = None,
    *,
    device: torch.device | str,
) -> tuple[nn.Module, torch.optim.Optimizer]:
    """Build DETR-VAE and its AdamW optimizer."""
    args = resolve_model_config(args_override, config)
    model = build_detr_vae(args).to(torch.device(device))
    param_dicts = [
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if "backbone" not in name and parameter.requires_grad
            ]
        },
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if "backbone" in name and parameter.requires_grad
            ],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(
        param_dicts,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    return model, optimizer


# Compatibility aliases for the original ACT naming.
build_ACT_model = build_act_model
build_ACT_model_and_optimizer = build_act_model_and_optimizer
