# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""DETR-VAE model for Action Chunking with Transformers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .backbone import Joiner, build_backbone
from .transformer import (
    Transformer,
    TransformerEncoder,
    TransformerEncoderLayer,
    build_transformer,
)


def reparametrize(mu: Tensor, logvar: Tensor) -> Tensor:
    """Sample a latent value using the reparameterization trick."""
    std = logvar.div(2).exp()
    return mu + std * torch.randn_like(std)


def get_sinusoid_encoding_table(n_position: int, d_hid: int) -> Tensor:
    """Create the fixed sinusoidal encoding used by the CVAE encoder."""

    def position_angles(position: int) -> list[float]:
        return [
            position / np.power(10000, 2 * (hidden_index // 2) / d_hid)
            for hidden_index in range(d_hid)
        ]

    table = np.array([position_angles(index) for index in range(n_position)])
    table[:, 0::2] = np.sin(table[:, 0::2])
    table[:, 1::2] = np.cos(table[:, 1::2])
    return torch.tensor(table, dtype=torch.float32).unsqueeze(0)


class DETRVAE(nn.Module):
    """Predict action chunks with a conditional VAE and DETR transformer."""

    def __init__(
        self,
        backbones: list[Joiner] | None,
        transformer: Transformer,
        encoder: TransformerEncoder,
        state_dim: int,
        num_queries: int,
        camera_names: list[str],
        action_dim: int | None = None,
    ) -> None:
        super().__init__()
        action_dim = state_dim if action_dim is None else action_dim
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(
                backbones[0].num_channels,
                hidden_dim,
                kernel_size=1,
            )
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = nn.Embedding(2, hidden_dim)
            self.backbones = None

        self.latent_dim = 32
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim)
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)
        self.register_buffer(
            "pos_table",
            get_sinusoid_encoding_table(2 + num_queries, hidden_dim),
        )

        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)

    def forward(
        self,
        qpos: Tensor,
        image: Tensor,
        env_state: Tensor | None,
        actions: Tensor | None = None,
        is_pad: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, tuple[Tensor | None, Tensor | None]]:
        """Run the CVAE encoder when training and DETR decoder in all modes."""
        is_training = actions is not None
        batch_size = qpos.shape[0]

        if is_training:
            if is_pad is None:
                raise ValueError("is_pad is required when actions are provided")
            action_embed = self.encoder_action_proj(actions)
            qpos_embed = self.encoder_joint_proj(qpos).unsqueeze(1)
            cls_embed = self.cls_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
            encoder_input = torch.cat(
                [cls_embed, qpos_embed, action_embed],
                dim=1,
            ).permute(1, 0, 2)
            prefix_is_pad = torch.zeros(
                (batch_size, 2),
                dtype=torch.bool,
                device=qpos.device,
            )
            encoder_is_pad = torch.cat([prefix_is_pad, is_pad], dim=1)
            pos_embed = self.pos_table.detach().permute(1, 0, 2)
            encoder_output = self.encoder(
                encoder_input,
                pos=pos_embed,
                src_key_padding_mask=encoder_is_pad,
            )[0]
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, : self.latent_dim]
            logvar = latent_info[:, self.latent_dim :]
            latent_input = self.latent_out_proj(reparametrize(mu, logvar))
        else:
            mu = logvar = None
            latent_sample = torch.zeros(
                (batch_size, self.latent_dim),
                dtype=qpos.dtype,
                device=qpos.device,
            )
            latent_input = self.latent_out_proj(latent_sample)

        if self.backbones is not None:
            camera_features = []
            camera_positions = []
            for camera_id, _ in enumerate(self.camera_names):
                features, positions = self.backbones[0](image[:, camera_id])
                camera_features.append(self.input_proj(features[0]))
                camera_positions.append(positions[0])
            proprio_input = self.input_proj_robot_state(qpos)
            src = torch.cat(camera_features, dim=3)
            pos = torch.cat(camera_positions, dim=3)
            hidden_states = self.transformer(
                src,
                None,
                self.query_embed.weight,
                pos,
                latent_input,
                proprio_input,
                self.additional_pos_embed.weight,
            )[0]
        else:
            if env_state is None:
                raise ValueError("env_state is required when no backbone is configured")
            qpos_input = self.input_proj_robot_state(qpos)
            environment_input = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos_input, environment_input], dim=1)
            hidden_states = self.transformer(
                transformer_input,
                None,
                self.query_embed.weight,
                self.pos.weight,
            )[0]

        action_hat = self.action_head(hidden_states)
        is_pad_hat = self.is_pad_head(hidden_states)
        return action_hat, is_pad_hat, (mu, logvar)


def build_encoder(args: Any) -> TransformerEncoder:
    """Build the CVAE sequence encoder."""
    encoder_layer = TransformerEncoderLayer(
        args.hidden_dim,
        args.nheads,
        args.dim_feedforward,
        args.dropout,
        "relu",
        args.pre_norm,
    )
    encoder_norm = nn.LayerNorm(args.hidden_dim) if args.pre_norm else None
    return TransformerEncoder(encoder_layer, args.enc_layers, encoder_norm)


def build(args: Any) -> DETRVAE:
    """Build DETR-VAE from a resolved ACT configuration."""
    state_dim = int(args.state_dim)
    action_dim = int(getattr(args, "action_dim", state_dim))
    backbone = build_backbone(args)
    model = DETRVAE(
        [backbone],
        build_transformer(args),
        build_encoder(args),
        state_dim=state_dim,
        num_queries=int(args.chunk_size),
        camera_names=list(args.camera_names),
        action_dim=action_dim,
    )
    return model
