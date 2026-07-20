# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""DETR transformer with explicit positional encoding inputs."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Transformer(nn.Module):
    """Encoder-decoder transformer used by ACT."""

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False,
        return_intermediate_dec: bool = False,
    ) -> None:
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer,
            num_encoder_layers,
            encoder_norm,
        )
        decoder_layer = TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
        )
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            nn.LayerNorm(d_model),
            return_intermediate=return_intermediate_dec,
        )
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(
        self,
        src: Tensor,
        mask: Tensor | None,
        query_embed: Tensor,
        pos_embed: Tensor,
        latent_input: Tensor | None = None,
        proprio_input: Tensor | None = None,
        additional_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """Decode action queries from image or state tokens."""
        if src.ndim == 4:
            batch_size = src.shape[0]
            src = src.flatten(2).permute(2, 0, 1)
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
            pos_embed = pos_embed.repeat(1, batch_size, 1)
            query_embed = query_embed.unsqueeze(1).repeat(1, batch_size, 1)
            if (
                latent_input is None
                or proprio_input is None
                or additional_pos_embed is None
            ):
                raise ValueError(
                    "image input requires latent and proprioception tokens"
                )
            extra_positions = additional_pos_embed.unsqueeze(1).repeat(
                1,
                batch_size,
                1,
            )
            pos_embed = torch.cat([extra_positions, pos_embed], dim=0)
            src = torch.cat(
                [torch.stack([latent_input, proprio_input], dim=0), src],
                dim=0,
            )
        else:
            if src.ndim != 3:
                raise ValueError("transformer input must have three or four dimensions")
            batch_size = src.shape[0]
            src = src.permute(1, 0, 2)
            pos_embed = pos_embed.unsqueeze(1).repeat(1, batch_size, 1)
            query_embed = query_embed.unsqueeze(1).repeat(1, batch_size, 1)

        target = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hidden_states = self.decoder(
            target,
            memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed,
        )
        return hidden_states.transpose(1, 2)


class TransformerEncoder(nn.Module):
    """Stack of DETR transformer encoder layers."""

    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        num_layers: int,
        norm: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
    ) -> Tensor:
        """Encode a sequence, adding positions inside attention."""
        output = src
        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )
        return self.norm(output) if self.norm is not None else output


class TransformerDecoder(nn.Module):
    """Stack of DETR transformer decoder layers."""

    def __init__(
        self,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int,
        norm: nn.Module | None = None,
        return_intermediate: bool = False,
    ) -> None:
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        target: Tensor,
        memory: Tensor,
        target_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        target_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
        query_pos: Tensor | None = None,
    ) -> Tensor:
        """Decode queries and optionally return every layer output."""
        output = target
        intermediate = []
        for layer in self.layers:
            output = layer(
                output,
                memory,
                target_mask=target_mask,
                memory_mask=memory_mask,
                target_key_padding_mask=target_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate and self.norm is not None:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate[-1] = output
        if self.return_intermediate:
            return torch.stack(intermediate)
        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    """A transformer encoder layer supporting pre- or post-normalization."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Tensor | None) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
    ) -> Tensor:
        query = key = self.with_pos_embed(src, pos)
        attended = self.self_attn(
            query,
            key,
            value=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]
        src = self.norm1(src + self.dropout1(attended))
        projected = self.linear2(self.dropout(self.activation(self.linear1(src))))
        return self.norm2(src + self.dropout2(projected))

    def forward_pre(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
    ) -> Tensor:
        normalized = self.norm1(src)
        query = key = self.with_pos_embed(normalized, pos)
        attended = self.self_attn(
            query,
            key,
            value=normalized,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]
        src = src + self.dropout1(attended)
        normalized = self.norm2(src)
        projected = self.linear2(
            self.dropout(self.activation(self.linear1(normalized)))
        )
        return src + self.dropout2(projected)

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
    ) -> Tensor:
        """Encode one sequence layer."""
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    """A transformer decoder layer supporting pre- or post-normalization."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Tensor | None) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        target: Tensor,
        memory: Tensor,
        target_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        target_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
        query_pos: Tensor | None = None,
    ) -> Tensor:
        query = key = self.with_pos_embed(target, query_pos)
        attended = self.self_attn(
            query,
            key,
            value=target,
            attn_mask=target_mask,
            key_padding_mask=target_key_padding_mask,
        )[0]
        target = self.norm1(target + self.dropout1(attended))
        attended = self.multihead_attn(
            query=self.with_pos_embed(target, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        target = self.norm2(target + self.dropout2(attended))
        projected = self.linear2(
            self.dropout(self.activation(self.linear1(target)))
        )
        return self.norm3(target + self.dropout3(projected))

    def forward_pre(
        self,
        target: Tensor,
        memory: Tensor,
        target_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        target_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
        query_pos: Tensor | None = None,
    ) -> Tensor:
        normalized = self.norm1(target)
        query = key = self.with_pos_embed(normalized, query_pos)
        attended = self.self_attn(
            query,
            key,
            value=normalized,
            attn_mask=target_mask,
            key_padding_mask=target_key_padding_mask,
        )[0]
        target = target + self.dropout1(attended)
        normalized = self.norm2(target)
        attended = self.multihead_attn(
            query=self.with_pos_embed(normalized, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        target = target + self.dropout2(attended)
        normalized = self.norm3(target)
        projected = self.linear2(
            self.dropout(self.activation(self.linear1(normalized)))
        )
        return target + self.dropout3(projected)

    def forward(
        self,
        target: Tensor,
        memory: Tensor,
        target_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        target_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
        query_pos: Tensor | None = None,
    ) -> Tensor:
        """Decode one query layer."""
        method = self.forward_pre if self.normalize_before else self.forward_post
        return method(
            target,
            memory,
            target_mask,
            memory_mask,
            target_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


def _get_clones(module: nn.Module, count: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(count)])


def build_transformer(args: Any) -> Transformer:
    """Build the ACT DETR transformer."""
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation must be relu, gelu, or glu, not {activation}")
