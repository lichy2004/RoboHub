"""Training wrapper for the ACT DETR-VAE model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .build import build_act_model_and_optimizer, resolve_model_config


class ACTModel(nn.Module):
    """Apply ACT image normalization, training losses, and serialization."""

    def __init__(
        self,
        args_override: Mapping[str, Any] | None = None,
        config: object | None = None,
        *,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        overrides = dict(args_override or {})
        self.model, self.optimizer = build_act_model_and_optimizer(
            overrides,
            config,
            device=device,
        )
        resolved = resolve_model_config(overrides, config)
        self.kl_weight = float(resolved.kl_weight)
        self.loss_function = str(resolved.loss_function)
        self.chunk_size = resolved.chunk_size
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1),
            persistent=False,
        )

    def forward(
        self,
        qpos: Tensor,
        image: Tensor,
        actions: Tensor | None = None,
        is_pad: Tensor | None = None,
    ) -> Tensor | dict[str, Tensor]:
        """Run inference or return ACT reconstruction and KL losses."""
        image = (image - self.image_mean) / self.image_std
        if actions is None:
            action_hat, _, _ = self.model(qpos, image, None)
            return action_hat
        if is_pad is None:
            raise ValueError("is_pad is required when actions are provided")

        actions = actions[:, : self.chunk_size]
        is_pad = is_pad[:, : self.chunk_size]
        action_hat, _, (mu, logvar) = self.model(
            qpos,
            image,
            None,
            actions,
            is_pad,
        )
        if self.loss_function == "l1":
            element_loss = F.l1_loss(actions, action_hat, reduction="none")
        elif self.loss_function == "l2":
            element_loss = F.mse_loss(actions, action_hat, reduction="none")
        else:
            element_loss = F.smooth_l1_loss(actions, action_hat, reduction="none")

        reconstruction = (element_loss * ~is_pad.unsqueeze(-1)).mean()
        losses = {"l1": reconstruction}
        if self.kl_weight:
            if mu is None or logvar is None:
                raise RuntimeError("latent statistics are missing during training")
            total_kld, _, _ = kl_divergence(mu, logvar)
            losses["kl"] = total_kld[0]
            losses["loss"] = reconstruction + total_kld[0] * self.kl_weight
        else:
            losses["loss"] = reconstruction
        return losses

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Return the optimizer owned by this training wrapper."""
        return self.optimizer

    def serialize(self) -> dict[str, Tensor]:
        """Return model state suitable for a checkpoint."""
        return self.state_dict()

    def deserialize(self, model_dict: Mapping[str, Tensor]) -> Any:
        """Load a serialized model state."""
        return self.load_state_dict(model_dict)


def kl_divergence(
    mu: Tensor,
    logvar: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Calculate total, dimension-wise, and mean KL divergence."""
    if mu.size(0) == 0:
        raise ValueError("KL divergence requires a non-empty batch")
    if mu.ndim == 4:
        mu = mu.reshape(mu.size(0), mu.size(1))
    if logvar.ndim == 4:
        logvar = logvar.reshape(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, keepdim=True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, keepdim=True)
    return total_kld, dimension_wise_kld, mean_kld
