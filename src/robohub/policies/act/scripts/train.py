"""Train ACT from a YAML configuration and converted dataset."""

from __future__ import annotations

import argparse
import pickle
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

from robohub.policies.act.configuration import (
    load_config,
    save_config,
    validate_config,
)
from robohub.policies.act.data.dataset import (
    compute_dict_mean,
    detach_dict,
    load_data,
    set_seed,
)
from robohub.policies.act.model import ACTModel


def _load_metadata(dataset_dir: Path, metadata_name: str) -> dict[str, Any]:
    metadata_path = Path(metadata_name).expanduser()
    if not metadata_path.is_absolute():
        metadata_path = dataset_dir / metadata_path
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Dataset metadata does not exist: {metadata_path}")
    with metadata_path.open(encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Dataset metadata must be a mapping: {metadata_path}")
    return dict(metadata)


def _metadata_int(metadata: Mapping[str, Any], key: str) -> int:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"dataset.yaml {key} must be a positive integer")
    return value


def _validate_metadata(
    metadata: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[int, list[str]]:
    task = config["task"]
    num_episodes = _metadata_int(metadata, "num_episodes")
    state_dim = _metadata_int(metadata, "state_dim")
    action_dim = _metadata_int(metadata, "action_dim")
    camera_names = metadata.get("camera_names")
    if not isinstance(camera_names, list) or not all(
        isinstance(name, str) for name in camera_names
    ):
        raise ValueError("dataset.yaml camera_names must be a list of strings")

    expected = {
        "state_dim": task["state_dim"],
        "action_dim": task["action_dim"],
        "camera_names": task["camera_names"],
    }
    actual = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "camera_names": camera_names,
    }
    if actual != expected:
        raise ValueError(
            "Dataset metadata does not match ACT task configuration: "
            f"expected {expected}, got {actual}"
        )
    return num_episodes, camera_names


def _resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {name!r} was requested, but CUDA is unavailable"
        )
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but MPS is unavailable")
    return device


def _model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    model_config = dict(config["model"])
    task = config["task"]
    training = config["training"]
    model_config.update(
        {
            "state_dim": task["state_dim"],
            "action_dim": task["action_dim"],
            "camera_names": task["camera_names"],
            "lr": training["lr"],
        }
    )
    return model_config


def _load_pretrain(
    model: ACTModel,
    checkpoint_path: str | None,
    device: torch.device,
) -> None:
    if checkpoint_path is None:
        return
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Pretrain checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(
            f"Pretrain checkpoint must contain an ACTModel state_dict: {path}"
        )
    try:
        model.deserialize(checkpoint)
    except RuntimeError as error:
        raise RuntimeError(
            f"Pretrain checkpoint is incompatible with this ACT model: {path}"
        ) from error
    print(f"Loaded pretrain checkpoint: {path}")


def _forward_batch(
    batch: Sequence[Tensor],
    model: ACTModel,
    device: torch.device,
) -> dict[str, Tensor]:
    image, qpos, actions, is_pad = (
        value.to(device, non_blocking=device.type == "cuda") for value in batch
    )
    output = model(qpos, image, actions, is_pad)
    if not isinstance(output, dict):
        raise RuntimeError("ACTModel did not return training losses")
    return output


def _float_summary(summary: Mapping[str, Tensor]) -> dict[str, float]:
    return {key: float(value.item()) for key, value in summary.items()}


def _save_loss_curves(
    train_history: Sequence[Mapping[str, float]],
    val_history: Sequence[Mapping[str, float]],
    checkpoint_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    for key in train_history[0]:
        figure, axes = plt.subplots()
        axes.plot(
            range(1, len(train_history) + 1),
            [summary[key] for summary in train_history],
            label="train",
        )
        axes.plot(
            range(1, len(val_history) + 1),
            [summary[key] for summary in val_history],
            label="validation",
        )
        axes.set_xlabel("epoch")
        axes.set_ylabel(key)
        axes.legend()
        figure.tight_layout()
        figure.savefig(checkpoint_dir / f"train_val_{key}.png")
        plt.close(figure)


def train(
    config: Mapping[str, Any],
    train_loader: Any,
    val_loader: Any,
    checkpoint_dir: Path,
    wandb_run: Any = None,
) -> tuple[int, float]:
    """Train, validate, and persist best, last, and periodic checkpoints."""
    training = config["training"]
    device = _resolve_device(training["device"])
    set_seed(training["seed"])
    model = ACTModel(_model_config(config), device=device).to(device)
    _load_pretrain(model, training["pretrain"], device)
    optimizer = model.configure_optimizers()

    best_epoch = -1
    best_val_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    train_history: list[dict[str, float]] = []
    val_history: list[dict[str, float]] = []

    for epoch in range(training["epochs"]):
        model.train()
        train_batches: list[dict[str, Tensor]] = []
        for batch in train_loader:
            optimizer.zero_grad()
            losses = _forward_batch(batch, model, device)
            losses["loss"].backward()
            optimizer.step()
            train_batches.append(detach_dict(losses))
        train_summary = compute_dict_mean(train_batches)
        train_values = _float_summary(train_summary)
        train_history.append(train_values)

        model.eval()
        val_batches: list[dict[str, Tensor]] = []
        with torch.inference_mode():
            for batch in val_loader:
                val_batches.append(detach_dict(_forward_batch(batch, model, device)))
        val_summary = compute_dict_mean(val_batches)
        val_values = _float_summary(val_summary)
        val_history.append(val_values)
        val_loss = val_values["loss"]

        if val_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            torch.save(best_state, checkpoint_dir / "policy_best.ckpt")

        if (epoch + 1) % training["save_freq"] == 0:
            torch.save(
                model.state_dict(),
                checkpoint_dir / f"policy_epoch_{epoch + 1}.ckpt",
            )
            _save_loss_curves(train_history, val_history, checkpoint_dir)

        metrics = {
            **{f"train/{key}": value for key, value in train_values.items()},
            **{f"val/{key}": value for key, value in val_values.items()},
        }
        if wandb_run is not None:
            wandb_run.log(metrics, step=epoch)
        print(
            f"Epoch {epoch + 1}/{training['epochs']} "
            f"train_loss={train_values['loss']:.6f} "
            f"val_loss={val_loss:.6f}"
        )

    torch.save(model.state_dict(), checkpoint_dir / "policy_last.ckpt")
    torch.save(best_state, checkpoint_dir / "policy_best.ckpt")
    _save_loss_curves(train_history, val_history, checkpoint_dir)
    if wandb_run is not None:
        wandb_run.summary["best_epoch"] = best_epoch + 1
        wandb_run.summary["best_val_loss"] = best_val_loss
    return best_epoch + 1, best_val_loss


def _apply_cli_overrides(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    if args.dataset_dir is not None:
        config["dataset"]["path"] = str(args.dataset_dir)
    if args.ckpt_dir is not None:
        config["training"]["ckpt_dir"] = str(args.ckpt_dir)
    if args.device is not None:
        config["training"]["device"] = args.device
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.wandb is not None:
        config["training"]["wandb"] = args.wandb
    validate_config(config)


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an ACT policy")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--ckpt-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable Weights & Biases logging",
    )
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> None:
    parsed = parse_args(args)
    config = load_config(parsed.config)
    _apply_cli_overrides(config, parsed)

    dataset_dir = Path(config["dataset"]["path"]).expanduser()
    metadata = _load_metadata(dataset_dir, config["dataset"]["metadata"])
    num_episodes, camera_names = _validate_metadata(metadata, config)
    training = config["training"]
    set_seed(training["seed"])
    train_loader, val_loader, stats, _ = load_data(
        dataset_dir,
        num_episodes,
        camera_names,
        training["batch_size"],
        training["batch_size"],
        arm_delay_time=training["arm_delay"],
        chunk_size=config["model"]["chunk_size"],
        down_sample=training["downsample"],
        num_workers=training["workers"],
    )

    checkpoint_dir = Path(training["ckpt_dir"]).expanduser()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (checkpoint_dir / "dataset_stats.pkl").open("wb") as stream:
        pickle.dump(stats, stream)
    save_config(config, checkpoint_dir)

    wandb_run = None
    if training["wandb"]:
        import wandb

        wandb_run = wandb.init(project="robohub-act", config=config)
    try:
        best_epoch, best_val_loss = train(
            config,
            train_loader,
            val_loader,
            checkpoint_dir,
            wandb_run,
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    print(
        f"Training finished: best validation loss {best_val_loss:.6f} "
        f"at epoch {best_epoch}"
    )


if __name__ == "__main__":
    main()
