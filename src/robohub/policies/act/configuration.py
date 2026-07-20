"""YAML configuration loading and validation for ACT."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config") / "default.yaml"
_TOP_LEVEL_KEYS = {"dataset", "task", "model", "training", "inference"}
_CAMERA_NAMES = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
_DEVICE_PATTERN = re.compile(r"^(?:cpu|mps|cuda(?::[0-9]+)?)$")


def deep_merge(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""
    merged = deepcopy(dict(base))
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"ACT config does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, Mapping):
        raise ValueError(f"ACT config must contain a YAML mapping: {path}")
    return dict(loaded)


def _positive_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _non_negative_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_device(value: Any, name: str) -> None:
    if not isinstance(value, str) or _DEVICE_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{name} must be cpu, mps, cuda, or cuda followed by a device index"
        )


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate the stable ACT configuration contract."""
    keys = set(config)
    if keys != _TOP_LEVEL_KEYS:
        missing = sorted(_TOP_LEVEL_KEYS - keys)
        unknown = sorted(keys - _TOP_LEVEL_KEYS)
        raise ValueError(
            f"ACT config top-level keys are invalid; missing={missing}, "
            f"unknown={unknown}"
        )
    for section_name in _TOP_LEVEL_KEYS:
        if not isinstance(config[section_name], Mapping):
            raise ValueError(f"{section_name} must be a mapping")

    dataset = config["dataset"]
    for key in ("path", "metadata"):
        if not isinstance(dataset.get(key), str) or not dataset[key].strip():
            raise ValueError(f"dataset.{key} must be a non-empty string")

    task = config["task"]
    if task.get("state_dim") != 25 or task.get("action_dim") != 25:
        raise ValueError("task.state_dim and task.action_dim must both be 25")
    if task.get("camera_names") != _CAMERA_NAMES:
        raise ValueError(
            "task.camera_names must be cam_high, cam_left_wrist, "
            "cam_right_wrist in that order"
        )

    model = config["model"]
    _positive_int(model.get("chunk_size"), "model.chunk_size")
    if model.get("loss_function") not in {"l1", "l2", "smooth_l1"}:
        raise ValueError("model.loss_function must be l1, l2, or smooth_l1")

    training = config["training"]
    for key in ("batch_size", "epochs", "downsample", "save_freq"):
        _positive_int(training.get(key), f"training.{key}")
    _non_negative_int(training.get("workers"), "training.workers")
    _non_negative_int(training.get("arm_delay"), "training.arm_delay")
    _non_negative_int(training.get("seed"), "training.seed")
    if not isinstance(training.get("lr"), (int, float)) or training["lr"] <= 0:
        raise ValueError("training.lr must be positive")
    if (
        not isinstance(training.get("ckpt_dir"), str)
        or not training["ckpt_dir"].strip()
    ):
        raise ValueError("training.ckpt_dir must be a non-empty string")
    pretrain = training.get("pretrain")
    if pretrain is not None and (
        not isinstance(pretrain, str) or not pretrain.strip()
    ):
        raise ValueError("training.pretrain must be null or a non-empty string")
    if not isinstance(training.get("wandb"), bool):
        raise ValueError("training.wandb must be a boolean")
    _validate_device(training.get("device"), "training.device")

    inference = config["inference"]
    if not isinstance(inference.get("temporal_agg"), bool):
        raise ValueError("inference.temporal_agg must be a boolean")
    temporal_agg_decay = inference.get("temporal_agg_decay")
    if (
        isinstance(temporal_agg_decay, bool)
        or not isinstance(temporal_agg_decay, (int, float))
        or not math.isfinite(temporal_agg_decay)
        or temporal_agg_decay < 0
    ):
        raise ValueError("inference.temporal_agg_decay must be non-negative")
    _validate_device(inference.get("device"), "inference.device")
    if (
        not isinstance(inference.get("checkpoint"), str)
        or not inference["checkpoint"].strip()
    ):
        raise ValueError("inference.checkpoint must be a non-empty string")
    _positive_int(inference.get("image_width"), "inference.image_width")
    _positive_int(inference.get("image_height"), "inference.image_height")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load defaults, apply an optional YAML override, and validate."""
    defaults = _read_yaml(DEFAULT_CONFIG_PATH)
    config = (
        defaults
        if path is None
        else deep_merge(defaults, _read_yaml(Path(path).expanduser()))
    )
    validate_config(config)
    return config


def save_config(config: Mapping[str, Any], save_dir: str | Path) -> Path:
    """Validate and save configuration as ``config.yaml``."""
    validate_config(config)
    directory = Path(save_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "config.yaml"
    with destination.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(dict(config), stream, sort_keys=False)
    return destination
