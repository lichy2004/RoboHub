"""Runtime inference policy for ACT checkpoints."""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from robohub.policies import Policy
from robohub.policies.act.configuration import (
    deep_merge,
    load_config,
    validate_config,
)
from robohub.policies.act.model import ACTModel
from robohub.schemas import Action, Observation

_CAMERA_KEYS = {
    "cam_high": "head",
    "cam_left_wrist": "wrist_left",
    "cam_right_wrist": "wrist_right",
}
_STAT_NAMES = ("qpos_mean", "qpos_std", "action_mean", "action_std")
_VECTOR_SIZE = 25


class ACTPolicy(Policy):
    """Run an ACT model checkpoint against RoboHub observations."""

    def __init__(
        self,
        checkpoint: str | Path,
        config: str | Path | Mapping[str, Any] | None = None,
    ) -> None:
        checkpoint_input = Path(checkpoint).expanduser()
        self.config = self._load_runtime_config(checkpoint_input, config)
        inference = self.config["inference"]
        self.device = self._resolve_device(inference["device"])

        checkpoint_path, artifact_dir = self._resolve_checkpoint(
            checkpoint_input,
            inference["checkpoint"],
        )
        self._stats = self._load_stats(artifact_dir / "dataset_stats.pkl")

        task = self.config["task"]
        if task["state_dim"] != _VECTOR_SIZE or task["action_dim"] != _VECTOR_SIZE:
            raise ValueError("ACT runtime requires state_dim=action_dim=25")

        model_config = dict(self.config["model"])
        model_config.update(
            {
                "state_dim": task["state_dim"],
                "action_dim": task["action_dim"],
                "camera_names": task["camera_names"],
                "lr": self.config["training"]["lr"],
            }
        )
        self.model: ACTModel | None = ACTModel(
            model_config,
            device=self.device,
        ).to(self.device)
        self.optimizer: torch.optim.Optimizer | None = (
            self.model.configure_optimizers()
        )
        self._load_checkpoint(checkpoint_path)
        self.model.eval()

        self._chunk_size = int(self.config["model"]["chunk_size"])
        self._temporal_agg = bool(inference["temporal_agg"])
        decay = inference.get("temporal_agg_decay", 0.01)
        if (
            isinstance(decay, bool)
            or not isinstance(decay, (int, float))
            or not np.isfinite(decay)
            or decay < 0
        ):
            raise ValueError("inference.temporal_agg_decay must be finite and >= 0")
        self._temporal_agg_decay = float(decay)
        self._image_size = (
            int(inference["image_width"]),
            int(inference["image_height"]),
        )
        self._camera_names = tuple(task["camera_names"])
        self._timestep = 0
        self._cached_chunk: NDArray[np.float32] | None = None
        self._cache_index = 0
        self._history: list[tuple[int, NDArray[np.float32]]] = []
        self._closed = False

    @staticmethod
    def _load_runtime_config(
        checkpoint: Path,
        config: str | Path | Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if isinstance(config, Mapping):
            merged = deep_merge(load_config(), config)
            validate_config(merged)
            return merged
        if config is not None:
            return load_config(config)

        config_dir = checkpoint if checkpoint.is_dir() else checkpoint.parent
        adjacent_config = config_dir / "config.yaml"
        return load_config(adjacent_config if adjacent_config.is_file() else None)

    @staticmethod
    def _resolve_checkpoint(
        checkpoint: Path,
        configured_checkpoint: str,
    ) -> tuple[Path, Path]:
        if checkpoint.is_dir():
            configured_path = Path(configured_checkpoint).expanduser()
            checkpoint_path = (
                configured_path
                if configured_path.is_absolute()
                else checkpoint / configured_path
            )
            artifact_dir = checkpoint
        else:
            checkpoint_path = checkpoint
            artifact_dir = checkpoint.parent
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"ACT checkpoint does not exist: {checkpoint_path}"
            )
        return checkpoint_path, artifact_dir

    @staticmethod
    def _resolve_device(name: str) -> torch.device:
        try:
            device = torch.device(name)
        except (RuntimeError, ValueError) as error:
            raise ValueError(f"Invalid ACT inference device: {name!r}") from error
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"CUDA device {name!r} was requested, but CUDA is unavailable"
                )
            if device.index is not None and device.index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"CUDA device index {device.index} is unavailable; "
                    f"found {torch.cuda.device_count()} device(s)"
                )
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but MPS is unavailable")
        return device

    @staticmethod
    def _load_stats(path: Path) -> dict[str, NDArray[np.float32]]:
        if not path.is_file():
            raise FileNotFoundError(f"ACT dataset statistics do not exist: {path}")
        with path.open("rb") as stream:
            loaded = pickle.load(stream)
        if not isinstance(loaded, Mapping):
            raise ValueError(f"ACT dataset statistics must be a mapping: {path}")

        stats: dict[str, NDArray[np.float32]] = {}
        for name in _STAT_NAMES:
            if name not in loaded:
                raise ValueError(f"ACT dataset statistics are missing {name!r}")
            value = np.asarray(loaded[name])
            if value.shape != (_VECTOR_SIZE,):
                raise ValueError(
                    f"ACT statistic {name!r} must have shape (25,), got {value.shape}"
                )
            if not np.issubdtype(value.dtype, np.number):
                raise ValueError(f"ACT statistic {name!r} must be numeric")
            if not np.isfinite(value).all():
                raise ValueError(f"ACT statistic {name!r} contains non-finite values")
            if name.endswith("_std") and np.any(value <= 0):
                raise ValueError(f"ACT statistic {name!r} must be strictly positive")
            stats[name] = value.astype(np.float32, copy=True)
        return stats

    def _load_checkpoint(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("ACT policy is closed")
        checkpoint = torch.load(path, map_location=self.device)
        if not isinstance(checkpoint, Mapping):
            raise ValueError(
                f"ACT checkpoint must contain an ACTModel state_dict: {path}"
            )
        try:
            self.model.load_state_dict(checkpoint, strict=True)
        except RuntimeError as error:
            raise RuntimeError(
                f"ACT checkpoint is incompatible with this ACTModel: {path}"
            ) from error

    def _prepare_inputs(
        self,
        observation: Observation,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qpos = np.asarray(observation.joints_position)
        if qpos.shape != (_VECTOR_SIZE,):
            raise ValueError(
                f"observation.joints_position must have shape (25,), got {qpos.shape}"
            )
        if not np.issubdtype(qpos.dtype, np.number):
            raise ValueError("observation.joints_position must be numeric")
        if not np.isfinite(qpos).all():
            raise ValueError("observation.joints_position contains non-finite values")

        images: list[NDArray[np.float32]] = []
        for camera_name in self._camera_names:
            camera_key = _CAMERA_KEYS.get(camera_name)
            if camera_key is None:
                raise ValueError(f"Unsupported ACT camera name: {camera_name!r}")
            if camera_key not in observation.rgb:
                raise ValueError(
                    f"observation.rgb is missing required camera {camera_key!r}"
                )
            image = np.asarray(observation.rgb[camera_key])
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(
                    f"camera {camera_key!r} must be an HWC3 image, got {image.shape}"
                )
            if not np.issubdtype(image.dtype, np.number):
                raise ValueError(f"camera {camera_key!r} image must be numeric")
            if not np.isfinite(image).all():
                raise ValueError(
                    f"camera {camera_key!r} image contains non-finite values"
                )
            # RoboHub images are already RGB; resizing must not swap channels.
            resized = cv2.resize(
                image,
                self._image_size,
                interpolation=cv2.INTER_LINEAR,
            )
            chw = np.ascontiguousarray(resized.transpose(2, 0, 1))
            images.append(chw.astype(np.float32) / 255.0)

        normalized_qpos = (
            qpos.astype(np.float32) - self._stats["qpos_mean"]
        ) / self._stats["qpos_std"]
        qpos_tensor = torch.from_numpy(normalized_qpos).unsqueeze(0).to(self.device)
        image_tensor = torch.from_numpy(np.stack(images)).unsqueeze(0).to(self.device)
        return qpos_tensor, image_tensor

    def _predict_chunk(
        self,
        qpos: torch.Tensor,
        images: torch.Tensor,
    ) -> NDArray[np.float32]:
        """Run one model query and return denormalized actions."""
        if self.model is None or self._closed:
            raise RuntimeError("ACT policy is closed")
        with torch.inference_mode():
            output = self.model(qpos, images)
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("ACTModel did not return an action tensor")
        expected_shape = (1, self._chunk_size, _VECTOR_SIZE)
        if tuple(output.shape) != expected_shape:
            raise RuntimeError(
                f"ACTModel output must have shape {expected_shape}, "
                f"got {tuple(output.shape)}"
            )
        chunk = output[0].detach().to(device="cpu", dtype=torch.float32).numpy()
        if not np.isfinite(chunk).all():
            raise RuntimeError("ACTModel output contains non-finite values")
        chunk = (
            chunk * self._stats["action_std"] + self._stats["action_mean"]
        )
        return chunk.astype(np.float32, copy=False)

    def infer(self, observation: Observation) -> Action:
        """Infer the action for the current observation."""
        if self._closed:
            raise RuntimeError("ACT policy is closed")
        qpos, images = self._prepare_inputs(observation)

        if self._temporal_agg:
            chunk = self._predict_chunk(qpos, images)
            self._history.append((self._timestep, chunk))
            self._history = [
                item
                for item in self._history
                if item[0] <= self._timestep < item[0] + len(item[1])
            ]
            action = self._aggregate_current_action()
        else:
            if (
                self._cached_chunk is None
                or self._cache_index >= len(self._cached_chunk)
            ):
                self._cached_chunk = self._predict_chunk(qpos, images)
                self._cache_index = 0
            action = self._cached_chunk[self._cache_index]
            self._cache_index += 1

        self._timestep += 1
        return self._to_action(action)

    def _aggregate_current_action(self) -> NDArray[np.float32]:
        starts = np.asarray([start for start, _ in self._history], dtype=np.int64)
        valid_mask = starts <= self._timestep
        valid_mask &= self._timestep < np.asarray(
            [start + len(chunk) for start, chunk in self._history],
            dtype=np.int64,
        )
        valid_indices = np.flatnonzero(valid_mask)
        if valid_indices.size == 0:
            raise RuntimeError("No ACT prediction covers the current timestep")

        actions = np.stack(
            [
                self._history[index][1][
                    self._timestep - self._history[index][0]
                ]
                for index in valid_indices
            ]
        )
        ages = self._timestep - starts[valid_indices]
        weights = np.exp(-self._temporal_agg_decay * ages.astype(np.float64))
        weights /= weights.sum()
        return np.sum(actions * weights[:, None], axis=0, dtype=np.float64).astype(
            np.float32
        )

    @staticmethod
    def _to_action(values: NDArray[np.float32]) -> Action:
        parts = np.split(np.asarray(values, dtype=np.float32), [7, 8, 15, 16, 18, 22])
        return Action(
            left_arm=parts[0].copy(),
            left_gripper=parts[1].copy(),
            right_arm=parts[2].copy(),
            right_gripper=parts[3].copy(),
            head=parts[4].copy(),
            torso=parts[5].copy(),
            base=parts[6].copy(),
        )

    def reset(self) -> None:
        """Clear all episode inference state."""
        self._timestep = 0
        self._cached_chunk = None
        self._cache_index = 0
        self._history.clear()

    def close(self) -> None:
        """Release model and optimizer resources."""
        if self._closed:
            return
        self.model = None
        self.optimizer = None
        self._cached_chunk = None
        self._history.clear()
        self._closed = True
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
