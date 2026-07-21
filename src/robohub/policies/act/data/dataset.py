"""Typed ACT dataset and training data helpers."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict

import h5py
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class NormStats(TypedDict):
    action_mean: NDArray[np.float32]
    action_std: NDArray[np.float32]
    qpos_mean: NDArray[np.float32]
    qpos_std: NDArray[np.float32]
    example_qpos: NDArray[np.float32]


Sample = tuple[Tensor, Tensor, Tensor, Tensor]


class EpisodicDataset(Dataset[Sample]):
    """Sample observations and fixed-size future action chunks."""

    def __init__(
        self,
        episode_ids: Sequence[int] | NDArray[np.integer[Any]],
        dataset_dir: str | Path,
        camera_names: Sequence[str],
        norm_stats: NormStats,
        chunk_size: int,
        arm_delay_time: int = 0,
        down_sample: int = 1,
    ) -> None:
        super().__init__()
        if len(episode_ids) == 0:
            raise ValueError("episode_ids must not be empty")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if arm_delay_time < 0:
            raise ValueError("arm_delay_time must be non-negative")
        if down_sample <= 0:
            raise ValueError("down_sample must be positive")
        if not camera_names:
            raise ValueError("camera_names must not be empty")

        self.episode_ids = [int(episode_id) for episode_id in episode_ids]
        self.dataset_dir = Path(dataset_dir)
        self.camera_names = tuple(camera_names)
        self.norm_stats = norm_stats
        self.chunk_size = chunk_size
        self.arm_delay_time = arm_delay_time
        self.down_sample = down_sample

        first_path = self._episode_path(self.episode_ids[0])
        with h5py.File(first_path, "r") as root:
            self.is_sim = bool(root.attrs.get("sim", False))

    def _episode_path(self, episode_id: int) -> Path:
        return self.dataset_dir / f"episode_{episode_id}.hdf5"

    def __len__(self) -> int:
        return len(self.episode_ids)

    def __getitem__(self, index: int) -> Sample:
        episode_id = self.episode_ids[index]
        dataset_path = self._episode_path(episode_id)
        with h5py.File(dataset_path, "r") as root:
            action_dataset = root["action"]
            qpos_dataset = root["observations/qpos"]
            if action_dataset.ndim != 2:
                raise ValueError(
                    f"Expected action rank 2 in {dataset_path}, "
                    f"got {action_dataset.shape}"
                )
            episode_len, action_dim = action_dataset.shape
            if episode_len == 0:
                raise ValueError(f"Episode contains no actions: {dataset_path}")
            if qpos_dataset.shape[0] != episode_len:
                raise ValueError(
                    f"qpos/action length mismatch in {dataset_path}: "
                    f"{qpos_dataset.shape[0]} != {episode_len}"
                )

            start_ts = int(np.random.randint(episode_len))
            qpos = np.asarray(qpos_dataset[start_ts], dtype=np.float32)
            image_dict = {
                camera_name: np.asarray(
                    root[f"observations/images/{camera_name}"][start_ts],
                    dtype=np.uint8,
                )
                for camera_name in self.camera_names
            }

            action_start = max(0, start_ts - self.arm_delay_time)
            raw_indices = (
                action_start
                + np.arange(self.chunk_size, dtype=np.int64) * self.down_sample
            )
            valid_indices = raw_indices[raw_indices < episode_len]
            valid_actions = np.asarray(
                action_dataset[valid_indices], dtype=np.float32
            )

        valid_count = len(valid_indices)
        padded_action = np.zeros(
            (self.chunk_size, action_dim),
            dtype=np.float32,
        )
        padded_action[:valid_count] = valid_actions
        is_pad = np.ones(self.chunk_size, dtype=np.bool_)
        is_pad[:valid_count] = False

        all_cam_images = np.stack(
            [image_dict[camera_name] for camera_name in self.camera_names],
            axis=0,
        )
        image_data = torch.from_numpy(all_cam_images).permute(0, 3, 1, 2).float()
        image_data /= 255.0
        qpos_data = torch.from_numpy(qpos)
        action_data = torch.from_numpy(padded_action)

        action_mean = torch.as_tensor(
            self.norm_stats["action_mean"], dtype=torch.float32
        )
        action_std = torch.as_tensor(
            self.norm_stats["action_std"], dtype=torch.float32
        )
        qpos_mean = torch.as_tensor(
            self.norm_stats["qpos_mean"], dtype=torch.float32
        )
        qpos_std = torch.as_tensor(
            self.norm_stats["qpos_std"], dtype=torch.float32
        )
        action_data = (action_data - action_mean) / action_std
        qpos_data = (qpos_data - qpos_mean) / qpos_std

        return image_data, qpos_data, action_data, torch.from_numpy(is_pad)


def get_norm_stats(
    dataset_dir: str | Path,
    num_episodes: int,
) -> tuple[NormStats, int]:
    """Compute per-dimension statistics from all unpadded episode frames."""
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")

    directory = Path(dataset_dir)
    all_qpos: list[Tensor] = []
    all_actions: list[Tensor] = []
    max_action_len = 0
    example_qpos: NDArray[np.float32] | None = None
    for episode_id in range(num_episodes):
        dataset_path = directory / f"episode_{episode_id}.hdf5"
        with h5py.File(dataset_path, "r") as root:
            qpos = np.asarray(root["observations/qpos"], dtype=np.float32)
            action = np.asarray(root["action"], dtype=np.float32)
        if qpos.ndim != 2 or action.ndim != 2:
            raise ValueError(
                f"Expected rank-2 qpos and action in {dataset_path}, "
                f"got {qpos.shape} and {action.shape}"
            )
        if qpos.shape[0] == 0 or action.shape[0] == 0:
            raise ValueError(f"Episode contains no frames: {dataset_path}")
        if qpos.shape[0] != action.shape[0]:
            raise ValueError(
                f"qpos/action length mismatch in {dataset_path}: "
                f"{qpos.shape[0]} != {action.shape[0]}"
            )
        all_qpos.append(torch.from_numpy(qpos))
        all_actions.append(torch.from_numpy(action))
        max_action_len = max(max_action_len, action.shape[0])
        example_qpos = qpos

    qpos_data = torch.cat(all_qpos, dim=0)
    action_data = torch.cat(all_actions, dim=0)
    action_std, action_mean = torch.std_mean(
        action_data, dim=0, unbiased=False
    )
    qpos_std, qpos_mean = torch.std_mean(qpos_data, dim=0, unbiased=False)
    action_std = torch.clamp(action_std, min=1e-2)
    qpos_std = torch.clamp(qpos_std, min=1e-2)
    if example_qpos is None:
        raise RuntimeError("No qpos data was loaded")

    stats: NormStats = {
        "action_mean": action_mean.numpy().astype(np.float32, copy=False),
        "action_std": action_std.numpy().astype(np.float32, copy=False),
        "qpos_mean": qpos_mean.numpy().astype(np.float32, copy=False),
        "qpos_std": qpos_std.numpy().astype(np.float32, copy=False),
        "example_qpos": example_qpos,
    }
    return stats, max_action_len


def load_data(
    dataset_dir: str | Path,
    num_episodes: int,
    camera_names: Sequence[str],
    batch_size_train: int,
    batch_size_val: int,
    arm_delay_time: int = 0,
    chunk_size: int = 30,
    down_sample: int = 1,
    *,
    num_workers: int = 1,
) -> tuple[
    DataLoader[Sample],
    DataLoader[Sample],
    NormStats,
    bool,
]:
    """Create train and validation loaders with a non-empty split."""
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    if batch_size_train <= 0 or batch_size_val <= 0:
        raise ValueError("batch sizes must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    shuffled_indices = np.random.permutation(num_episodes)
    if num_episodes == 1:
        train_indices = shuffled_indices
        val_indices = shuffled_indices.copy()
    else:
        split_index = max(1, min(num_episodes - 1, int(0.8 * num_episodes)))
        train_indices = shuffled_indices[:split_index]
        val_indices = shuffled_indices[split_index:]

    norm_stats, _ = get_norm_stats(dataset_dir, num_episodes)
    dataset_args = (
        dataset_dir,
        camera_names,
        norm_stats,
        chunk_size,
        arm_delay_time,
        down_sample,
    )
    train_dataset = EpisodicDataset(train_indices, *dataset_args)
    val_dataset = EpisodicDataset(val_indices, *dataset_args)
    loader_options: dict[str, Any] = {
        "pin_memory": True,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        loader_options["prefetch_factor"] = 1
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        **loader_options,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=False,
        **loader_options,
    )
    return train_loader, val_loader, norm_stats, train_dataset.is_sim


def compute_dict_mean(
    epoch_dicts: Sequence[Mapping[str, Tensor]],
) -> dict[str, Tensor]:
    """Average matching tensors across a non-empty sequence of mappings."""
    if not epoch_dicts:
        raise ValueError("epoch_dicts must not be empty")
    return {
        key: torch.stack([item[key] for item in epoch_dicts]).mean(dim=0)
        for key in epoch_dicts[0]
    }


def detach_dict(values: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Detach all tensors in a mapping."""
    return {key: value.detach() for key, value in values.items()}


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
