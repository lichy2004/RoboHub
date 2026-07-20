# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Tensor and distributed helpers adapted from torchvision references."""

from __future__ import annotations

import pickle
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
import torch.distributed as dist
import torchvision
from torch import Tensor


class SmoothedValue:
    """Track recent values and their global average."""

    def __init__(self, window_size: int = 20, fmt: str | None = None) -> None:
        self.deque: deque[float] = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt or "{median:.4f} ({global_avg:.4f})"

    def update(self, value: float, count: int = 1) -> None:
        """Add one observation."""
        self.deque.append(value)
        self.count += count
        self.total += value * count

    def synchronize_between_processes(
        self,
        device: torch.device | str,
    ) -> None:
        """Synchronize totals while retaining each process's local window."""
        if not is_dist_avail_and_initialized():
            return
        totals = torch.tensor(
            [self.count, self.total],
            dtype=torch.float64,
            device=device,
        )
        dist.barrier()
        dist.all_reduce(totals)
        count, total = totals.tolist()
        self.count = int(count)
        self.total = total

    @property
    def median(self) -> float:
        return torch.tensor(list(self.deque)).median().item()

    @property
    def avg(self) -> float:
        return torch.tensor(list(self.deque), dtype=torch.float32).mean().item()

    @property
    def global_avg(self) -> float:
        return self.total / self.count

    @property
    def max(self) -> float:
        return max(self.deque)

    @property
    def value(self) -> float:
        return self.deque[-1]

    def __str__(self) -> str:
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


def all_gather(
    data: Any,
    device: torch.device | str,
) -> list[Any]:
    """Gather arbitrary picklable values from every distributed process."""
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    serialized = pickle.dumps(data)
    tensor = torch.tensor(list(serialized), dtype=torch.uint8, device=device)
    local_size = torch.tensor([tensor.numel()], device=device)
    size_tensors = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(
        world_size
    )]
    dist.all_gather(size_tensors, local_size)
    sizes = [int(size.item()) for size in size_tensors]
    max_size = max(sizes)
    if tensor.numel() < max_size:
        padding = torch.empty(
            max_size - tensor.numel(),
            dtype=torch.uint8,
            device=device,
        )
        tensor = torch.cat((tensor, padding))

    gathered = [
        torch.empty(max_size, dtype=torch.uint8, device=device)
        for _ in range(world_size)
    ]
    dist.all_gather(gathered, tensor)
    return [
        pickle.loads(bytes(value.cpu().tolist()[:size]))
        for size, value in zip(sizes, gathered, strict=True)
    ]


def reduce_dict(
    input_dict: Mapping[str, Tensor],
    average: bool = True,
) -> dict[str, Tensor]:
    """Reduce tensors with matching names across distributed processes."""
    if get_world_size() < 2:
        return dict(input_dict)
    with torch.no_grad():
        names = sorted(input_dict)
        values = torch.stack([input_dict[name] for name in names])
        dist.all_reduce(values)
        if average:
            values /= get_world_size()
        return dict(zip(names, values, strict=True))


class NestedTensor:
    """A tensor batch paired with a padding mask."""

    def __init__(self, tensors: Tensor, mask: Tensor | None) -> None:
        self.tensors = tensors
        self.mask = mask

    def to(self, device: torch.device | str) -> NestedTensor:
        """Move both tensor and mask to a device."""
        mask = self.mask.to(device) if self.mask is not None else None
        return NestedTensor(self.tensors.to(device), mask)

    def decompose(self) -> tuple[Tensor, Tensor | None]:
        """Return the underlying tensor and mask."""
        return self.tensors, self.mask

    def __repr__(self) -> str:
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: Sequence[Tensor]) -> NestedTensor:
    """Pad CHW images into one batch and create the padding mask."""
    if not tensor_list or tensor_list[0].ndim != 3:
        raise ValueError("expected a non-empty sequence of CHW tensors")
    max_size = _max_by_axis([list(image.shape) for image in tensor_list])
    batch_size = len(tensor_list)
    channels, height, width = max_size
    tensor = torch.zeros(
        (batch_size, channels, height, width),
        dtype=tensor_list[0].dtype,
        device=tensor_list[0].device,
    )
    mask = torch.ones(
        (batch_size, height, width),
        dtype=torch.bool,
        device=tensor_list[0].device,
    )
    for image, padded_image, padded_mask in zip(
        tensor_list,
        tensor,
        mask,
        strict=True,
    ):
        padded_image[: image.shape[0], : image.shape[1], : image.shape[2]].copy_(image)
        padded_mask[: image.shape[1], : image.shape[2]] = False
    return NestedTensor(tensor, mask)


def _max_by_axis(sizes: list[list[int]]) -> list[int]:
    maximums = sizes[0].copy()
    for size in sizes[1:]:
        maximums = [max(current, candidate) for current, candidate in zip(
            maximums,
            size,
            strict=True,
        )]
    return maximums


def collate_fn(batch: Iterable[tuple[Any, ...]]) -> tuple[Any, ...]:
    """Collate samples whose first item is an image tensor."""
    columns = list(zip(*batch))
    columns[0] = nested_tensor_from_tensor_list(columns[0])
    return tuple(columns)


def is_dist_avail_and_initialized() -> bool:
    """Return whether torch distributed is ready."""
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """Return the distributed world size, or one outside distributed mode."""
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_rank() -> int:
    """Return the distributed rank, or zero outside distributed mode."""
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process() -> bool:
    """Return whether the current process is rank zero."""
    return get_rank() == 0


def save_on_master(*args: Any, **kwargs: Any) -> None:
    """Save with torch only on rank zero."""
    if is_main_process():
        torch.save(*args, **kwargs)


def interpolate(
    inputs: Tensor,
    size: int | tuple[int, int] | None = None,
    scale_factor: float | tuple[float, float] | None = None,
    mode: str = "nearest",
    align_corners: bool | None = None,
) -> Tensor:
    """Interpolate tensors through torchvision's empty-batch-safe wrapper."""
    return torchvision.ops.misc.interpolate(
        inputs,
        size,
        scale_factor,
        mode,
        align_corners,
    )
