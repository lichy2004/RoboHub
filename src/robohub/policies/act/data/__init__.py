"""ACT data conversion and loading interfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .convert_astribot import convert_episode, recorded_to_robohub_qpos
    from .dataset import EpisodicDataset

_CONVERSION_EXPORTS = {
    "convert_episode",
    "recorded_to_robohub_qpos",
}
_DATASET_EXPORTS = {
    "EpisodicDataset",
    "compute_dict_mean",
    "detach_dict",
    "get_norm_stats",
    "load_data",
    "set_seed",
}

__all__ = [
    "EpisodicDataset",
    "compute_dict_mean",
    "convert_episode",
    "detach_dict",
    "get_norm_stats",
    "load_data",
    "recorded_to_robohub_qpos",
    "set_seed",
]


def __getattr__(name: str) -> Any:
    if name in _CONVERSION_EXPORTS:
        from . import convert_astribot

        return getattr(convert_astribot, name)
    if name in _DATASET_EXPORTS:
        from . import dataset

        return getattr(dataset, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
