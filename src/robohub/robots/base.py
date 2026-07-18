from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Self

from robohub.utils.types import Action, Observation


class Robot(ABC):
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = MappingProxyType(dict(config))
        name = self.config["name"]
        joints_order = self.config["joints_order"]
        joints_num = self.config["joints_num"]
        if joints_num != len(joints_order):
            raise ValueError("joints_num must match the length of joints_order")

        self.name = name
        self.joints_order = tuple(joints_order)
        self.joints_num = joints_num

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def get_observation(self) -> Observation:
        pass

    @abstractmethod
    def set_action(self, action: Action) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass