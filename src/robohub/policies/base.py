from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from robohub.utils.types import Action, Observation


class Policy(ABC):
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = MappingProxyType(dict(config))
        self.model: Any = None

    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def encode_obs(self, obs: Observation) -> Any:
        pass

    @abstractmethod
    def get_action(self, obs: Observation) -> Action:
        pass

    def close(self) -> None:
        self.model = None
