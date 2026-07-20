"""Project policy implementation."""

from collections.abc import Mapping
from typing import Any

import numpy as np

from robohub.policies import Policy
from robohub.schemas import Action, Observation


class MyPolicy(Policy):
    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self._action_value = float(self.config.get("action_value", 0.0))

    def infer(self, observation: Observation) -> Action:
        joint_count = observation.joints_position.size
        zeros = np.full(joint_count, self._action_value, dtype=np.float32)
        empty = np.zeros(0, dtype=np.float32)
        return Action(
            zeros,
            empty,
            zeros.copy(),
            empty.copy(),
            empty.copy(),
            empty.copy(),
            empty.copy(),
        )
