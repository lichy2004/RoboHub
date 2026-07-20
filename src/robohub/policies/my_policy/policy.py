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
        del observation

        def values(name: str, default: int) -> np.ndarray:
            size = int(self.config.get(name, default))
            if size < 0:
                raise ValueError(f"{name} must be non-negative")
            return np.full(size, self._action_value, dtype=np.float32)

        return Action(
            left_arm=values("arm_size", 7),
            left_gripper=values("gripper_size", 1),
            right_arm=values("arm_size", 7),
            right_gripper=values("gripper_size", 1),
            head=values("head_size", 2),
            torso=values("torso_size", 4),
            base=values("base_size", 3),
        )
