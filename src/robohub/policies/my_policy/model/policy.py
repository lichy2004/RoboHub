"""Basic project policy implementation."""

from collections.abc import Mapping
from typing import Any

import numpy as np

from robohub.policies import Policy
from robohub.schemas import Action, Observation


class MyPolicy(Policy):
    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    def infer(self, observation: Observation) -> Action:
        value = float(self.config.get("action_value", 0.0))
        arm_size = int(
            self.config.get("arm_size", observation.joints_position.size // 2)
        )

        def values(size: int) -> np.ndarray:
            return np.full(size, value, dtype=np.float32)

        return Action(
            left_arm=values(arm_size),
            left_gripper=values(1),
            right_arm=values(arm_size),
            right_gripper=values(1),
            torso=values(int(self.config.get("torso_size", 1))),
            head=values(int(self.config.get("head_size", 2))),
            base=values(int(self.config.get("base_size", 3))),
        )
