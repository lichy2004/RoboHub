from __future__ import annotations

from pathlib import Path

import numpy as np

from robohub.policies.base import Policy
from robohub.utils.config import load_config
from robohub.utils.types import ACTION_NAMES, Action, Observation


class FakePolicy(Policy):
    """Deterministic policy for development and network integration tests."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        path = config_path or Path(__file__).parent / "config" / "default.yaml"
        super().__init__(load_config(path))

    def load_model(self) -> None:
        self.model = None

    def encode_obs(self, obs: Observation) -> dict[str, np.ndarray]:
        obs = obs
        return 

    def get_action(self, obs: Observation) -> Action:
        dimensions = self.config["action_dimensions"]
        action = {
            name: np.zeros(int(dimensions[name]), dtype=np.float32)
            for name in ACTION_NAMES
        }
        return action

    def close(self) -> None:
        return
