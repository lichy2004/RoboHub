from __future__ import annotations

from pathlib import Path
from threading import Lock

import numpy as np

from robohub.robots.base import Robot
from robohub.utils.config import load_config
from robohub.utils.types import ACTION_NAMES, Action, Observation


class FakeRobot(Robot):
    """In-memory robot for development and network integration tests."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        path = config_path or Path(__file__).parent / "configs" / "default.yaml"
        config = load_config(path)
        super().__init__(config)

    def reset(self) -> None:
        return

    def get_observation(self) -> Observation:
        height, width = self.config["image_size"]
        rgb = {
            name: np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            for name in ("head", "wrist_left", "wrist_right")
        }
        depth = {
            name: np.random.random((height, width, 1)).astype(np.float32)
            for name in ("head", "wrist_left", "wrist_right")
        }
        joints_position = np.random.random(self.joints_num)
        joints_velocity = np.random.random(self.joints_num)
        joints_torque = np.random.random(self.joints_num)
        return {
            "rgb": rgb,
            "depth": depth,
            "joints_position": joints_position,
            "joints_velocity": joints_velocity,
            "joints_torque": joints_torque,
        }

    def set_action(self, action: Action) -> None:
        return

    def close(self) -> None:
        return
