"""Observation schema exchanged by RobotServer and RobotClient."""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Observation:
    rgb: dict[str, np.ndarray]
    depth: dict[str, np.ndarray]
    joints_position: np.ndarray
    joints_velocity: np.ndarray
    joints_torque: np.ndarray
