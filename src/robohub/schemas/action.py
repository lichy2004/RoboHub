"""Action schema exchanged by RobotServer and RobotClient."""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Action:
    left_arm: np.ndarray
    left_gripper: np.ndarray
    right_arm: np.ndarray
    right_gripper: np.ndarray
    torso: np.ndarray
    head: np.ndarray
    base: np.ndarray