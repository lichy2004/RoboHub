from __future__ import annotations

from typing import TypedDict

import numpy as np


CAMERA_NAMES = (
    "head",
    "wrist_left",
    "wrist_right",
)
ACTION_NAMES = (
    "left_arm",
    "left_gripper",
    "right_arm",
    "right_gripper",
    "torso",
    "head",
    "base",
)


class CameraImages(TypedDict):
    head: np.ndarray
    wrist_left: np.ndarray
    wrist_right: np.ndarray


class Observation(TypedDict):
    rgb: CameraImages
    depth: CameraImages
    joints_position: np.ndarray
    joints_velocity: np.ndarray
    joints_torque: np.ndarray


class Action(TypedDict):
    left_arm: np.ndarray
    left_gripper: np.ndarray
    right_arm: np.ndarray
    right_gripper: np.ndarray
    torso: np.ndarray
    head: np.ndarray
    base: np.ndarray
