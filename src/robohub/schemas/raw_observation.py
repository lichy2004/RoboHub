"""Raw robot observations exchanged before workstation-side image decoding."""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class RawImage:
    data: bytes
    width: int
    height: int


@dataclass(slots=True)
class RawObservation:
    rgb: dict[str, RawImage]
    depth: dict[str, RawImage]
    joints_position: np.ndarray
    joints_velocity: np.ndarray
    joints_torque: np.ndarray
