"""Observation processing interfaces."""

from abc import ABC, abstractmethod

import numpy as np

from robohub.schemas import Observation


class Processor(ABC):
    """Workstation-side interface for on-demand observation processing."""

    @abstractmethod
    def get_point_cloud(
        self, observation: Observation, camera_name: str
    ) -> np.ndarray:
        raise NotImplementedError