"""Project robot SDK and backend adapter."""

import numpy as np

from robohub.schemas import Action, Observation


class MySDK:
    def __init__(self) -> None:
        self.received_action: Action | None = None
        self._rng = np.random.default_rng()

    def get_observation(self) -> Observation:
        return Observation(
            rgb={
                "head": self._random_image(),
                "wrist_left": self._random_image(),
                "wrist_right": self._random_image(),
            },
            depth={"head": self._rng.random((480, 640, 1), dtype=np.float32)},
            joints_position=np.zeros(4, dtype=np.float32),
            joints_velocity=np.zeros(4, dtype=np.float32),
            joints_torque=np.zeros(4, dtype=np.float32),
        )

    def _random_image(self) -> np.ndarray:
        return self._rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)

    def set_action(self, action: Action) -> None:
        self.received_action = action

    def close(self) -> None:
        pass


class MyRobotBackend:
    def __init__(self) -> None:
        self.sdk = MySDK()

    def get_observation(self) -> Observation:
        return self.sdk.get_observation()

    def set_action(self, action: Action) -> None:
        self.sdk.set_action(action)

    def close(self) -> None:
        self.sdk.close()
