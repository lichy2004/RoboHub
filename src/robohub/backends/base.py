"""Hardware backend interface."""

from typing import Protocol

from robohub.schemas import Action, Observation, RawObservation


class RobotBackend(Protocol):
    def get_observation(self) -> Observation | RawObservation: ...

    def set_action(self, action: Action) -> None: ...

    def close(self) -> None: ...
