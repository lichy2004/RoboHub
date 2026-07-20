"""Unified workstation-side robot interface."""

from abc import ABC, abstractmethod

from robohub.schemas import Action, Observation


class Robot(ABC):
    @abstractmethod
    def get_observation(self) -> Observation:
        raise NotImplementedError

    @abstractmethod
    def set_action(self, action: Action) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    def __enter__(self) -> "Robot":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
