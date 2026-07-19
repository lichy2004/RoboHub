"""Policy inference interface."""

from abc import ABC, abstractmethod

from robohub.schemas import Action, Observation


class Policy(ABC):
    @abstractmethod
    def infer(self, observation: Observation) -> Action:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> "Policy":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
