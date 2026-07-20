"""Policy inference interface."""

from abc import ABC, abstractmethod

from robohub.schemas import Action, Observation


class Policy(ABC):
    """Convert robot observations into actions.

    Stateful policies should clear episode state in :meth:`reset` and release
    external resources in :meth:`close`.
    """

    @abstractmethod
    def infer(self, observation: Observation) -> Action:
        """Infer one action from the latest robot observation."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset state carried between inference steps."""

    def close(self) -> None:
        """Release resources owned by the policy."""

    def __enter__(self) -> "Policy":
        """Return this policy as a managed resource."""
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Close the policy when leaving a context."""
        self.close()
