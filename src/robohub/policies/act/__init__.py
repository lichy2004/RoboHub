"""ACT runtime policy."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from robohub.policies.act.policy import ACTPolicy

__all__ = ["ACTPolicy"]


def __getattr__(name: str) -> Any:
    if name == "ACTPolicy":
        from robohub.policies.act.policy import ACTPolicy

        return ACTPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
