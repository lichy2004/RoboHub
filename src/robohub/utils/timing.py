"""Timing helpers for control and profiling code."""

from dataclasses import dataclass, field
from time import monotonic_ns


@dataclass(slots=True)
class Timer:
    started_ns: int = field(default_factory=monotonic_ns)

    def reset(self) -> None:
        self.started_ns = monotonic_ns()

    @property
    def elapsed_ns(self) -> int:
        return monotonic_ns() - self.started_ns

    @property
    def elapsed_seconds(self) -> float:
        return self.elapsed_ns / 1_000_000_000
