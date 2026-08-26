from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    name: str
    category: str
    resources: tuple[str, ...]
    dependencies: tuple[str, ...]
    duration_us: float
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("event name cannot be empty")
        if self.duration_us < 0:
            raise ValueError(f"event {self.name} has negative duration")
        if len(set(self.resources)) != len(self.resources):
            raise ValueError(f"event {self.name} repeats a resource")
