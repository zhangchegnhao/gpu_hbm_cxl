from __future__ import annotations

from dataclasses import dataclass

from .event import Event


@dataclass(frozen=True)
class ScheduledEvent:
    event: Event
    start_us: float
    end_us: float
    blocking_event: str | None


class EventEngine:
    """Deterministic non-preemptive list scheduler for a topological event list."""

    def run(self, events: tuple[Event, ...]) -> tuple[ScheduledEvent, ...]:
        names = [event.name for event in events]
        if len(set(names)) != len(names):
            raise ValueError("event names must be unique")
        scheduled: dict[str, ScheduledEvent] = {}
        resource_available: dict[str, float] = {}
        resource_owner: dict[str, str] = {}
        result: list[ScheduledEvent] = []

        for event in events:
            missing = [dependency for dependency in event.dependencies if dependency not in scheduled]
            if missing:
                raise ValueError(
                    f"event {event.name} is not topologically ordered; missing dependencies {missing}"
                )
            blockers: list[tuple[float, str]] = [
                (scheduled[dependency].end_us, dependency) for dependency in event.dependencies
            ]
            blockers.extend(
                (resource_available.get(resource, 0.0), resource_owner[resource])
                for resource in event.resources
                if resource in resource_owner
            )
            if blockers:
                start, blocking_event = max(blockers, key=lambda item: (item[0], item[1]))
            else:
                start, blocking_event = 0.0, None
            end = start + event.duration_us
            item = ScheduledEvent(event, start, end, blocking_event)
            result.append(item)
            scheduled[event.name] = item
            for resource in event.resources:
                resource_available[resource] = end
                resource_owner[resource] = event.name
        return tuple(result)

    @staticmethod
    def critical_path(events: tuple[ScheduledEvent, ...]) -> tuple[str, ...]:
        if not events:
            return ()
        by_name = {event.event.name: event for event in events}
        current = max(events, key=lambda event: (event.end_us, event.event.name))
        path = [current.event.name]
        while current.blocking_event is not None:
            current = by_name[current.blocking_event]
            path.append(current.event.name)
        path.reverse()
        return tuple(path)
