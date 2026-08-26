from __future__ import annotations

from ..timing import AnalyticTimingModel
from ..trace import TraceBatch
from ..types import PlacementDecision
from .event import Event
from .layer_graph import build_layer_graph


def build_decode_graph(
    units: tuple[tuple[TraceBatch, PlacementDecision], ...],
    timing: AnalyticTimingModel,
) -> tuple[Event, ...]:
    events: list[Event] = []
    previous_layer_tail: str | None = None
    for trace, decision in units:
        layer_events, previous_layer_tail = build_decode_layer_graph(
            trace, decision, timing, previous_layer_tail
        )
        events.extend(layer_events)
    return tuple(events)


def build_decode_layer_graph(
    trace: TraceBatch,
    decision: PlacementDecision,
    timing: AnalyticTimingModel,
    previous_layer_tail: str | None,
) -> tuple[tuple[Event, ...], str]:
    prefix = f"step{trace.step}.layer{trace.layer}."
    namespaced: list[Event] = []
    for event in build_layer_graph(trace, decision, timing):
        dependencies = tuple(prefix + dependency for dependency in event.dependencies)
        if event.name == "norm1" and previous_layer_tail is not None:
            dependencies = (previous_layer_tail,)
        namespaced.append(
            Event(
                name=prefix + event.name,
                category=event.category,
                resources=event.resources,
                dependencies=dependencies,
                duration_us=event.duration_us,
                description=event.description,
                step=trace.step,
                layer=trace.layer,
            )
        )
    return tuple(namespaced), prefix + "residual2"
