from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import LoadedConfiguration, load_configuration
from .model import MemoryFootprint, estimate_memory_footprint
from .policy import create_policy
from .report import write_results
from .simulation import EventEngine, ScheduledEvent
from .simulation.layer_graph import build_layer_graph
from .timing import (
    AnalyticTimingModel,
    RamulatorContentionTable,
    RamulatorContentionTimingModel,
    RamulatorTableTimingModel,
    RamulatorTimingTable,
)
from .trace import TraceBatch, load_trace
from .types import PlacementDecision


@dataclass(frozen=True)
class ReplayResult:
    configuration: LoadedConfiguration
    trace: TraceBatch
    decision: PlacementDecision
    events: tuple[ScheduledEvent, ...]
    memory: MemoryFootprint
    summary: dict[str, Any]


def run_experiment(
    experiment_path: str | Path,
    policy_name: str,
    output_dir: str | Path,
) -> ReplayResult:
    configuration = load_configuration(experiment_path)
    if policy_name not in configuration.experiment.policies:
        raise ValueError(
            f"policy {policy_name!r} is not enabled by experiment; "
            f"allowed: {', '.join(configuration.experiment.policies)}"
        )
    trace = load_trace(
        configuration.experiment.trace_path,
        configuration.model,
        configuration.experiment.layer,
        configuration.experiment.step,
    )
    if configuration.hardware.timing_backend == "analytic-v0":
        timing = AnalyticTimingModel(configuration.model, configuration.hardware)
    elif configuration.hardware.timing_backend == "ramulator-table-v0":
        table_path = configuration.experiment.pim_timing_table_path
        if table_path is None:
            raise ValueError("ramulator-table-v0 requires experiment field pim_timing_table")
        timing = RamulatorTableTimingModel(
            configuration.model,
            configuration.hardware,
            RamulatorTimingTable.load(table_path),
        )
    elif configuration.hardware.timing_backend == "ramulator-contention-v1":
        isolated_path = configuration.experiment.pim_timing_table_path
        contention_path = configuration.experiment.contention_timing_table_path
        cycle_config_path = configuration.experiment.ramulator_cycle_config_path
        if isolated_path is None or contention_path is None or cycle_config_path is None:
            raise ValueError(
                "ramulator-contention-v1 requires pim_timing_table, contention_timing_table, "
                "and ramulator_cycle_config"
            )
        contention_table = RamulatorContentionTable.load(contention_path)
        contention_table.validate_inputs(
            isolated_path,
            configuration.experiment.trace_path,
            cycle_config_path,
            configuration.experiment.model_path,
            configuration.experiment.hardware_path,
        )
        timing = RamulatorContentionTimingModel(
            configuration.model,
            configuration.hardware,
            RamulatorTimingTable.load(isolated_path),
            contention_table,
        )
    else:
        raise ValueError(f"unsupported timing backend: {configuration.hardware.timing_backend}")
    policy = create_policy(policy_name, timing)
    decision = policy.place(trace)
    graph = build_layer_graph(trace, decision, timing)
    events = EventEngine().run(graph)
    memory = estimate_memory_footprint(configuration.model, trace)
    contention = getattr(timing, "last_contention_report", None)
    summary = write_results(
        output_dir,
        configuration,
        trace,
        decision,
        events,
        memory,
        contention=contention,
    )
    return ReplayResult(configuration, trace, decision, events, memory, summary)
