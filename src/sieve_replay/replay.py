from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import LoadedConfiguration, load_configuration
from .model import MemoryFootprint, estimate_memory_footprint
from .policy import create_policy
from .report import write_decode_results, write_results
from .simulation import EventEngine, ScheduledEvent, build_decode_layer_graph
from .simulation.layer_graph import build_layer_graph
from .timing import (
    AnalyticTimingModel,
    CXLReadConfig,
    RamulatorContentionTable,
    RamulatorContentionTimingModel,
    RamulatorTableTimingModel,
    RamulatorTimingTable,
    RuntimeExpertTimingModel,
)
from .trace import TraceBatch, load_trace_set
from .trace_manifest import TraceManifest
from .types import PlacementDecision


@dataclass(frozen=True)
class ReplayUnit:
    trace: TraceBatch
    decision: PlacementDecision
    memory: MemoryFootprint
    contention: dict[str, Any] | None


@dataclass(frozen=True)
class ReplayResult:
    configuration: LoadedConfiguration
    units: tuple[ReplayUnit, ...]
    events: tuple[ScheduledEvent, ...]
    summary: dict[str, Any]

    @property
    def trace(self) -> TraceBatch:
        return self._single_unit.trace

    @property
    def decision(self) -> PlacementDecision:
        return self._single_unit.decision

    @property
    def memory(self) -> MemoryFootprint:
        return self._single_unit.memory

    @property
    def _single_unit(self) -> ReplayUnit:
        if len(self.units) != 1:
            raise ValueError("single-layer compatibility property used for a decode replay")
        return self.units[0]


def _create_timing(configuration: LoadedConfiguration) -> AnalyticTimingModel:
    runtime_timing = _load_runtime_timing(configuration)
    cxl_config = CXLReadConfig.from_raw(configuration.experiment.raw, configuration.hardware)
    if configuration.hardware.timing_backend == "analytic-v0":
        return AnalyticTimingModel(
            configuration.model, configuration.hardware, runtime_timing, cxl_config
        )
    if configuration.hardware.timing_backend == "ramulator-table-v0":
        table_path = configuration.experiment.pim_timing_table_path
        if table_path is None:
            raise ValueError("ramulator-table-v0 requires experiment field pim_timing_table")
        return RamulatorTableTimingModel(
            configuration.model,
            configuration.hardware,
            RamulatorTimingTable.load(table_path),
            runtime_timing,
            cxl_config,
        )
    if configuration.hardware.timing_backend == "ramulator-contention-v1":
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
        return RamulatorContentionTimingModel(
            configuration.model,
            configuration.hardware,
            RamulatorTimingTable.load(isolated_path),
            contention_table,
            runtime_timing,
            cxl_config,
        )
    raise ValueError(f"unsupported timing backend: {configuration.hardware.timing_backend}")


def _load_runtime_timing(
    configuration: LoadedConfiguration,
) -> RuntimeExpertTimingModel | None:
    calibration_path = configuration.experiment.runtime_calibration_path
    if calibration_path is None:
        return None
    cycle_path = configuration.experiment.ramulator_cycle_config_path
    if cycle_path is None:
        raise ValueError("runtime_calibration requires ramulator_cycle_config")
    calibration = RuntimeExpertTimingModel.load(calibration_path)
    calibration.validate_inputs(
        configuration.experiment.model_path,
        configuration.experiment.hardware_path,
        cycle_path,
    )
    return calibration


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
    trace_set = load_trace_set(
        configuration.experiment.trace_path,
        configuration.model,
        configuration.experiment.layers,
        configuration.experiment.steps,
    )
    manifest_path = configuration.experiment.trace_manifest_path
    is_real_trace = "real" in configuration.experiment.trace_path.parts
    if is_real_trace and manifest_path is None:
        raise ValueError("traces under traces/real require an experiment trace_manifest")
    if manifest_path is not None:
        TraceManifest.load_and_validate(
            manifest_path,
            configuration.experiment.trace_path,
            configuration.model,
            trace_set,
        )

    timing = _create_timing(configuration)
    policy = create_policy(policy_name, timing)
    if configuration.experiment.is_single_layer_step:
        trace = trace_set.batches[0]
        decision = policy.place(trace)
        if hasattr(timing, "last_contention_report"):
            timing.last_contention_report = None
        events = EventEngine().run(
            build_layer_graph(trace, decision, timing, configuration.experiment.kv_read_mode)
        )
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
        unit = ReplayUnit(trace, decision, memory, contention)
        return ReplayResult(configuration, (unit,), events, summary)

    graph = []
    units: list[ReplayUnit] = []
    previous_layer_tail: str | None = None
    for trace in trace_set.batches:
        decision = policy.place(trace)
        if hasattr(timing, "last_contention_report"):
            timing.last_contention_report = None
        layer_graph, previous_layer_tail = build_decode_layer_graph(
            trace,
            decision,
            timing,
            previous_layer_tail,
            configuration.experiment.kv_read_mode,
        )
        graph.extend(layer_graph)
        memory = estimate_memory_footprint(configuration.model, trace)
        contention = getattr(timing, "last_contention_report", None)
        if contention is not None:
            contention = dict(contention)
        units.append(ReplayUnit(trace, decision, memory, contention))
    events = EventEngine().run(tuple(graph))
    summary = write_decode_results(
        output_dir,
        configuration,
        tuple(unit.trace for unit in units),
        tuple(unit.decision for unit in units),
        events,
        tuple(unit.memory for unit in units),
        tuple(unit.contention for unit in units),
    )
    return ReplayResult(configuration, tuple(units), events, summary)
