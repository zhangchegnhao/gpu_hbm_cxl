from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..config import LoadedConfiguration
from ..model import MemoryFootprint, classify_capacity, estimate_memory_footprint
from ..simulation import EventEngine, ScheduledEvent
from ..trace import TraceBatch
from ..types import PlacementDecision
from ..timing import CXLReadConfig


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_hashes(configuration: LoadedConfiguration) -> dict[str, str]:
    hashes = {
        "model_sha256": sha256_file(configuration.experiment.model_path),
        "hardware_sha256": sha256_file(configuration.experiment.hardware_path),
        "trace_sha256": sha256_file(configuration.experiment.trace_path),
    }
    optional_paths = {
        "pim_timing_table_sha256": configuration.experiment.pim_timing_table_path,
        "contention_timing_table_sha256": configuration.experiment.contention_timing_table_path,
        "ramulator_cycle_config_sha256": configuration.experiment.ramulator_cycle_config_path,
        "trace_manifest_sha256": configuration.experiment.trace_manifest_path,
        "runtime_calibration_sha256": configuration.experiment.runtime_calibration_path,
    }
    for name, path in optional_paths.items():
        if path is not None:
            hashes[name] = sha256_file(path)
    return hashes


def _result_classification(timing_backend: str) -> str:
    classifications = {
        "analytic-v0": "functional analytic estimate; not a formal performance result",
        "ramulator-table-v0": (
            "Ramulator cycle-v0 PIM timing with analytic-v0 GPU timing; calibration required"
        ),
        "ramulator-contention-v1": (
            "Ramulator cycle-v1 mixed expert-memory timing with analytic GPU compute; "
            "contention-aware but not GPU-calibrated"
        ),
    }
    return classifications[timing_backend]


def _kv_read_summary(
    configuration: LoadedConfiguration,
    traces: tuple[TraceBatch, ...],
    critical_duration_by_category: dict[str, float],
) -> dict[str, Any]:
    mode = configuration.experiment.kv_read_mode
    total_bytes = (
        sum(
            2
            * sum(trace.context_lengths)
            * configuration.model.num_key_value_heads
            * configuration.model.head_dim
            * configuration.model.dtype_bytes
            for trace in traces
        )
        if mode != "disabled"
        else 0
    )
    return {
        "mode": mode,
        "model": "analytic-local-hbm-v1" if mode != "disabled" else "disabled",
        "execution": (
            "serial_dependency"
            if mode == "serial-local-hbm-v1"
            else "parallel_read_and_compute_join"
            if mode == "overlap-local-hbm-v1"
            else "not_modeled"
        ),
        "total_read_bytes": total_bytes,
        "critical_path_latency_us": _round(
            critical_duration_by_category.get("kv_read", 0.0)
        ),
    }


def _cxl_summary(
    configuration: LoadedConfiguration,
    traces: tuple[TraceBatch, ...],
    critical_duration_by_category: dict[str, float],
) -> dict[str, Any]:
    raw = configuration.experiment.raw
    mode = raw.get("cxl_mode", "disabled")
    if mode == "disabled":
        return {
            "mode": "disabled",
            "model": "disabled",
            "capacity_state": "not_modeled",
            "feasible": True,
            "local_capacity_bytes": 0,
            "capacity_bytes": 0,
            "peak_spill_bytes": 0,
            "total_read_bytes": 0,
            "critical_path_latency_us": 0.0,
        }
    cxl = CXLReadConfig.from_raw(raw, configuration.hardware)
    admissions = [
        classify_capacity(
            estimate_memory_footprint(configuration.model, trace).total_bytes,
            cxl.local_capacity_bytes,
            cxl.capacity_bytes,
        )
        for trace in traces
    ]
    peak = max(admissions, key=lambda row: int(row["resident_bytes"]) + int(row["spill_bytes"]) + int(row["unallocated_bytes"]))
    total_read_bytes = 0.0
    for trace, admission in zip(traces, admissions):
        memory = estimate_memory_footprint(configuration.model, trace)
        layer_bytes = (
            2
            * sum(trace.context_lengths)
            * configuration.model.num_key_value_heads
            * configuration.model.head_dim
            * configuration.model.dtype_bytes
        )
        fraction = float(admission["spill_bytes"]) / memory.kv_cache_bytes if memory.kv_cache_bytes else 0.0
        total_read_bytes += layer_bytes * fraction
    return {
        "mode": mode,
        "model": "analytic-cxl-memory-only-v1",
        "execution": (
            "serial_dependency"
            if configuration.experiment.kv_read_mode == "serial-local-hbm-v1"
            else "parallel_read_and_compute_join"
        ),
        "local_capacity_bytes": cxl.local_capacity_bytes,
        "capacity_bytes": cxl.capacity_bytes,
        "bandwidth_gb_s": _round(cxl.bandwidth_bytes_per_second / 1e9),
        "latency_us": _round(cxl.latency_us),
        "capacity_state": peak["state"],
        "feasible": bool(peak["feasible"]),
        "peak_spill_bytes": int(max(int(row["spill_bytes"]) for row in admissions)),
        "total_read_bytes": int(round(total_read_bytes)),
        "critical_path_latency_us": _round(
            critical_duration_by_category.get("cxl_kv_read", 0.0)
        ),
    }


def _round(value: float) -> float:
    return round(value, 9)


def _round_nested(value: Any) -> Any:
    if isinstance(value, float):
        return _round(value)
    if isinstance(value, dict):
        return {key: _round_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_nested(item) for item in value]
    return value


def build_summary(
    configuration: LoadedConfiguration,
    trace: TraceBatch,
    decision: PlacementDecision,
    events: tuple[ScheduledEvent, ...],
    memory: MemoryFootprint,
    contention: dict[str, Any] | None = None,
) -> dict[str, Any]:
    duration_by_category: dict[str, float] = defaultdict(float)
    for event in events:
        duration_by_category[event.event.category] += event.event.duration_us
    critical_path = EventEngine.critical_path(events)
    critical_set = set(critical_path)
    critical_duration_by_category: dict[str, float] = defaultdict(float)
    for event in events:
        if event.event.name in critical_set:
            critical_duration_by_category[event.event.category] += event.event.duration_us
    total_latency = max((event.end_us for event in events), default=0.0)
    capacity_bytes = int(configuration.hardware.hbm_pim_capacity_gb * 1e9)
    loads = {load.expert_id: load.token_count for load in trace.expert_loads}
    input_hashes = _input_hashes(configuration)
    summary = {
        "experiment": configuration.experiment.name,
        "policy": decision.policy,
        "timing_backend": configuration.hardware.timing_backend,
        "result_classification": _result_classification(
            configuration.hardware.timing_backend
        ),
        "layer": trace.layer,
        "step": trace.step,
        "batch_size": trace.batch_size,
        "context_lengths": list(trace.context_lengths),
        "total_expert_assignments": trace.total_expert_assignments,
        "total_latency_us": _round(total_latency),
        "estimated_policy_objective_us": _round(decision.estimated_objective_us),
        "event_duration_us_by_category": {
            key: _round(value) for key, value in sorted(duration_by_category.items())
        },
        "critical_path": list(critical_path),
        "critical_path_duration_us_by_category": {
            key: _round(value) for key, value in sorted(critical_duration_by_category.items())
        },
        "placement": {
            "attention": decision.attention_target.value,
            "gpu_experts": list(decision.gpu_experts),
            "pim_experts": list(decision.pim_experts),
            "gpu_expert_tokens": sum(loads[expert] for expert in decision.gpu_experts),
            "pim_expert_tokens": sum(loads[expert] for expert in decision.pim_experts),
            "tokens_per_active_expert": {str(key): value for key, value in sorted(loads.items())},
        },
        "memory": {
            "model_weights_bytes": memory.model_weights_bytes,
            "kv_cache_bytes": memory.kv_cache_bytes,
            "activation_bytes": memory.activation_bytes,
            "total_bytes": memory.total_bytes,
            "capacity_bytes": capacity_bytes,
            "feasible": memory.total_bytes <= capacity_bytes,
        },
        "input_hashes": input_hashes,
        "kv_read": _kv_read_summary(
            configuration, (trace,), critical_duration_by_category
        ),
        "cxl": _cxl_summary(configuration, (trace,), critical_duration_by_category),
    }
    if contention is not None:
        summary["contention"] = {
            key: _round(value) if isinstance(value, float) else value
            for key, value in sorted(contention.items())
        }
    if decision.search_report is not None:
        summary["placement_search"] = _round_nested(decision.search_report)
    return summary


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _bounds(events: list[ScheduledEvent]) -> tuple[float, float, float]:
    if not events:
        return 0.0, 0.0, 0.0
    start = min(event.start_us for event in events)
    end = max(event.end_us for event in events)
    return start, end, end - start


def _contention_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = sorted(
        key
        for key in rows[0]
        if key not in {"step", "layer", "dual_row_buffer"}
        and all(
            isinstance(row.get(key), (int, float))
            and not isinstance(row.get(key), bool)
            for row in rows
        )
    )
    totals = {key: sum(row[key] for row in rows) for key in numeric_keys}
    return {
        "layer_records": len(rows),
        "dual_row_buffer": all(row.get("dual_row_buffer") is True for row in rows),
        "totals": _round_nested(totals),
        "means": _round_nested(
            {key: totals[key] / len(rows) for key in numeric_keys}
        ),
    }


def build_decode_summary(
    configuration: LoadedConfiguration,
    traces: tuple[TraceBatch, ...],
    decisions: tuple[PlacementDecision, ...],
    events: tuple[ScheduledEvent, ...],
    memories: tuple[MemoryFootprint, ...],
    contentions: tuple[dict[str, Any] | None, ...],
) -> dict[str, Any]:
    if not traces or len(traces) != len(decisions) or len(traces) != len(memories):
        raise ValueError("decode summary inputs must contain aligned non-empty layer records")
    duration_by_category: dict[str, float] = defaultdict(float)
    for event in events:
        duration_by_category[event.event.category] += event.event.duration_us
    critical_path = EventEngine.critical_path(events)
    critical_set = set(critical_path)
    critical_duration_by_category: dict[str, float] = defaultdict(float)
    for event in events:
        if event.event.name in critical_set:
            critical_duration_by_category[event.event.category] += event.event.duration_us

    events_by_layer: dict[tuple[int, int], list[ScheduledEvent]] = defaultdict(list)
    events_by_step: dict[int, list[ScheduledEvent]] = defaultdict(list)
    for event in events:
        if event.event.step is None or event.event.layer is None:
            raise ValueError("decode events must include step and layer annotations")
        events_by_layer[(event.event.step, event.event.layer)].append(event)
        events_by_step[event.event.step].append(event)

    layer_rows: list[dict[str, Any]] = []
    total_gpu_experts = 0
    total_pim_experts = 0
    total_gpu_tokens = 0
    total_pim_tokens = 0
    for trace, decision in zip(traces, decisions):
        loads = {load.expert_id: load.token_count for load in trace.expert_loads}
        start, end, duration = _bounds(events_by_layer[(trace.step, trace.layer)])
        gpu_tokens = sum(loads[expert] for expert in decision.gpu_experts)
        pim_tokens = sum(loads[expert] for expert in decision.pim_experts)
        total_gpu_experts += len(decision.gpu_experts)
        total_pim_experts += len(decision.pim_experts)
        total_gpu_tokens += gpu_tokens
        total_pim_tokens += pim_tokens
        layer_rows.append(
            {
                "step": trace.step,
                "layer": trace.layer,
                "start_us": _round(start),
                "end_us": _round(end),
                "latency_us": _round(duration),
                "attention_target": decision.attention_target.value,
                "gpu_experts": len(decision.gpu_experts),
                "pim_experts": len(decision.pim_experts),
                "gpu_expert_tokens": gpu_tokens,
                "pim_expert_tokens": pim_tokens,
                "estimated_policy_objective_us": _round(
                    decision.estimated_objective_us
                ),
            }
        )
    step_rows = []
    for step in sorted(events_by_step):
        start, end, duration = _bounds(events_by_step[step])
        step_rows.append(
            {
                "step": step,
                "start_us": _round(start),
                "end_us": _round(end),
                "latency_us": _round(duration),
            }
        )
    step_latencies = [float(row["latency_us"]) for row in step_rows]
    total_latency = max((event.end_us for event in events), default=0.0)
    generated_tokens = traces[0].batch_size * len(step_rows)
    capacity_bytes = int(configuration.hardware.hbm_pim_capacity_gb * 1e9)
    peak_memory = max(memories, key=lambda memory: memory.total_bytes)
    summary: dict[str, Any] = {
        "experiment": configuration.experiment.name,
        "policy": decisions[0].policy,
        "timing_backend": configuration.hardware.timing_backend,
        "result_classification": _result_classification(
            configuration.hardware.timing_backend
        ),
        "scope": {
            "steps": list(configuration.experiment.steps),
            "layers": list(configuration.experiment.layers),
            "layer_executions": len(traces),
            "batch_size": traces[0].batch_size,
            "generated_request_tokens": generated_tokens,
        },
        "total_latency_us": _round(total_latency),
        "throughput_request_tokens_per_s": _round(
            generated_tokens * 1e6 / total_latency if total_latency else 0.0
        ),
        "decode_step_latency_us": {
            "mean": _round(sum(step_latencies) / len(step_latencies)),
            "p50": _round(_percentile(step_latencies, 0.50)),
            "p95": _round(_percentile(step_latencies, 0.95)),
            "min": _round(min(step_latencies)),
            "max": _round(max(step_latencies)),
        },
        "event_duration_us_by_category": {
            key: _round(value) for key, value in sorted(duration_by_category.items())
        },
        "critical_path": list(critical_path),
        "critical_path_duration_us_by_category": {
            key: _round(value)
            for key, value in sorted(critical_duration_by_category.items())
        },
        "placement_totals": {
            "gpu_expert_executions": total_gpu_experts,
            "pim_expert_executions": total_pim_experts,
            "gpu_expert_tokens": total_gpu_tokens,
            "pim_expert_tokens": total_pim_tokens,
        },
        "memory": {
            "model_weights_bytes": peak_memory.model_weights_bytes,
            "peak_kv_cache_bytes": peak_memory.kv_cache_bytes,
            "peak_activation_bytes": peak_memory.activation_bytes,
            "peak_total_bytes": peak_memory.total_bytes,
            "capacity_bytes": capacity_bytes,
            "feasible": peak_memory.total_bytes <= capacity_bytes,
        },
        "steps": step_rows,
        "layers": layer_rows,
        "input_hashes": _input_hashes(configuration),
        "kv_read": _kv_read_summary(
            configuration, traces, critical_duration_by_category
        ),
        "cxl": _cxl_summary(configuration, traces, critical_duration_by_category),
    }
    contention_rows = [
        {"step": trace.step, "layer": trace.layer, **contention}
        for trace, contention in zip(traces, contentions)
        if contention is not None
    ]
    if contention_rows:
        rounded_contentions = _round_nested(contention_rows)
        summary["contention_by_layer"] = rounded_contentions
        summary["contention_summary"] = _contention_summary(rounded_contentions)
    search_rows = [
        {
            "step": trace.step,
            "layer": trace.layer,
            **decision.search_report,
        }
        for trace, decision in zip(traces, decisions)
        if decision.search_report is not None
    ]
    if search_rows:
        summary["placement_search_by_layer"] = _round_nested(search_rows)
    return summary


def write_results(
    output_dir: str | Path,
    configuration: LoadedConfiguration,
    trace: TraceBatch,
    decision: PlacementDecision,
    events: tuple[ScheduledEvent, ...],
    memory: MemoryFootprint,
    contention: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = build_summary(
        configuration, trace, decision, events, memory, contention=contention
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with (output / "events.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "name",
                "category",
                "resources",
                "dependencies",
                "start_us",
                "duration_us",
                "end_us",
                "blocking_event",
                "description",
            ]
        )
        for scheduled in events:
            event = scheduled.event
            writer.writerow(
                [
                    event.name,
                    event.category,
                    ";".join(event.resources),
                    ";".join(event.dependencies),
                    f"{scheduled.start_us:.9f}",
                    f"{event.duration_us:.9f}",
                    f"{scheduled.end_us:.9f}",
                    scheduled.blocking_event or "",
                    event.description,
                ]
            )

    load_by_id = {load.expert_id: load.token_count for load in trace.expert_loads}
    with (output / "placement.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["expert_id", "token_count", "target"])
        for expert_id in sorted(load_by_id):
            writer.writerow([expert_id, load_by_id[expert_id], decision.target_for(expert_id).value])

    if decision.search_report is not None:
        search_candidates = decision.search_report.get("candidates", [])
        selected_prefix = decision.search_report.get("selected_gpu_prefix_length")
        search_fields = [
            "gpu_prefix_length",
            "gpu_experts",
            "gpu_tokens",
            "pim_tokens",
            "gpu_memory_us",
            "gpu_compute_us",
            "gpu_path_us",
            "pim_path_us",
            "scheduler_us",
            "objective_us",
            "isolation_available",
            "selected",
        ]
        with (output / "placement_search.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=search_fields, lineterminator="\n")
            writer.writeheader()
            for candidate in search_candidates:
                row = _round_nested(dict(candidate))
                row["gpu_experts"] = ";".join(
                    str(expert) for expert in candidate["gpu_experts"]
                )
                row["selected"] = candidate["gpu_prefix_length"] == selected_prefix
                writer.writerow({field: row[field] for field in search_fields})

    manifest = {
        "experiment": configuration.experiment.raw,
        "model": configuration.model_raw,
        "hardware": configuration.hardware_raw,
        "trace_selection": {"layer": trace.layer, "step": trace.step},
        "policy": decision.policy,
        "memory": asdict(memory),
        "input_hashes": summary["input_hashes"],
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def write_decode_results(
    output_dir: str | Path,
    configuration: LoadedConfiguration,
    traces: tuple[TraceBatch, ...],
    decisions: tuple[PlacementDecision, ...],
    events: tuple[ScheduledEvent, ...],
    memories: tuple[MemoryFootprint, ...],
    contentions: tuple[dict[str, Any] | None, ...],
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = build_decode_summary(
        configuration, traces, decisions, events, memories, contentions
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with (output / "events.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "step",
                "layer",
                "name",
                "category",
                "resources",
                "dependencies",
                "start_us",
                "duration_us",
                "end_us",
                "blocking_event",
                "description",
            ]
        )
        for scheduled in events:
            event = scheduled.event
            writer.writerow(
                [
                    event.step,
                    event.layer,
                    event.name,
                    event.category,
                    ";".join(event.resources),
                    ";".join(event.dependencies),
                    f"{scheduled.start_us:.9f}",
                    f"{event.duration_us:.9f}",
                    f"{scheduled.end_us:.9f}",
                    scheduled.blocking_event or "",
                    event.description,
                ]
            )

    _write_dict_rows(output / "layers.csv", summary["layers"])
    _write_dict_rows(output / "steps.csv", summary["steps"])
    if "contention_by_layer" in summary:
        _write_dict_rows(output / "contention.csv", summary["contention_by_layer"])

    with (output / "placement.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["step", "layer", "expert_id", "token_count", "target"])
        for trace, decision in zip(traces, decisions):
            load_by_id = {load.expert_id: load.token_count for load in trace.expert_loads}
            for expert_id in sorted(load_by_id):
                writer.writerow(
                    [
                        trace.step,
                        trace.layer,
                        expert_id,
                        load_by_id[expert_id],
                        decision.target_for(expert_id).value,
                    ]
                )

    search_rows: list[dict[str, Any]] = []
    for trace, decision in zip(traces, decisions):
        if decision.search_report is None:
            continue
        selected = decision.search_report.get("selected_gpu_prefix_length")
        for candidate in decision.search_report.get("candidates", []):
            row = _round_nested(dict(candidate))
            row["step"] = trace.step
            row["layer"] = trace.layer
            row["gpu_experts"] = ";".join(
                str(expert) for expert in candidate["gpu_experts"]
            )
            row["selected"] = candidate["gpu_prefix_length"] == selected
            search_rows.append(row)
    if search_rows:
        _write_dict_rows(output / "placement_search.csv", search_rows)

    manifest = {
        "experiment": configuration.experiment.raw,
        "model": configuration.model_raw,
        "hardware": configuration.hardware_raw,
        "trace_selection": {
            "layers": list(configuration.experiment.layers),
            "steps": list(configuration.experiment.steps),
        },
        "policy": decisions[0].policy,
        "input_hashes": summary["input_hashes"],
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _write_dict_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
