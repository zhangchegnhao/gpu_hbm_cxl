from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..config import LoadedConfiguration
from ..model import MemoryFootprint
from ..simulation import EventEngine, ScheduledEvent
from ..trace import TraceBatch
from ..types import PlacementDecision


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    input_hashes = {
        "model_sha256": sha256_file(configuration.experiment.model_path),
        "hardware_sha256": sha256_file(configuration.experiment.hardware_path),
        "trace_sha256": sha256_file(configuration.experiment.trace_path),
    }
    if configuration.experiment.pim_timing_table_path is not None:
        input_hashes["pim_timing_table_sha256"] = sha256_file(
            configuration.experiment.pim_timing_table_path
        )
    if configuration.experiment.contention_timing_table_path is not None:
        input_hashes["contention_timing_table_sha256"] = sha256_file(
            configuration.experiment.contention_timing_table_path
        )
    if configuration.experiment.ramulator_cycle_config_path is not None:
        input_hashes["ramulator_cycle_config_sha256"] = sha256_file(
            configuration.experiment.ramulator_cycle_config_path
        )
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
    summary = {
        "experiment": configuration.experiment.name,
        "policy": decision.policy,
        "timing_backend": configuration.hardware.timing_backend,
        "result_classification": classifications[configuration.hardware.timing_backend],
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
    }
    if contention is not None:
        summary["contention"] = {
            key: _round(value) if isinstance(value, float) else value
            for key, value in sorted(contention.items())
        }
    if decision.search_report is not None:
        summary["placement_search"] = _round_nested(decision.search_report)
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
