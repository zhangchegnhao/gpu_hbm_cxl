#!/usr/bin/env python3
"""Connect request-level Local-HBM/CXL KV timing to the full Decode graph.

Stage 5 uses the manifest-validated B8/C4k router batch as a controlled routing
template.  The template is tiled to four capacity pressure points and replayed
for ``gpu-only`` and analytic ``sieve`` placement.  Each unique request shape is
run exactly once through the isolated CXL Ramulator build, then its completion
milestones are used as event durations for a 48-layer, one-step Decode graph.

The CXL profile and route tiling are assumptions.  They are recorded as such;
this command does not claim A800 timing, a physical CXL device, or CXL-PIM
computation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sieve_replay.config import load_configuration
from sieve_replay.model import classify_capacity, estimate_memory_footprint
from sieve_replay.policy import create_policy
from sieve_replay.ramulator.cxl_kv_workload import (
    CXLKVWorkloadResult,
    cxl_kv_cache_context,
    run_cxl_kv_workload,
)
from sieve_replay.ramulator.mixed_workload import SieveCycleV1Config
from sieve_replay.ramulator.workload_cache import shape_for_loads
from sieve_replay.report.writer import sha256_file
from sieve_replay.simulation import EventEngine
from sieve_replay.simulation.decode_graph import build_decode_layer_graph
from sieve_replay.timing import AnalyticTimingModel, CXLReadConfig, CoupledExpertTiming
from sieve_replay.trace import RouterTraceRecord, TraceBatch, load_trace_set
from sieve_replay.trace_manifest import TraceManifest
from sieve_replay.types import ExpertLoad, PlacementDecision, TimingEstimate


CYCLE = ROOT / "configs/ramulator/sieve_hbm3e_cycle_v1.json"
SOURCE_EXPERIMENT = ROOT / "configs/experiments/full_decode_real_kv_b8_c4k.json"
CAPACITY = ROOT / "results/kv_cxl_capacity_stage2_v1/cxl_capacity_states.json"
RAMULATOR = ROOT / "third_party/ramulator2_cxl"
DEFAULT_OUTPUT = ROOT / "results/kv_cxl_decode_stage5_v1"

PRESSURE_POINTS: tuple[tuple[str, int, int], ...] = (
    ("b8_c32k", 8, 32768),
    ("b16_c16k", 16, 16384),
    ("b16_c32k", 16, 32768),
    ("b32_c16k", 32, 16384),
)
POLICIES = ("gpu-only", "sieve")
PROFILE_NAME = "nominal-assumption"
PROFILE: dict[str, float | int | str] = {
    "bandwidth_gb_s": 64.0,
    "latency_us": 0.25,
    "channel_count": 4,
    "capacity_gb": 64,
    "transaction_bytes": 32,
    "queue_depth": 256,
    "address_mapping": "static-spill-offset; pass-through channel mapper",
}
REQUEST_SCALE = 4096
DECODE_LAYERS = 48


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _key(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _source_configuration() -> tuple[Any, TraceBatch]:
    configuration = load_configuration(SOURCE_EXPERIMENT)
    traces = load_trace_set(
        configuration.experiment.trace_path,
        configuration.model,
        (0,),
        (0,),
    )
    if configuration.experiment.trace_manifest_path is None:
        raise ValueError("Stage 5 requires the source trace manifest")
    TraceManifest.load_and_validate(
        configuration.experiment.trace_manifest_path,
        configuration.experiment.trace_path,
        configuration.model,
        traces,
    )
    return configuration, traces.batches[0]


def _template(source: TraceBatch, batch_size: int, context_length: int) -> TraceBatch:
    if batch_size <= 0 or batch_size % source.batch_size:
        raise ValueError("Stage 5 batch sizes must be positive multiples of the B8 template")
    records = tuple(
        RouterTraceRecord(
            step=0,
            layer=0,
            token_id=f"stage5-b{batch_size}-c{context_length}-{index}",
            context_length=context_length,
            expert_ids=source.records[index % source.batch_size].expert_ids,
            expert_weights=source.records[index % source.batch_size].expert_weights,
        )
        for index in range(batch_size)
    )
    return TraceBatch(step=0, layer=0, records=records)


def _capacity_rows() -> dict[tuple[str, int, int, int], dict[str, Any]]:
    raw = json.loads(CAPACITY.read_text(encoding="utf-8"))
    rows: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for row in raw["rows"]:
        rows[
            (
                str(row["domain"]),
                int(row["batch_size"]),
                int(row["context_length"]),
                int(row["cxl_capacity_bytes"]),
            )
        ] = row
    return rows


def _request_counts(
    trace: TraceBatch,
    capacity: dict[str, Any],
    cycle: SieveCycleV1Config,
    model: Any,
    request_scale: int,
) -> tuple[int, int, int]:
    if request_scale <= 0:
        raise ValueError("request_scale must be positive")
    kv_bytes = (
        2
        * sum(trace.context_lengths)
        * model.num_key_value_heads
        * model.head_dim
        * model.dtype_bytes
    )
    total_transactions = math.ceil(kv_bytes / cycle.transaction_bytes)
    scaled_total = math.ceil(total_transactions / request_scale)
    spill_fraction = (
        float(capacity["spill_bytes"]) / float(capacity["kv_cache_bytes"])
        if capacity["kv_cache_bytes"]
        else 0.0
    )
    cxl_transactions = math.ceil(scaled_total * spill_fraction)
    return scaled_total - cxl_transactions, cxl_transactions, total_transactions


def _scaled_shape(
    decision: PlacementDecision,
    trace: TraceBatch,
    model: Any,
    cycle: SieveCycleV1Config,
    request_scale: int,
) -> dict[str, int]:
    by_id = {load.expert_id: load for load in trace.expert_loads}
    gpu_loads = tuple(by_id[expert] for expert in decision.gpu_experts)
    pim_loads = tuple(by_id[expert] for expert in decision.pim_experts)
    shape = asdict(shape_for_loads(gpu_loads, pim_loads, model, cycle))
    return {key: math.ceil(int(value) / request_scale) for key, value in shape.items()}


def _execution_provenance() -> dict[str, str]:
    revision = subprocess.check_output(
        ["git", "-C", str(RAMULATOR), "rev-parse", "HEAD"], text=True
    ).strip()
    bindings = sorted((RAMULATOR / "python/ramulator").glob("_ramulator.cpython-*.so"))
    if len(bindings) != 1:
        raise ValueError("expected exactly one CXL Ramulator Python binding")
    library = RAMULATOR / "libramulator.so"
    if not library.is_file():
        raise ValueError("CXL Ramulator shared library is missing")
    return {
        "ramulator_commit": revision,
        "binding_sha256": _sha256(bindings[0]),
        "ramulator_library_sha256": _sha256(library),
    }


class RequestLevelTiming(AnalyticTimingModel):
    """Timing adapter that injects one exact CXL workload's milestones."""

    def __init__(
        self,
        model: Any,
        hardware: Any,
        cxl_config: CXLReadConfig,
        result: CXLKVWorkloadResult,
        shape: dict[str, int],
        transaction_bytes: int,
    ) -> None:
        super().__init__(model, hardware, cxl_config=cxl_config)
        self.result = result
        self.shape = shape
        self.transaction_bytes = transaction_bytes

    def coupled_expert_timing(
        self,
        gpu_loads: tuple[ExpertLoad, ...],
        pim_loads: tuple[ExpertLoad, ...],
    ) -> CoupledExpertTiming:
        gpu_bytes = self.shape["gpu_read_transactions"] * self.transaction_bytes
        pim_bytes = (
            self.shape["pim_gwrite_waves"]
            + self.shape["pim_mac_waves"]
            + self.shape["pim_read_waves"]
        ) * self.transaction_bytes
        return CoupledExpertTiming(
            gpu_weight_load=TimingEstimate(
                self.result.gpu_completion_cycles * self.result.tick_ps / 1_000_000.0,
                bytes_accessed=float(gpu_bytes),
                description="GPU Expert READ milestone from exact combined CXL workload",
            ),
            # The CXL frontend exposes the PIM stream's final completion
            # milestone.  Keep the intermediate events zero-duration so the
            # graph carries one exact PIM critical-path milestone without
            # inventing stage-level interpolation.
            pim_token_write=TimingEstimate(0.0, description="PIM request stream included in exact workload"),
            pim_expert_compute=TimingEstimate(0.0, description="PIM request stream included in exact workload"),
            pim_result_read=TimingEstimate(
                self.result.pim_completion_cycles * self.result.tick_ps / 1_000_000.0,
                bytes_accessed=float(pim_bytes),
                description="PIM final completion milestone from exact combined CXL workload",
            ),
            report={
                "model": "cxl-kv-memory-only-request-v1",
                "request_scale": self.shape.get("request_scale", 0),
                "gpu_completion_cycles": self.result.gpu_completion_cycles,
                "pim_completion_cycles": self.result.pim_completion_cycles,
            },
        )

    def local_attention_kv_read(self, trace: TraceBatch) -> TimingEstimate:
        total = self.attention_kv_read(trace.context_lengths)
        local_bytes = self.shape["local_kv_read_transactions"] * self.transaction_bytes
        duration = self.result.local_kv_completion_cycles * self.result.tick_ps / 1_000_000.0
        return TimingEstimate(duration, bytes_accessed=float(local_bytes), description="Local KV READ milestone from exact combined CXL workload")

    def cxl_kv_read(self, trace: TraceBatch) -> TimingEstimate:
        cxl_bytes = self.shape["cxl_kv_read_transactions"] * self.transaction_bytes
        duration = self.result.cxl_kv_completion_cycles * self.result.tick_ps / 1_000_000.0
        return TimingEstimate(duration, bytes_accessed=float(cxl_bytes), description="CXL KV READ milestone from exact combined CXL workload")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _duration_categories(events: tuple[Any, ...], critical: set[str] | None = None) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in events:
        if critical is not None and item.event.name not in critical:
            continue
        values[item.event.category] = values.get(item.event.category, 0.0) + item.event.duration_us
    return values


def _run_one(
    output: Path,
    configuration: Any,
    source: TraceBatch,
    case: str,
    batch_size: int,
    context_length: int,
    policy_name: str,
    capacity: dict[str, Any],
    simulated_capacity: dict[str, Any],
    cycle: SieveCycleV1Config,
    request_scale: int,
    workload_runner: Callable[..., CXLKVWorkloadResult] = run_cxl_kv_workload,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trace = _template(source, batch_size, context_length)
    decision_timing = AnalyticTimingModel(configuration.model, configuration.hardware)
    decision = create_policy(policy_name, decision_timing).place(trace)
    shape = _scaled_shape(decision, trace, configuration.model, cycle, request_scale)
    local_kv, cxl_kv, full_kv_transactions = _request_counts(
        trace, capacity, cycle, configuration.model, request_scale
    )
    shape.update(
        local_kv_read_transactions=local_kv,
        cxl_kv_read_transactions=cxl_kv,
        request_scale=request_scale,
    )
    shape_for_runner = {key: value for key, value in shape.items() if key != "request_scale"}
    cache_input = {
        "schema_version": 1,
        "case": case,
        "policy": policy_name,
        "batch_size": batch_size,
        "context_length": context_length,
        "request_scale": request_scale,
        "profile": PROFILE_NAME,
        "profile_parameters": PROFILE,
        "placement": {
            "gpu_experts": list(decision.gpu_experts),
            "pim_experts": list(decision.pim_experts),
        },
        "shape": shape_for_runner,
        "capacity": {
            "domain": "a800",
            "local_capacity_bytes": int(capacity["local_capacity_bytes"]),
            "cxl_capacity_bytes": int(capacity["cxl_capacity_bytes"]),
            "spill_bytes": int(capacity["spill_bytes"]),
        },
        "cycle_config_sha256": _sha256(CYCLE),
        "source_trace_sha256": _sha256(configuration.experiment.trace_path),
    }
    cache_key = _key(cache_input)
    cache_dir = output / "exact_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key}.json"
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("cache_input") != cache_input:
            raise ValueError(f"Stage-5 cache provenance mismatch: {cache_path}")
        result = CXLKVWorkloadResult(**cached["result"])
    else:
        result = workload_runner(
            RAMULATOR,
            cycle,
            **shape_for_runner,
            cxl_bandwidth_bytes_per_second=float(PROFILE["bandwidth_gb_s"]) * 1e9,
            cxl_latency_us=float(PROFILE["latency_us"]),
            cxl_channels=int(PROFILE["channel_count"]),
        )
        cache_path.write_text(
            json.dumps({"cache_input": cache_input, "result": asdict(result)}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    expected = {
        "gpu_read_transactions": shape_for_runner["gpu_read_transactions"],
        "local_kv_read_transactions": local_kv,
        "cxl_kv_read_transactions": cxl_kv,
        "pim_completed_requests": (
            shape_for_runner["pim_gwrite_waves"]
            + shape_for_runner["pim_mac_waves"]
            + shape_for_runner["pim_read_waves"]
        ) * cycle.total_pseudo_channels,
    }
    if result.gpu_completed_requests != expected["gpu_read_transactions"]:
        raise ValueError(f"GPU request completion mismatch for {case}/{policy_name}")
    if result.local_kv_completed_requests != expected["local_kv_read_transactions"]:
        raise ValueError(f"Local KV request completion mismatch for {case}/{policy_name}")
    if result.cxl_kv_completed_requests != expected["cxl_kv_read_transactions"]:
        raise ValueError(f"CXL KV request completion mismatch for {case}/{policy_name}")
    if result.pim_completed_requests != expected["pim_completed_requests"]:
        raise ValueError(f"PIM request completion mismatch for {case}/{policy_name}")

    cxl_config = CXLReadConfig(
        mode="memory-only-v1",
        local_capacity_bytes=int(capacity["local_capacity_bytes"]),
        capacity_bytes=int(capacity["cxl_capacity_bytes"]),
        bandwidth_bytes_per_second=float(PROFILE["bandwidth_gb_s"]) * 1e9,
        latency_us=float(PROFILE["latency_us"]),
    )
    timing = RequestLevelTiming(
        configuration.model,
        configuration.hardware,
        cxl_config,
        result,
        shape,
        cycle.transaction_bytes,
    )
    events: list[Any] = []
    traces: list[TraceBatch] = []
    previous_tail: str | None = None
    for layer in range(DECODE_LAYERS):
        layer_trace = TraceBatch(
            step=0,
            layer=layer,
            records=tuple(
                RouterTraceRecord(
                    step=0,
                    layer=layer,
                    token_id=record.token_id,
                    context_length=record.context_length,
                    expert_ids=record.expert_ids,
                    expert_weights=record.expert_weights,
                )
                for record in trace.records
            ),
        )
        layer_events, previous_tail = build_decode_layer_graph(
            layer_trace,
            decision,
            timing,
            previous_tail,
            "overlap-local-hbm-v1",
        )
        events.extend(layer_events)
        traces.append(layer_trace)
    scheduled = EventEngine().run(tuple(events))
    critical_path = EventEngine.critical_path(scheduled)
    critical_set = set(critical_path)
    durations = _duration_categories(scheduled)
    critical_durations = _duration_categories(scheduled, critical_set)
    total_latency = max((item.end_us for item in scheduled), default=0.0)
    attention_critical = sum(
        critical_durations.get(category, 0.0)
        for category in ("attention", "kv_read", "cxl_kv_read")
    )
    expert_categories = ("gpu_weight_load", "gpu_expert", "pim_write", "pim_expert", "pim_read")
    expert_critical = sum(critical_durations.get(category, 0.0) for category in expert_categories)
    memory = estimate_memory_footprint(configuration.model, trace)
    admission = classify_capacity(
        memory.total_bytes,
        int(capacity["local_capacity_bytes"]),
        int(capacity["cxl_capacity_bytes"]),
    )
    full_attention_bytes = (
        2
        * sum(trace.context_lengths)
        * configuration.model.num_key_value_heads
        * configuration.model.head_dim
        * configuration.model.dtype_bytes
    )
    layer_rows: list[dict[str, Any]] = []
    for layer in range(DECODE_LAYERS):
        layer_items = [item for item in scheduled if item.event.layer == layer]
        layer_critical = {name for name in critical_set if name.startswith(f"step0.layer{layer}.")}
        layer_critical_durations = _duration_categories(tuple(layer_items), layer_critical)
        layer_start = min((item.start_us for item in layer_items), default=0.0)
        layer_end = max((item.end_us for item in layer_items), default=0.0)
        layer_attention_critical = sum(
            layer_critical_durations.get(category, 0.0)
            for category in ("attention", "kv_read", "cxl_kv_read")
        )
        layer_rows.append(
            {
                "case": case,
                "policy": policy_name,
                "layer": layer,
                "batch_size": batch_size,
                "context_length": context_length,
                "request_scale": request_scale,
                "capacity_state": admission["state"],
                "layer_start_us": layer_start,
                "layer_end_us": layer_end,
                "layer_span_us": layer_end - layer_start,
                "attention_latency_us": sum(item.event.duration_us for item in layer_items if item.event.category == "attention"),
                "local_kv_latency_us": sum(item.event.duration_us for item in layer_items if item.event.category == "kv_read"),
                "cxl_kv_latency_us": sum(item.event.duration_us for item in layer_items if item.event.category == "cxl_kv_read"),
                "attention_critical_path_us": layer_attention_critical,
                "expert_critical_path_us": sum(layer_critical_durations.get(category, 0.0) for category in expert_categories),
                "cxl_critical_path_us": layer_critical_durations.get("cxl_kv_read", 0.0),
                "gpu_request_completion_us": result.gpu_completion_cycles * result.tick_ps / 1_000_000.0,
                "local_kv_request_completion_us": result.local_kv_completion_cycles * result.tick_ps / 1_000_000.0,
                "cxl_kv_request_completion_us": result.cxl_kv_completion_cycles * result.tick_ps / 1_000_000.0,
                "pim_request_completion_us": result.pim_completion_cycles * result.tick_ps / 1_000_000.0,
                "cxl_queue_wait_us_all_requests": result.cxl_queue_wait_us,
                "cxl_link_busy_ratio": result.cxl_link_busy_ratio,
            }
        )

    summary = {
        "case": case,
        "policy": policy_name,
        "batch_size": batch_size,
        "context_length": context_length,
        "layers": DECODE_LAYERS,
        "decode_steps": 1,
        "request_scale": request_scale,
        "classification": "48-layer Decode graph with exact request-level CXL memory-only timing; route template and CXL profile are assumptions",
        "capacity_domain": "a800-80gb-plus-hypothetical-cxl-64gb",
        "capacity_state": admission["state"],
        "capacity_feasible": bool(admission["feasible"]),
        "simulated_local_state_without_cxl": simulated_capacity["state"],
        "simulated_local_feasible_without_cxl": bool(simulated_capacity["feasible"]),
        "model_weights_bytes": memory.model_weights_bytes,
        "kv_cache_bytes": memory.kv_cache_bytes,
        "peak_memory_bytes": memory.total_bytes,
        "local_kv_read_bytes": local_kv * DECODE_LAYERS * cycle.transaction_bytes,
        "cxl_kv_read_bytes": cxl_kv * DECODE_LAYERS * cycle.transaction_bytes,
        "full_layer_kv_read_bytes": full_attention_bytes,
        "full_layer_kv_transactions": full_kv_transactions,
        "gpu_read_transactions_per_layer": shape_for_runner["gpu_read_transactions"],
        "local_kv_read_transactions_per_layer": local_kv,
        "cxl_kv_read_transactions_per_layer": cxl_kv,
        "pim_gwrite_waves_per_layer": shape_for_runner["pim_gwrite_waves"],
        "pim_mac_waves_per_layer": shape_for_runner["pim_mac_waves"],
        "pim_read_waves_per_layer": shape_for_runner["pim_read_waves"],
        "total_latency_us": total_latency,
        "throughput_request_tokens_per_s": batch_size * 1_000_000.0 / total_latency if total_latency else 0.0,
        "attention_latency_us": durations.get("attention", 0.0),
        "attention_kv_local_latency_us": durations.get("kv_read", 0.0),
        "attention_kv_cxl_latency_us": durations.get("cxl_kv_read", 0.0),
        "attention_critical_path_us": attention_critical,
        "attention_latency_share": attention_critical / total_latency if total_latency else 0.0,
        "expert_latency_us": sum(durations.get(category, 0.0) for category in expert_categories),
        "expert_critical_path_us": expert_critical,
        "cxl_critical_path_latency_us": critical_durations.get("cxl_kv_read", 0.0),
        "cxl_queue_wait_us_all_requests": result.cxl_queue_wait_us * DECODE_LAYERS,
        "cxl_max_queue_wait_us": result.cxl_max_queue_wait_cycles * result.tick_ps / 1_000_000.0,
        "cxl_link_busy_ratio": result.cxl_link_busy_ratio,
        "critical_path": ";".join(critical_path),
        "cache_key": cache_key,
        "cache_sha256": _sha256(cache_path),
        "exact_runs": 1,
        "interpolation": 0,
        "placement": {
            "attention_target": decision.attention_target.value,
            "gpu_experts": list(decision.gpu_experts),
            "pim_experts": list(decision.pim_experts),
            "estimated_objective_us": decision.estimated_objective_us,
        },
    }
    return summary, layer_rows


def _write_report(output: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    by_case = {(row["case"], row["policy"]): row for row in rows}
    lines = [
        "# KV CXL Decode Stage 5",
        "",
        "本阶段把 CXL memory-only KV READ 的请求级完成 milestone 接入 48 层、1 step 的 Decode event graph。",
        "路由来自 manifest 校验的真实 B8/C4k trace，并按 Batch/Context 做受控 tiling；CXL 带宽、延迟和容量是明确假设。",
        "结果是 Ramulator request-level + 事件图模拟，不是 A800 实机计时，也没有实现 CXL-PIM 计算。",
        "请求级 CXL queue wait 是所有请求等待时间之和，单独列出，不作为端到端关键路径。",
        "",
        "| case | policy | capacity | total us | throughput tok/s | attention share | expert critical us | CXL critical us | CXL queue wait (all requests) us |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in metadata["cases"]:
        for policy in metadata["policies"]:
            row = by_case[(case, policy)]
            lines.append(
                f"| {case} | {policy} | {row['capacity_state']} | {row['total_latency_us']:.6f} | "
                f"{row['throughput_request_tokens_per_s']:.3f} | {row['attention_latency_share']:.6f} | "
                f"{row['expert_critical_path_us']:.6f} | {row['cxl_critical_path_latency_us']:.6f} | "
                f"{row['cxl_queue_wait_us_all_requests']:.6f} |"
            )
    lines += [
        "",
        "## Scope and limitations",
        "",
        "- The two policies use the same controlled B8/C4k route template tiled to the selected Batch; no new A800 Router capture is created.",
        "- The CXL controller is the Stage-4 FIFO memory-only model with the nominal 64 GB/s, 0.25 us, four-channel profile.",
        "- `request_scale=4096` reduces Expert/PIM/KV request counts for a tractable exact shape run; the graph still contains analytic GPU compute and fixed model stages.",
        "- PIM timing uses the final PIM completion milestone exposed by the frontend; intermediate PIM events are zero-duration bookkeeping, so no stage-level interpolation is claimed.",
        "- A800 80 GB plus CXL 64 GB is kept separate from the 96 GB simulated Local-HBM capacity. The latter is reported only as a comparison state without CXL.",
        "- No paging, eviction, recovery, physical CXL protocol, multi-GPU, or CXL-PIM computation is modeled.",
        "- A combined `request_scale=1` B8/C32k gate was attempted separately but stopped before its first cache entry because the current per-cycle frontend makes the roughly 19 million-request shape too slow; it is not part of this matrix.",
        "",
        f"Exact request runs: `{metadata['exact_runs']}`; interpolation: `{metadata['interpolation']}`.",
        "",
    ]
    (output / "stage5_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_stage5(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    cases: tuple[str, ...] = tuple(case for case, _, _ in PRESSURE_POINTS),
    policies: tuple[str, ...] = POLICIES,
    request_scale: int = REQUEST_SCALE,
    workload_runner: Callable[..., CXLKVWorkloadResult] = run_cxl_kv_workload,
) -> dict[str, Any]:
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Stage-5 output already exists and is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    configuration, source = _source_configuration()
    cycle = SieveCycleV1Config.load(CYCLE)
    capacity_rows = _capacity_rows()
    point_map = {case: (batch, context) for case, batch, context in PRESSURE_POINTS}
    if not cases or any(case not in point_map for case in cases):
        raise ValueError("unknown Stage-5 pressure point")
    if not policies or any(policy not in POLICIES for policy in policies):
        raise ValueError("unknown Stage-5 policy")
    rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    for case in cases:
        batch, context = point_map[case]
        capacity = capacity_rows[
            ("a800", batch, context, int(PROFILE["capacity_gb"]) * 1_000_000_000)
        ]
        simulated_capacity = capacity_rows[("simulated", batch, context, 0)]
        if not capacity["feasible"]:
            raise ValueError(f"Stage-5 CXL capacity is insufficient for {case}")
        for policy in policies:
            summary, details = _run_one(
                output,
                configuration,
                source,
                case,
                batch,
                context,
                policy,
                capacity,
                simulated_capacity,
                cycle,
                request_scale,
                workload_runner,
            )
            rows.append(summary)
            layer_rows.extend(details)
    _write_csv(output / "comparison.csv", rows)
    _write_csv(output / "layer_results.csv", layer_rows)
    metadata = {
        "schema_version": 1,
        "stage": "kv-cxl-decode-stage5-v1",
        "method": "request-level-cxl-kv-milestones-in-48-layer-decode-graph",
        "classification": "Ramulator request-level CXL memory-only timing connected to an analytic Decode event graph; not A800 timing",
        "cases": list(cases),
        "policies": list(policies),
        "layers": DECODE_LAYERS,
        "decode_steps": 1,
        "request_scale": request_scale,
        "profile": {"name": PROFILE_NAME, **PROFILE},
        "capacity_domain": "A800 80 GB + hypothetical CXL 64 GB; simulated 96 GB reported separately",
        "source_routing": "manifest-validated real B8/C4k step0/layer0 tiled by integer batch factor",
        "execution": _execution_provenance(),
        "ramulator_context": cxl_kv_cache_context(ROOT, CYCLE),
        "exact_runs": len(rows),
        "interpolation": 0,
        "rows": len(rows),
        "layer_rows": len(layer_rows),
        "input_hashes": {
            "source_experiment": _sha256(SOURCE_EXPERIMENT),
            "source_trace": _sha256(configuration.experiment.trace_path),
            "source_manifest": _sha256(configuration.experiment.trace_manifest_path),
            "model": _sha256(configuration.experiment.model_path),
            "hardware": _sha256(configuration.experiment.hardware_path),
            "cycle_config": _sha256(CYCLE),
            "capacity_source": _sha256(CAPACITY),
        },
        "limitations": [
            "CXL profile is a fixed memory-only FIFO assumption, not a physical CXL measurement.",
            "Routing is a controlled tiled template from the manifest-validated B8/C4k trace.",
            f"Request-level Expert/PIM/KV counts use request_scale={request_scale}; this is not a full-count Decode timing result.",
            "PIM intermediate event stages are zero-duration bookkeeping around the exact final PIM milestone.",
            "No paging, eviction, recovery, CXL-PIM computation, or multi-GPU behavior.",
        ],
    }
    comparison = {"metadata": metadata, "rows": rows}
    comparison_path = output / "comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(output, rows, metadata)
    manifest = {
        "schema_version": 1,
        "stage": metadata["stage"],
        "exact_runs": len(rows),
        "interpolation": 0,
        "rows": len(rows),
        "layer_rows": len(layer_rows),
        "cases": list(cases),
        "policies": list(policies),
        "request_scale": request_scale,
        "execution": metadata["execution"],
        "input_hashes": metadata["input_hashes"],
        "output_hashes": {
            "comparison.csv": _sha256(output / "comparison.csv"),
            "comparison.json": _sha256(comparison_path),
            "layer_results.csv": _sha256(output / "layer_results.csv"),
            "stage5_report.md": _sha256(output / "stage5_report.md"),
        },
    }
    (output / "stage_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"metadata": metadata, "rows": rows, "stage_manifest": manifest}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--request-scale", type=int, default=REQUEST_SCALE)
    parser.add_argument("--cases", default=",".join(case for case, _, _ in PRESSURE_POINTS))
    parser.add_argument("--policies", default=",".join(POLICIES))
    args = parser.parse_args(argv)
    cases = tuple(value for value in args.cases.split(",") if value)
    policies = tuple(value for value in args.policies.split(",") if value)
    result = run_stage5(
        args.output,
        request_scale=args.request_scale,
        cases=cases,
        policies=policies,
    )
    print(json.dumps(result["stage_manifest"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
