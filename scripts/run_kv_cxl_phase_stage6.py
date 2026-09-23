#!/usr/bin/env python3
"""Run phase-faithful request-level CXL timing in the full Decode graph.

Unlike the controlled Stage-5 combined workload, this stage follows the
existing Decode dependencies: Local/CXL KV requests execute in the Attention
    phase, while GPU/PIM Expert requests execute after routing.  Identical request
    shapes are deduplicated across policies.  The formal gate compares the tractable
    scale=4096 reference with an exact full-count scale=1 B8/C32k run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_kv_cxl_decode_stage5 as stage5
from sieve_replay.model import classify_capacity, estimate_memory_footprint
from sieve_replay.policy import create_policy
from sieve_replay.ramulator.cxl_kv_workload import CXLKVWorkloadResult, run_cxl_kv_workload
from sieve_replay.ramulator.mixed_workload import SieveCycleV1Config
from sieve_replay.simulation import EventEngine
from sieve_replay.simulation.decode_graph import build_decode_layer_graph
from sieve_replay.timing import AnalyticTimingModel, CXLReadConfig, CoupledExpertTiming
from sieve_replay.trace import RouterTraceRecord, TraceBatch
from sieve_replay.types import ExpertLoad, PlacementDecision, TimingEstimate


DEFAULT_OUTPUT = ROOT / "results/kv_cxl_phase_stage6_b8_c32k_v1"
CASE = "b8_c32k"
BATCH_SIZE = 8
CONTEXT_LENGTH = 32768
POLICIES = ("gpu-only", "sieve")
REQUEST_SCALES = (4096, 1)
LAYERS = 48


def _key(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _shape_identity(
    phase: str,
    shape: dict[str, int],
    request_scale: int,
    context: dict[str, str],
) -> dict[str, Any]:
    """Return only inputs that can change the exact Ramulator result."""
    return {
        "schema_version": 1,
        "phase": phase,
        "request_scale": request_scale,
        "profile": stage5.PROFILE_NAME,
        "profile_parameters": stage5.PROFILE,
        "simulation_context": context,
        "shape": shape,
    }


def _phase_shapes(
    trace: TraceBatch,
    decision: PlacementDecision,
    capacity: dict[str, Any],
    configuration: Any,
    cycle: SieveCycleV1Config,
    request_scale: int,
) -> dict[str, dict[str, int]]:
    expert = stage5._scaled_shape(
        decision, trace, configuration.model, cycle, request_scale
    )
    local_kv, cxl_kv, _ = stage5._request_counts(
        trace, capacity, cycle, configuration.model, request_scale
    )
    return {
        "attention": {
            "gpu_read_transactions": 0,
            "local_kv_read_transactions": local_kv,
            "cxl_kv_read_transactions": cxl_kv,
            "pim_gwrite_waves": 0,
            "pim_mac_waves": 0,
            "pim_read_waves": 0,
        },
        "expert": {
            **expert,
            "local_kv_read_transactions": 0,
            "cxl_kv_read_transactions": 0,
        },
    }


def _run_or_load(
    output: Path,
    phase: str,
    shape: dict[str, int],
    request_scale: int,
    cycle: SieveCycleV1Config,
    simulation_context: dict[str, str],
    workload_runner: Callable[..., CXLKVWorkloadResult],
) -> tuple[str, Path, CXLKVWorkloadResult, bool]:
    cache_input = _shape_identity(phase, shape, request_scale, simulation_context)
    cache_key = _key(cache_input)
    cache_dir = output / "exact_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key}.json"
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("cache_input") != cache_input:
            raise ValueError(f"Stage-6 cache provenance mismatch: {cache_path}")
        return cache_key, cache_path, CXLKVWorkloadResult(**cached["result"]), False
    result = workload_runner(
        stage5.RAMULATOR,
        cycle,
        **shape,
        cxl_bandwidth_bytes_per_second=float(stage5.PROFILE["bandwidth_gb_s"]) * 1e9,
        cxl_latency_us=float(stage5.PROFILE["latency_us"]),
        cxl_channels=int(stage5.PROFILE["channel_count"]),
    )
    cache_path.write_text(
        json.dumps(
            {"cache_input": cache_input, "result": asdict(result)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return cache_key, cache_path, result, True


def _validate_completion(
    phase: str,
    shape: dict[str, int],
    result: CXLKVWorkloadResult,
    cycle: SieveCycleV1Config,
) -> None:
    expected_pim = (
        shape["pim_gwrite_waves"]
        + shape["pim_mac_waves"]
        + shape["pim_read_waves"]
    ) * cycle.total_pseudo_channels
    expected = {
        "gpu_completed_requests": shape["gpu_read_transactions"],
        "local_kv_completed_requests": shape["local_kv_read_transactions"],
        "cxl_kv_completed_requests": shape["cxl_kv_read_transactions"],
        "pim_completed_requests": expected_pim,
    }
    for field, value in expected.items():
        if getattr(result, field) != value:
            raise ValueError(
                f"{phase} completion mismatch for {field}: "
                f"{getattr(result, field)} != {value}"
            )


class PhaseTiming(AnalyticTimingModel):
    """Inject separate exact Attention-memory and Expert-memory milestones."""

    def __init__(
        self,
        model: Any,
        hardware: Any,
        cxl_config: CXLReadConfig,
        attention: CXLKVWorkloadResult,
        expert: CXLKVWorkloadResult,
        attention_shape: dict[str, int],
        expert_shape: dict[str, int],
        transaction_bytes: int,
    ) -> None:
        super().__init__(model, hardware, cxl_config=cxl_config)
        self.attention_result = attention
        self.expert_result = expert
        self.attention_shape = attention_shape
        self.expert_shape = expert_shape
        self.transaction_bytes = transaction_bytes

    @staticmethod
    def _duration(result: CXLKVWorkloadResult, cycles: int) -> float:
        return cycles * result.tick_ps / 1_000_000.0

    def local_attention_kv_read(self, trace: TraceBatch) -> TimingEstimate:
        transactions = self.attention_shape["local_kv_read_transactions"]
        return TimingEstimate(
            self._duration(
                self.attention_result,
                self.attention_result.local_kv_completion_cycles,
            ),
            bytes_accessed=float(transactions * self.transaction_bytes),
            description="phase-faithful Local KV READ completion milestone",
        )

    def cxl_kv_read(self, trace: TraceBatch) -> TimingEstimate:
        transactions = self.attention_shape["cxl_kv_read_transactions"]
        return TimingEstimate(
            self._duration(
                self.attention_result,
                self.attention_result.cxl_kv_completion_cycles,
            ),
            bytes_accessed=float(transactions * self.transaction_bytes),
            description="phase-faithful CXL KV READ completion milestone",
        )

    def coupled_expert_timing(
        self,
        gpu_loads: tuple[ExpertLoad, ...],
        pim_loads: tuple[ExpertLoad, ...],
    ) -> CoupledExpertTiming:
        gpu_bytes = self.expert_shape["gpu_read_transactions"] * self.transaction_bytes
        pim_waves = sum(
            self.expert_shape[key]
            for key in ("pim_gwrite_waves", "pim_mac_waves", "pim_read_waves")
        )
        return CoupledExpertTiming(
            gpu_weight_load=TimingEstimate(
                self._duration(
                    self.expert_result,
                    self.expert_result.gpu_completion_cycles,
                ),
                bytes_accessed=float(gpu_bytes),
                description="phase-faithful GPU Expert READ completion milestone",
            ),
            pim_token_write=TimingEstimate(
                0.0, description="PIM request stream included in exact Expert phase"
            ),
            pim_expert_compute=TimingEstimate(
                0.0, description="PIM request stream included in exact Expert phase"
            ),
            pim_result_read=TimingEstimate(
                self._duration(
                    self.expert_result,
                    self.expert_result.pim_completion_cycles,
                ),
                bytes_accessed=float(pim_waves * self.transaction_bytes),
                description="phase-faithful final PIM Expert completion milestone",
            ),
            report={
                "model": "phase-faithful-request-level-v1",
                "gpu_completion_cycles": self.expert_result.gpu_completion_cycles,
                "pim_completion_cycles": self.expert_result.pim_completion_cycles,
            },
        )


def _layer_trace(template: TraceBatch, layer: int) -> TraceBatch:
    return TraceBatch(
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
            for record in template.records
        ),
    )


def _event_totals(events: tuple[Any, ...], names: set[str] | None = None) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in events:
        if names is not None and item.event.name not in names:
            continue
        category = item.event.category
        values[category] = values.get(category, 0.0) + item.event.duration_us
    return values


def _replay(
    configuration: Any,
    template: TraceBatch,
    policy_name: str,
    request_scale: int,
    capacity: dict[str, Any],
    simulated_capacity: dict[str, Any],
    cycle: SieveCycleV1Config,
    phase_shapes: dict[str, dict[str, int]],
    phase_results: dict[str, CXLKVWorkloadResult],
    phase_keys: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    decision_timing = AnalyticTimingModel(configuration.model, configuration.hardware)
    decision = create_policy(policy_name, decision_timing).place(template)
    cxl_config = CXLReadConfig(
        mode="memory-only-v1",
        local_capacity_bytes=int(capacity["local_capacity_bytes"]),
        capacity_bytes=int(capacity["cxl_capacity_bytes"]),
        bandwidth_bytes_per_second=float(stage5.PROFILE["bandwidth_gb_s"]) * 1e9,
        latency_us=float(stage5.PROFILE["latency_us"]),
    )
    timing = PhaseTiming(
        configuration.model,
        configuration.hardware,
        cxl_config,
        phase_results["attention"],
        phase_results["expert"],
        phase_shapes["attention"],
        phase_shapes["expert"],
        cycle.transaction_bytes,
    )
    graph = []
    previous_tail: str | None = None
    for layer in range(LAYERS):
        events, previous_tail = build_decode_layer_graph(
            _layer_trace(template, layer),
            decision,
            timing,
            previous_tail,
            "overlap-local-hbm-v1",
        )
        graph.extend(events)
    scheduled = EventEngine().run(tuple(graph))
    critical_path = EventEngine.critical_path(scheduled)
    critical_set = set(critical_path)
    durations = _event_totals(scheduled)
    critical = _event_totals(scheduled, critical_set)
    total = max(item.end_us for item in scheduled)
    attention_categories = ("attention", "kv_read", "cxl_kv_read")
    expert_categories = (
        "gpu_weight_load",
        "gpu_expert",
        "pim_write",
        "pim_expert",
        "pim_read",
    )
    attention_critical = sum(critical.get(key, 0.0) for key in attention_categories)
    expert_critical = sum(critical.get(key, 0.0) for key in expert_categories)
    memory = estimate_memory_footprint(configuration.model, template)
    admission = classify_capacity(
        memory.total_bytes,
        int(capacity["local_capacity_bytes"]),
        int(capacity["cxl_capacity_bytes"]),
    )
    attention_result = phase_results["attention"]
    expert_result = phase_results["expert"]
    summary = {
        "case": CASE,
        "policy": policy_name,
        "request_scale": request_scale,
        "layers": LAYERS,
        "decode_steps": 1,
        "classification": "phase-faithful 48-layer Decode graph with exact request-level memory milestones; CXL profile and routing tiling are assumptions",
        "capacity_state": admission["state"],
        "capacity_feasible": bool(admission["feasible"]),
        "simulated_local_state_without_cxl": simulated_capacity["state"],
        "kv_cache_bytes": memory.kv_cache_bytes,
        "peak_memory_bytes": memory.total_bytes,
        "total_latency_us": total,
        "throughput_request_tokens_per_s": BATCH_SIZE * 1_000_000.0 / total,
        "attention_latency_us": durations.get("attention", 0.0),
        "local_kv_event_latency_us": durations.get("kv_read", 0.0),
        "cxl_kv_event_latency_us": durations.get("cxl_kv_read", 0.0),
        "attention_critical_path_us": attention_critical,
        "attention_latency_share": attention_critical / total,
        "expert_event_latency_us": sum(durations.get(key, 0.0) for key in expert_categories),
        "expert_critical_path_us": expert_critical,
        "cxl_critical_path_us": critical.get("cxl_kv_read", 0.0),
        "local_kv_completion_us_per_layer": PhaseTiming._duration(
            attention_result, attention_result.local_kv_completion_cycles
        ),
        "cxl_kv_completion_us_per_layer": PhaseTiming._duration(
            attention_result, attention_result.cxl_kv_completion_cycles
        ),
        "gpu_expert_completion_us_per_layer": PhaseTiming._duration(
            expert_result, expert_result.gpu_completion_cycles
        ),
        "pim_expert_completion_us_per_layer": PhaseTiming._duration(
            expert_result, expert_result.pim_completion_cycles
        ),
        "cxl_queue_wait_us_all_requests": attention_result.cxl_queue_wait_us * LAYERS,
        "cxl_max_queue_wait_us": attention_result.cxl_max_queue_wait_cycles
        * attention_result.tick_ps
        / 1_000_000.0,
        "cxl_link_busy_ratio": attention_result.cxl_link_busy_ratio,
        "attention_cache_key": phase_keys["attention"],
        "expert_cache_key": phase_keys["expert"],
        "attention_shape": phase_shapes["attention"],
        "expert_shape": phase_shapes["expert"],
        "placement": {
            "attention_target": decision.attention_target.value,
            "gpu_experts": list(decision.gpu_experts),
            "pim_experts": list(decision.pim_experts),
        },
        "critical_path": ";".join(critical_path),
        "interpolation": 0,
    }
    layer_rows: list[dict[str, Any]] = []
    for layer in range(LAYERS):
        layer_items = tuple(item for item in scheduled if item.event.layer == layer)
        layer_critical_names = {
            name for name in critical_set if name.startswith(f"step0.layer{layer}.")
        }
        layer_critical = _event_totals(layer_items, layer_critical_names)
        start = min(item.start_us for item in layer_items)
        end = max(item.end_us for item in layer_items)
        layer_rows.append(
            {
                "case": CASE,
                "policy": policy_name,
                "request_scale": request_scale,
                "layer": layer,
                "layer_start_us": start,
                "layer_end_us": end,
                "layer_span_us": end - start,
                "attention_critical_path_us": sum(
                    layer_critical.get(key, 0.0) for key in attention_categories
                ),
                "expert_critical_path_us": sum(
                    layer_critical.get(key, 0.0) for key in expert_categories
                ),
                "cxl_critical_path_us": layer_critical.get("cxl_kv_read", 0.0),
                "cxl_queue_wait_us_all_requests": attention_result.cxl_queue_wait_us,
                "cxl_link_busy_ratio": attention_result.cxl_link_busy_ratio,
            }
        )
    return summary, layer_rows


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


def _report(output: Path, rows: list[dict[str, Any]], exact_runs: int) -> None:
    lines = [
        "# KV CXL phase-faithful Stage 6",
        "",
        "Stage 6 separates Attention KV traffic from the later Expert phase to match the current Decode dependencies.",
        "Identical shapes are executed once and shared by `gpu-only` and `sieve` consumers.",
        "CXL remains a hypothetical memory-only FIFO profile; these are Ramulator and analytic event-graph results, not A800 measurements.",
        "",
        "| scale | policy | total us | throughput tok/s | attention share | CXL critical us | CXL per-layer us | link busy |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['request_scale']} | {row['policy']} | {row['total_latency_us']:.6f} | "
            f"{row['throughput_request_tokens_per_s']:.3f} | {row['attention_latency_share']:.6f} | "
            f"{row['cxl_critical_path_us']:.6f} | {row['cxl_kv_completion_us_per_layer']:.6f} | "
            f"{row['cxl_link_busy_ratio']:.6f} |"
        )
    lines += [
        "",
        "The scale=4096 result is an exact request-level run with request counts scaled for tractability; the scale=1 result is an exact full-count run for the generated B8/C32k shape.",
        "It does not implement paging transfers, eviction, recovery, a physical CXL protocol, or CXL-PIM computation.",
        "Aggregate queue wait is reported separately and is never treated as one request's critical-path latency.",
        "",
        f"Unique exact Ramulator runs: `{exact_runs}`.",
        "",
    ]
    (output / "stage6_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_stage6(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    request_scales: tuple[int, ...] = REQUEST_SCALES,
    policies: tuple[str, ...] = POLICIES,
    workload_runner: Callable[..., CXLKVWorkloadResult] = run_cxl_kv_workload,
) -> dict[str, Any]:
    output = Path(output_dir)
    if output.exists():
        existing = {item.name for item in output.iterdir()}
        allowed_resume = existing.issubset({"exact_cache"})
        if existing and not allowed_resume:
            raise ValueError(f"Stage-6 output already exists and is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if not request_scales or any(scale <= 0 for scale in request_scales):
        raise ValueError("Stage-6 request scales must be positive")
    if not policies or any(policy not in POLICIES for policy in policies):
        raise ValueError("unknown Stage-6 policy")
    configuration, source = stage5._source_configuration()
    template = stage5._template(source, BATCH_SIZE, CONTEXT_LENGTH)
    cycle = SieveCycleV1Config.load(stage5.CYCLE)
    capacity_rows = stage5._capacity_rows()
    capacity = capacity_rows[
        ("a800", BATCH_SIZE, CONTEXT_LENGTH, int(stage5.PROFILE["capacity_gb"]) * 1_000_000_000)
    ]
    simulated_capacity = capacity_rows[("simulated", BATCH_SIZE, CONTEXT_LENGTH, 0)]
    simulation_context = stage5.cxl_kv_cache_context(ROOT, stage5.CYCLE)

    planned: dict[tuple[int, str], dict[str, Any]] = {}
    consumer_shapes: dict[tuple[int, str], dict[str, dict[str, int]]] = {}
    consumer_decisions: dict[tuple[int, str], PlacementDecision] = {}
    decision_timing = AnalyticTimingModel(configuration.model, configuration.hardware)
    for scale in request_scales:
        for policy_name in policies:
            decision = create_policy(policy_name, decision_timing).place(template)
            phase_shapes = _phase_shapes(
                template, decision, capacity, configuration, cycle, scale
            )
            consumer_shapes[(scale, policy_name)] = phase_shapes
            consumer_decisions[(scale, policy_name)] = decision
            for phase, shape in phase_shapes.items():
                identity = _shape_identity(phase, shape, scale, simulation_context)
                key = _key(identity)
                entry = planned.setdefault(
                    (scale, key),
                    {
                        "phase": phase,
                        "shape": shape,
                        "request_scale": scale,
                        "consumers": [],
                    },
                )
                entry["consumers"].append(policy_name)

    results_by_key: dict[tuple[int, str], CXLKVWorkloadResult] = {}
    new_runs = 0
    cache_records: list[dict[str, Any]] = []
    for (scale, expected_key), entry in planned.items():
        cache_key, cache_path, result, executed = _run_or_load(
            output,
            str(entry["phase"]),
            dict(entry["shape"]),
            scale,
            cycle,
            simulation_context,
            workload_runner,
        )
        if cache_key != expected_key:
            raise ValueError("Stage-6 plan/cache key mismatch")
        _validate_completion(str(entry["phase"]), dict(entry["shape"]), result, cycle)
        results_by_key[(scale, cache_key)] = result
        new_runs += int(executed)
        cache_records.append(
            {
                "cache_key": cache_key,
                "cache_sha256": stage5._sha256(cache_path),
                "phase": entry["phase"],
                "request_scale": scale,
                "shape": entry["shape"],
                "consumers": sorted(set(entry["consumers"])),
                "executed": executed,
            }
        )

    rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    for scale in request_scales:
        for policy_name in policies:
            shapes = consumer_shapes[(scale, policy_name)]
            keys = {
                phase: _key(_shape_identity(phase, shape, scale, simulation_context))
                for phase, shape in shapes.items()
            }
            phase_results = {
                phase: results_by_key[(scale, key)] for phase, key in keys.items()
            }
            summary, details = _replay(
                configuration,
                template,
                policy_name,
                scale,
                capacity,
                simulated_capacity,
                cycle,
                shapes,
                phase_results,
                keys,
            )
            rows.append(summary)
            layer_rows.extend(details)

    _write_csv(output / "comparison.csv", rows)
    _write_csv(output / "layer_results.csv", layer_rows)
    metadata = {
        "schema_version": 1,
        "stage": "kv-cxl-phase-stage6-v1",
        "method": "phase-faithful-request-level-cxl-in-48-layer-decode-graph",
        "classification": "Ramulator request-level phase milestones plus analytic Decode graph; no A800 or physical CXL timing",
        "case": CASE,
        "request_scales": list(request_scales),
        "policies": list(policies),
        "layers": LAYERS,
        "decode_steps": 1,
        "profile": {"name": stage5.PROFILE_NAME, **stage5.PROFILE},
        "unique_shapes": len(planned),
        "exact_runs": len(planned),
        "new_runs": new_runs,
        "interpolation": 0,
        "shape_deduplication": "simulation inputs only; policy and case labels are consumers",
        "execution": stage5._execution_provenance(),
        "input_hashes": {
            "source_experiment": stage5._sha256(stage5.SOURCE_EXPERIMENT),
            "source_trace": stage5._sha256(configuration.experiment.trace_path),
            "source_manifest": stage5._sha256(configuration.experiment.trace_manifest_path),
            "model": stage5._sha256(configuration.experiment.model_path),
            "hardware": stage5._sha256(configuration.experiment.hardware_path),
            "cycle_config": stage5._sha256(stage5.CYCLE),
            "capacity_source": stage5._sha256(stage5.CAPACITY),
        },
        "cache_records": cache_records,
        "limitations": [
            "Attention KV and Expert requests are separate phases because the current Decode graph orders Expert after Attention.",
            "CXL is a fixed-bandwidth, fixed-latency FIFO memory-only assumption.",
            "Routing is a controlled B8/C4k template assigned C32k context, not a new A800 capture.",
            "GPU arithmetic and Attention compute remain analytic; PIM intermediate Expert milestones are unavailable.",
            "No paging transfer state machine, eviction, recovery, physical CXL protocol, multi-GPU, or CXL-PIM computation.",
        ],
    }
    comparison_path = output / "comparison.json"
    comparison_path.write_text(
        json.dumps({"metadata": metadata, "rows": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _report(output, rows, len(planned))
    manifest = {
        "schema_version": 1,
        "stage": metadata["stage"],
        "case": CASE,
        "request_scales": list(request_scales),
        "policies": list(policies),
        "rows": len(rows),
        "layer_rows": len(layer_rows),
        "unique_shapes": len(planned),
        "exact_runs": len(planned),
        "interpolation": 0,
        "execution": metadata["execution"],
        "input_hashes": metadata["input_hashes"],
        "output_hashes": {
            "comparison.csv": stage5._sha256(output / "comparison.csv"),
            "comparison.json": stage5._sha256(comparison_path),
            "layer_results.csv": stage5._sha256(output / "layer_results.csv"),
            "stage6_report.md": stage5._sha256(output / "stage6_report.md"),
        },
    }
    (output / "stage_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"metadata": metadata, "rows": rows, "stage_manifest": manifest}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--request-scales", default="4096,1")
    parser.add_argument("--policies", default=",".join(POLICIES))
    args = parser.parse_args(argv)
    request_scales = tuple(int(value) for value in args.request_scales.split(",") if value)
    policies = tuple(value for value in args.policies.split(",") if value)
    result = run_stage6(
        args.output, request_scales=request_scales, policies=policies
    )
    print(json.dumps(result["stage_manifest"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
