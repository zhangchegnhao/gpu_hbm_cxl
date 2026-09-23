#!/usr/bin/env python3
"""Run Stage-12 exact chunked CXL-PIM Attention and 48-layer Decode replay."""

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
import run_kv_cxl_phase_stage6 as stage6
import run_kv_cxl_pim_decode_stage11 as stage11
import run_kv_cxl_pim_pipeline_stage10 as stage10
import run_kv_cxl_pim_sensitivity_stage9 as stage9
from sieve_replay.ramulator.cxl_pim_chunk_pipeline import (
    CXLPIMChunkPipelineResult,
    cxl_pim_chunk_pipeline_context,
    run_cxl_pim_chunk_pipeline,
)
from sieve_replay.simulation import EventEngine
from sieve_replay.simulation.event import Event
from sieve_replay.simulation.layer_graph import GPU_COMPUTE, build_layer_graph
from sieve_replay.trace import TraceBatch
from sieve_replay.types import PlacementDecision


DEFAULT_OUTPUT = ROOT / "results/kv_cxl_pim_chunk_stage12_v1"
STAGE11_DIR = ROOT / "results/kv_cxl_pim_decode_stage11_v1"
STAGE11_COMPARISON = STAGE11_DIR / "comparison.json"
STAGE11_MANIFEST = STAGE11_DIR / "stage_manifest.json"
STAGE11_VALIDATION = STAGE11_DIR / "validation.json"
STAGE11_LAYERS = STAGE11_DIR / "layer_results.csv"

CASE = stage11.CASE
BATCH_SIZE = stage11.BATCH_SIZE
CONTEXT_LENGTH = stage11.CONTEXT_LENGTH
LAYERS = stage11.LAYERS
POLICY = stage11.POLICY
DESIGN_IDS = stage10.DESIGN_IDS
PIPELINE_CONFIGS = (
    ("chunks1_slots1", 1, 1),
    ("chunks8_slots1", 8, 1),
    ("chunks8_slots2", 8, 2),
)
ATTENTION_CATEGORIES = stage11.ATTENTION_CATEGORIES | {"attention_merge"}
MERGE_FLOPS_PER_STATE_BASE = 10


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _key(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _input_hashes(configuration: Any) -> dict[str, str]:
    return {
        "stage11_comparison": stage5._sha256(STAGE11_COMPARISON),
        "stage11_manifest": stage5._sha256(STAGE11_MANIFEST),
        "stage11_validation": stage5._sha256(STAGE11_VALIDATION),
        "stage11_layer_results": stage5._sha256(STAGE11_LAYERS),
        "stage10_comparison": stage5._sha256(stage11.STAGE10_COMPARISON),
        "stage10_manifest": stage5._sha256(stage11.STAGE10_MANIFEST),
        "stage10_validation": stage5._sha256(stage11.STAGE10_VALIDATION),
        "source_experiment": stage5._sha256(stage5.SOURCE_EXPERIMENT),
        "source_trace": stage5._sha256(configuration.experiment.trace_path),
        "source_manifest": stage5._sha256(
            configuration.experiment.trace_manifest_path
        ),
        "model": stage5._sha256(configuration.experiment.model_path),
        "hardware": stage5._sha256(configuration.experiment.hardware_path),
        "cycle_config": stage5._sha256(stage5.CYCLE),
        "stage12_runner": stage5._sha256(Path(__file__)),
    }


def _load_stage11() -> dict[str, Any]:
    validation = _read_json(STAGE11_VALIDATION)
    if validation.get("status") != "passed":
        raise ValueError("Stage 12 requires passed Stage-11 validation")
    raw = _read_json(STAGE11_COMPARISON)
    if raw["metadata"].get("stage") != "kv-cxl-pim-decode-stage11-v1":
        raise ValueError("unexpected Stage-11 source identifier")
    if raw["metadata"].get("interpolation") != 0:
        raise ValueError("Stage-11 source contains interpolation")
    rows = raw["rows"]
    memory = [row for row in rows if row["scenario"] == "memory-only-cxl"]
    if len(memory) != 1:
        raise ValueError("Stage-11 memory-only reference is missing")
    references: dict[str, dict[str, dict[str, Any]]] = {}
    for design_id in DESIGN_IDS:
        design_rows = {
            row["schedule_mode"]: row
            for row in rows
            if row["scenario"] == "cxl-pim-attention"
            and row["design_id"] == design_id
        }
        if set(design_rows) != set(stage11.SCHEDULE_MODES):
            raise ValueError(f"Stage-11 reference rows are incomplete: {design_id}")
        references[design_id] = design_rows
    return {"raw": raw, "memory": memory[0], "references": references}


def _selected_designs(configuration: Any, cycle: Any) -> dict[str, dict[str, Any]]:
    stage10._load_stage9()
    stage8 = stage9._load_stage8()
    designs = stage9._designs(
        configuration.model,
        cycle,
        int(stage8["derived"]["memory_only_cxl_link_bytes_per_layer"]),
    )
    selected = {
        design["design_id"]: design
        for design in designs
        if design["design_id"] in DESIGN_IDS
    }
    if set(selected) != set(DESIGN_IDS):
        raise ValueError("failed to reconstruct every Stage-12 boundary design")
    return selected


def _cache_input(
    design: dict[str, Any],
    config_id: str,
    chunk_count: int,
    buffer_slots: int,
    execution: dict[str, str],
    input_hashes: dict[str, str],
) -> dict[str, Any]:
    shape = design["shape"]
    return {
        "schema_version": 1,
        "stage": "kv-cxl-pim-chunk-stage12-v1",
        "case": CASE,
        "design_id": design["design_id"],
        "config_id": config_id,
        "chunk_count": chunk_count,
        "buffer_slots": buffer_slots,
        "topology": asdict(design["topology"]),
        "mac_interval_ps": design["mac_interval_ps"],
        "base_shape": asdict(shape),
        "chunk_shape": {
            "query_link_transactions_once": shape.query_link_transactions,
            "pim_gwrite_waves_once": shape.pim_gwrite_waves,
            "pim_mac_waves_total": shape.pim_mac_waves,
            "pim_read_waves_per_chunk": shape.pim_read_waves,
            "result_link_transactions_per_chunk": shape.result_link_transactions,
        },
        "link_profile": {
            "bandwidth_gb_s": stage5.PROFILE["bandwidth_gb_s"],
            "latency_us": stage5.PROFILE["latency_us"],
            "channel_count": stage5.PROFILE["channel_count"],
        },
        "pipeline_context": cxl_pim_chunk_pipeline_context(ROOT, stage5.CYCLE),
        "execution": execution,
        "input_hashes": input_hashes,
    }


def _run_or_load(
    output: Path,
    design: dict[str, Any],
    config_id: str,
    chunk_count: int,
    buffer_slots: int,
    execution: dict[str, str],
    input_hashes: dict[str, str],
    runner: Callable[..., CXLPIMChunkPipelineResult],
) -> tuple[str, Path, CXLPIMChunkPipelineResult, bool, dict[str, Any]]:
    cache_input = _cache_input(
        design,
        config_id,
        chunk_count,
        buffer_slots,
        execution,
        input_hashes,
    )
    key = _key(cache_input)
    cache_dir = output / "exact_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    if path.is_file():
        cached = _read_json(path)
        if cached.get("cache_input") != cache_input:
            raise ValueError(f"Stage-12 cache provenance mismatch: {path}")
        return (
            key,
            path,
            CXLPIMChunkPipelineResult.from_dict(cached["result"]),
            False,
            cache_input,
        )

    shape = design["shape"]
    result = runner(
        stage5.RAMULATOR,
        design["cycle"],
        design["topology"],
        chunk_count=chunk_count,
        buffer_slots=buffer_slots,
        query_link_transactions=shape.query_link_transactions,
        pim_gwrite_waves=shape.pim_gwrite_waves,
        pim_mac_waves=shape.pim_mac_waves,
        pim_read_waves_per_chunk=shape.pim_read_waves,
        result_link_transactions_per_chunk=shape.result_link_transactions,
        cxl_bandwidth_bytes_per_second=float(stage5.PROFILE["bandwidth_gb_s"])
        * 1e9,
        cxl_latency_us=float(stage5.PROFILE["latency_us"]),
        link_channels=int(stage5.PROFILE["channel_count"]),
    )
    path.write_text(
        json.dumps(
            {"cache_input": cache_input, "result": asdict(result)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return key, path, result, True, cache_input


def _stage10_identity(
    chunked: CXLPIMChunkPipelineResult, stage10_result: Any
) -> dict[str, Any]:
    stage10_cycles = {
        "query_link_completion_cycles": stage10_result.query_link_completion_cycles,
        "pim_gwrite_start_cycles": stage10_result.pim_gwrite_start_cycles,
        "pim_gwrite_completion_cycles": stage10_result.pim_gwrite_completion_cycles,
        "pim_mac_start_cycles": stage10_result.pim_mac_start_cycles,
        "pim_mac_completion_cycles": stage10_result.pim_mac_completion_cycles,
        "pim_read_start_cycles": stage10_result.pim_read_start_cycles,
        "pim_read_completion_cycles": stage10_result.pim_read_completion_cycles,
        "result_link_start_cycles": stage10_result.result_link_start_cycles,
        "result_link_completion_cycles": stage10_result.result_link_completion_cycles,
        "pipeline_completion_cycles": stage10_result.pipeline_completion_cycles,
    }
    chunk_cycles = {
        "query_link_completion_cycles": chunked.query_link_completion_cycles,
        "pim_gwrite_start_cycles": chunked.pim_gwrite_start_cycles,
        "pim_gwrite_completion_cycles": chunked.pim_gwrite_completion_cycles,
        "pim_mac_start_cycles": chunked.chunk_mac_start_cycles[0],
        "pim_mac_completion_cycles": chunked.chunk_mac_completion_cycles[0],
        "pim_read_start_cycles": chunked.chunk_read_start_cycles[0],
        "pim_read_completion_cycles": chunked.chunk_read_completion_cycles[0],
        "result_link_start_cycles": chunked.chunk_result_link_start_cycles[0],
        "result_link_completion_cycles": chunked.chunk_result_link_completion_cycles[0],
        "pipeline_completion_cycles": chunked.pipeline_completion_cycles,
    }
    differing = [
        name for name, value in stage10_cycles.items() if chunk_cycles[name] != value
    ]
    return {
        "passed": not differing,
        "fields_checked": len(stage10_cycles),
        "differing_fields": differing,
        "stage10_cycles": stage10_cycles,
        "stage12_cycles": chunk_cycles,
    }


def _merge_model(configuration: Any, design: dict[str, Any]) -> dict[str, Any]:
    model = configuration.model
    hardware = configuration.hardware
    states = BATCH_SIZE * model.num_attention_heads * design["topology"].total_pseudo_channels
    flops_per_state = 3 * model.head_dim + MERGE_FLOPS_PER_STATE_BASE
    flops = states * flops_per_state
    partial_bytes = int(design["shape"].partial_result_bytes)
    accumulator_read_write_bytes = 2 * partial_bytes
    bytes_accessed = partial_bytes + accumulator_read_write_bytes
    compute_us = flops / (
        hardware.gpu_peak_tflops * 1e12 * hardware.gpu_compute_efficiency
    ) * 1e6
    memory_us = bytes_accessed / (
        hardware.hbm_bandwidth_tb_s * 1e12 * hardware.hbm_bandwidth_efficiency
    ) * 1e6
    duration_us = max(compute_us, memory_us) + hardware.gpu_kernel_overhead_us
    return {
        "classification": "analytic roofline assumption; not A800 measured",
        "partial_states_per_chunk": states,
        "flops_per_state": flops_per_state,
        "flops_per_chunk": flops,
        "partial_result_bytes_per_chunk": partial_bytes,
        "accumulator_read_write_bytes_per_chunk": accumulator_read_write_bytes,
        "bytes_accessed_per_chunk": bytes_accessed,
        "compute_us_per_chunk": compute_us,
        "memory_us_per_chunk": memory_us,
        "kernel_overhead_us_per_chunk": hardware.gpu_kernel_overhead_us,
        "duration_us_per_chunk": duration_us,
        "gpu_peak_tflops_assumption": hardware.gpu_peak_tflops,
        "gpu_hbm_bandwidth_tb_s_assumption": hardware.hbm_bandwidth_tb_s,
    }


def _build_chunk_layer_events(
    trace: TraceBatch,
    decision: PlacementDecision,
    timing: stage11.Stage11Timing,
    previous_tail: str | None,
    *,
    partial_ready_us: tuple[float, ...],
    merge_us_per_chunk: float,
    non_attention_residual_us: float,
) -> tuple[tuple[Event, ...], str]:
    if not partial_ready_us or tuple(sorted(partial_ready_us)) != partial_ready_us:
        raise ValueError("partial-ready milestones must be nonempty and ordered")
    base = build_layer_graph(trace, decision, timing)
    expanded: list[Event] = []
    for event in base:
        if event.name == "o_proj":
            previous_ready = 0.0
            for chunk, ready_us in enumerate(partial_ready_us):
                ready_name = f"cxl_partial_ready_{chunk}"
                ready_dependency = "rope" if chunk == 0 else f"cxl_partial_ready_{chunk - 1}"
                expanded.append(
                    Event(
                        name=ready_name,
                        category="cxl_pim_attention",
                        resources=("CXL_PIM_TIMELINE",),
                        dependencies=(ready_dependency,),
                        duration_us=ready_us - previous_ready,
                        description="exact Ramulator partial-result readiness delta",
                    )
                )
                expanded.append(
                    Event(
                        name=f"gpu_partial_merge_{chunk}",
                        category="attention_merge",
                        resources=(GPU_COMPUTE,),
                        dependencies=(ready_name,),
                        duration_us=merge_us_per_chunk,
                        description="analytic FP32 online-softmax partial merge",
                    )
                )
                previous_ready = ready_us
            expanded.append(
                Event(
                    name="attention_join",
                    category="attention_join",
                    resources=(),
                    dependencies=(
                        "attention",
                        f"gpu_partial_merge_{len(partial_ready_us) - 1}",
                    ),
                    duration_us=0.0,
                    description="join local Attention and all merged CXL-PIM chunks",
                )
            )
            expanded.append(
                Event(
                    name=event.name,
                    category=event.category,
                    resources=event.resources,
                    dependencies=("attention_join",),
                    duration_us=event.duration_us,
                    description=event.description,
                )
            )
            continue
        if event.name == "residual2":
            expanded.append(
                Event(
                    name="a800_non_attention_residual",
                    category="non_attention_residual",
                    resources=(GPU_COMPUTE,),
                    dependencies=("combine",),
                    duration_us=non_attention_residual_us,
                    description="A800 non-Attention aggregate calibration residual",
                )
            )
            expanded.append(
                Event(
                    name=event.name,
                    category=event.category,
                    resources=event.resources,
                    dependencies=("a800_non_attention_residual",),
                    duration_us=event.duration_us,
                    description=event.description,
                )
            )
            continue
        expanded.append(event)

    prefix = f"step{trace.step}.layer{trace.layer}."
    namespaced: list[Event] = []
    for event in expanded:
        dependencies = tuple(prefix + dependency for dependency in event.dependencies)
        if event.name == "norm1" and previous_tail is not None:
            dependencies = (previous_tail,)
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


def _replay(
    configuration: Any,
    template: TraceBatch,
    decision: PlacementDecision,
    expert_result: Any,
    expert_shape: dict[str, int],
    result: CXLPIMChunkPipelineResult,
    design: dict[str, Any],
    config_id: str,
    merge: dict[str, Any],
    cache_key: str,
    non_attention_residual_us: float,
    transaction_bytes: int,
    memory: dict[str, Any],
    stage11_source: dict[str, Any],
    stage6_expert_key: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    local_attention_us = float(
        _read_json(stage11.STAGE10_COMPARISON)["metadata"]["baselines"][
            "local_gpu_attention_ms"
        ]
    ) * 1000.0 / LAYERS
    timing = stage11.Stage11Timing(
        configuration.model,
        configuration.hardware,
        local_attention_us,
        expert_result,
        expert_shape,
        transaction_bytes,
    )
    graph: list[Event] = []
    previous_tail: str | None = None
    for layer in range(LAYERS):
        events, previous_tail = _build_chunk_layer_events(
            stage6._layer_trace(template, layer),
            decision,
            timing,
            previous_tail,
            partial_ready_us=result.partial_ready_us,
            merge_us_per_chunk=float(merge["duration_us_per_chunk"]),
            non_attention_residual_us=non_attention_residual_us,
        )
        graph.extend(events)
    scheduled = EventEngine().run(tuple(graph))
    critical_names = set(EventEngine.critical_path(scheduled))
    critical = stage11._category_totals(scheduled, critical_names)
    all_durations = stage11._category_totals(scheduled)
    total_us = max(item.end_us for item in scheduled)
    attention_us = stage11._sum_categories(critical, ATTENTION_CATEGORIES)
    routing_us = stage11._sum_categories(critical, stage11.ROUTING_CATEGORIES)
    expert_us = stage11._sum_categories(critical, stage11.EXPERT_CATEGORIES)
    combine_us = stage11._sum_categories(critical, stage11.COMBINE_CATEGORIES)
    other_us = total_us - attention_us - routing_us - expert_us - combine_us
    if other_us < -1e-8:
        raise ValueError("Stage-12 component accounting is negative")

    references = stage11_source["references"][design["design_id"]]
    ideal_us = float(references["ideal-overlap-bound"]["total_latency_us"])
    serial_us = float(references["fully-serialized-bound"]["total_latency_us"])
    memory_us = float(stage11_source["memory"]["total_latency_us"])
    shape = design["shape"]
    chunk_count = result.chunk_count
    summary = {
        "scenario": "cxl-pim-chunked-attention",
        "design_id": design["design_id"],
        "config_id": config_id,
        "chunk_count": chunk_count,
        "buffer_slots": result.buffer_slots,
        "policy": POLICY,
        "case": CASE,
        "batch_size": BATCH_SIZE,
        "context_length": CONTEXT_LENGTH,
        "layer_count": LAYERS,
        "decode_steps": 1,
        "trace_classification": "controlled B8/C4k Router template tiled to B8/C32k; not an A800 B8/C32k Router capture",
        "timing_classification": "A800 local-branch trend extrapolation plus exact chunked Ramulator CXL-PIM milestones, analytic GPU merge, and exact Stage-6 Expert timing",
        "scheduling_assumption": "RoPE launches local Attention and CXL-PIM together; local GPU Attention is non-preemptive and precedes ready merge kernels on shared GPU_COMPUTE",
        "a800_capacity_state": "spill",
        "a800_capacity_feasible": True,
        "simulated_96gb_capacity_state": "resident",
        "simulated_96gb_capacity_feasible": True,
        "kv_cache_bytes": memory["kv_cache_bytes"],
        "peak_memory_bytes": memory["peak_memory_bytes"],
        "query_bytes_per_layer": int(shape.query_bytes),
        "partial_result_bytes_per_chunk": int(shape.partial_result_bytes),
        "total_cxl_link_bytes_per_layer": int(shape.query_bytes)
        + chunk_count * int(shape.partial_result_bytes),
        "stage10_link_bytes_per_layer": int(shape.query_bytes)
        + int(shape.partial_result_bytes),
        "link_byte_inflation_over_stage10": (
            int(shape.query_bytes) + chunk_count * int(shape.partial_result_bytes)
        )
        / (int(shape.query_bytes) + int(shape.partial_result_bytes)),
        "cxl_pim_pipeline_us_per_layer": result.pipeline_completion_us,
        "first_partial_ready_us": result.partial_ready_us[0],
        "last_partial_ready_us": result.partial_ready_us[-1],
        "gpu_merge_us_per_chunk": float(merge["duration_us_per_chunk"]),
        "gpu_merge_us_per_layer": float(merge["duration_us_per_chunk"]) * chunk_count,
        "local_gpu_attention_us": local_attention_us * LAYERS,
        "attention_latency_us": attention_us,
        "attention_latency_share": attention_us / total_us,
        "router_latency_us": critical.get("router", 0.0),
        "routing_control_latency_us": routing_us,
        "expert_latency_us": expert_us,
        "combine_latency_us": combine_us,
        "other_non_attention_latency_us": max(0.0, other_us),
        "total_latency_us": total_us,
        "throughput_tokens_per_s": BATCH_SIZE * 1_000_000.0 / total_us,
        "stage11_ideal_overlap_total_us": ideal_us,
        "stage11_fully_serialized_total_us": serial_us,
        "stage11_memory_only_total_us": memory_us,
        "delta_vs_stage11_ideal_us": total_us - ideal_us,
        "delta_vs_stage11_serialized_us": total_us - serial_us,
        "delta_vs_memory_only_us": total_us - memory_us,
        "throughput_change_vs_memory_only": memory_us / total_us - 1.0,
        "max_active_buffers": result.max_active_buffers,
        "buffer_stall_cycles": result.buffer_stall_cycles,
        "all_cxl_pim_work_us": all_durations.get("cxl_pim_attention", 0.0),
        "all_gpu_merge_work_us": all_durations.get("attention_merge", 0.0),
        "non_attention_calibration_residual_us_per_layer": non_attention_residual_us,
        "stage6_expert_cache_key": stage6_expert_key,
        "stage12_pipeline_cache_key": cache_key,
        "interpolation": 0,
    }

    layer_rows: list[dict[str, Any]] = []
    for layer in range(LAYERS):
        layer_items = tuple(item for item in scheduled if item.event.layer == layer)
        layer_names = {
            name for name in critical_names if name.startswith(f"step0.layer{layer}.")
        }
        layer_critical = stage11._category_totals(layer_items, layer_names)
        start_us = min(item.start_us for item in layer_items)
        end_us = max(item.end_us for item in layer_items)
        merges = [
            item
            for item in layer_items
            if item.event.name.startswith(f"step0.layer{layer}.gpu_partial_merge_")
        ]
        layer_rows.append(
            {
                "design_id": design["design_id"],
                "config_id": config_id,
                "chunk_count": chunk_count,
                "buffer_slots": result.buffer_slots,
                "policy": POLICY,
                "layer": layer,
                "layer_start_us": start_us,
                "layer_end_us": end_us,
                "layer_span_us": end_us - start_us,
                "attention_latency_us": stage11._sum_categories(
                    layer_critical, ATTENTION_CATEGORIES
                ),
                "routing_control_latency_us": stage11._sum_categories(
                    layer_critical, stage11.ROUTING_CATEGORIES
                ),
                "expert_latency_us": stage11._sum_categories(
                    layer_critical, stage11.EXPERT_CATEGORIES
                ),
                "combine_latency_us": stage11._sum_categories(
                    layer_critical, stage11.COMBINE_CATEGORIES
                ),
                "first_partial_ready_offset_us": result.partial_ready_us[0],
                "last_partial_ready_offset_us": result.partial_ready_us[-1],
                "first_merge_start_offset_us": merges[0].start_us - start_us,
                "last_merge_end_offset_us": merges[-1].end_us - start_us,
                "gpu_merge_us": sum(item.event.duration_us for item in merges),
                "cxl_pim_pipeline_us": result.pipeline_completion_us,
                "interpolation": 0,
            }
        )
    return summary, layer_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    output: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]
) -> None:
    lines = [
        "# KV CXL-PIM chunk pipeline Stage 12",
        "",
        "Stage 12 replaces the Stage-11 overlap bounds with exact chunk readiness and an explicit GPU merge model for B8/C32k.",
        "",
        "| design | chunks/slots | pipeline us/layer | merge us/layer | Decode ms | tok/s | vs memory-only |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['design_id']} | {row['chunk_count']}/{row['buffer_slots']} | "
            f"{row['cxl_pim_pipeline_us_per_layer']:.3f} | "
            f"{row['gpu_merge_us_per_layer']:.3f} | "
            f"{row['total_latency_us'] / 1000.0:.3f} | "
            f"{row['throughput_tokens_per_s']:.3f} | "
            f"{row['throughput_change_vs_memory_only'] * 100.0:+.3f}% |"
        )
    lines += [
        "",
        f"Exact Ramulator workloads: `{metadata['exact_runs']}`; new executions in this invocation: `{metadata['new_runs_executed']}`; interpolation: `0`.",
        f"All three one-chunk runs reproduce the Stage-10 pipeline in `{metadata['one_chunk_identity_fields_per_design']}` cycle fields.",
        "",
        "Each chunk returns a complete FP32 online-softmax partial state. Therefore the 8-chunk cases transfer and read eight partial states per PCH, increasing result traffic rather than assuming a free tile boundary. One buffer slot waits for result-link completion before launching the next chunk; two slots permit the following PIM chunk to overlap the prior result transfer.",
        "",
        "GPU merge is an analytic roofline model using the configured simulated GPU peak and HBM bandwidth plus one kernel launch per chunk. It is not an A800 measurement. The event scheduler launches local Attention and CXL-PIM after RoPE, treats local GPU Attention as non-preemptive, and schedules ready merge kernels afterward on the same GPU_COMPUTE resource.",
        "",
        "B8/C32k routing remains a controlled B8/C4k template and local Attention remains an A800 C4k-C16k trend extrapolation. CXL-PIM topology and timing are Ramulator assumptions, not physical CXL-PIM, Shared CXL-PIM, or multi-GPU results.",
        "",
    ]
    (output / "stage12_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_stage12(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    pipeline_runner: Callable[..., CXLPIMChunkPipelineResult] = run_cxl_pim_chunk_pipeline,
) -> dict[str, Any]:
    output = Path(output_dir)
    if output.exists():
        existing = {item.name for item in output.iterdir()}
        if existing and not existing.issubset({"exact_cache"}):
            raise ValueError(f"Stage-12 output already exists and is not resumable: {output}")
    output.mkdir(parents=True, exist_ok=True)

    stage11_source = _load_stage11()
    stage6_source = stage11._load_stage6()
    stage7_source = stage11._load_stage7()
    stage10_source = stage11._load_stage10()
    configuration, source = stage5._source_configuration()
    template = stage5._template(source, BATCH_SIZE, CONTEXT_LENGTH)
    decision = stage11._decision(stage6_source["row"])
    cycle = stage6.SieveCycleV1Config.load(stage5.CYCLE)
    designs = _selected_designs(configuration, cycle)
    execution = stage5._execution_provenance()
    input_hashes = _input_hashes(configuration)

    resident = stage7_source["rows"]["resident-local-counterfactual"]
    non_attention_total_us = float(
        stage10_source["raw"]["metadata"]["baselines"]["non_attention_ms"]
    ) * 1000.0
    calibration_timing = stage11.Stage11Timing(
        configuration.model,
        configuration.hardware,
        float(resident["estimated_attention_ms"]) * 1000.0 / LAYERS,
        stage6_source["expert_result"],
        stage6_source["row"]["expert_shape"],
        cycle.transaction_bytes,
    )
    residual_us, modeled_us = stage11._calibration_residual_us(
        template,
        decision,
        calibration_timing,
        non_attention_total_us / LAYERS,
    )
    capacity = stage7_source["raw"]["metadata"]["capacity"]
    memory = {
        "kv_cache_bytes": int(capacity["kv_cache_bytes"]),
        "peak_memory_bytes": int(capacity["peak_memory_bytes"]),
    }

    cache_records: list[dict[str, Any]] = []
    exact_rows: list[dict[str, Any]] = []
    results: dict[tuple[str, str], CXLPIMChunkPipelineResult] = {}
    new_runs = 0
    identity_checks: dict[str, dict[str, Any]] = {}
    for design_id in DESIGN_IDS:
        design = designs[design_id]
        for config_id, chunk_count, buffer_slots in PIPELINE_CONFIGS:
            key, path, result, executed, cache_input = _run_or_load(
                output,
                design,
                config_id,
                chunk_count,
                buffer_slots,
                execution,
                input_hashes,
                pipeline_runner,
            )
            new_runs += int(executed)
            results[(design_id, config_id)] = result
            cache_records.append(
                {
                    "design_id": design_id,
                    "config_id": config_id,
                    "key": key,
                    "cache_path": _relative(path),
                    "sha256": stage5._sha256(path),
                    "executed_this_invocation": executed,
                    "cache_input": cache_input,
                }
            )
            if chunk_count == 1:
                identity = _stage10_identity(result, stage10_source["results"][design_id])
                identity_checks[design_id] = identity
                if not identity["passed"]:
                    raise ValueError(
                        f"Stage-12 one-chunk identity failed: {design_id}: "
                        f"{identity['differing_fields']}"
                    )
            exact_rows.append(
                {
                    "design_id": design_id,
                    "config_id": config_id,
                    "chunk_count": chunk_count,
                    "buffer_slots": buffer_slots,
                    "pipeline_completion_cycles": result.pipeline_completion_cycles,
                    "pipeline_completion_us": result.pipeline_completion_us,
                    "first_partial_ready_cycles": result.chunk_result_link_completion_cycles[0],
                    "last_partial_ready_cycles": result.chunk_result_link_completion_cycles[-1],
                    "max_active_buffers": result.max_active_buffers,
                    "buffer_stall_cycles": result.buffer_stall_cycles,
                    "query_link_requests": result.query_link_completed_requests,
                    "pim_gwrite_requests": result.pim_gwrite_completed_requests,
                    "pim_mac_requests": result.pim_mac_completed_requests,
                    "pim_read_requests": result.pim_read_completed_requests,
                    "result_link_requests": result.result_link_completed_requests,
                    "cache_key": key,
                    "interpolation": 0,
                }
            )

    rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    merge_models: dict[str, dict[str, Any]] = {}
    records_by_key = {
        (record["design_id"], record["config_id"]): record
        for record in cache_records
    }
    for design_id in DESIGN_IDS:
        design = designs[design_id]
        merge = _merge_model(configuration, design)
        merge_models[design_id] = merge
        for config_id, _, _ in PIPELINE_CONFIGS:
            result = results[(design_id, config_id)]
            summary, details = _replay(
                configuration,
                template,
                decision,
                stage6_source["expert_result"],
                stage6_source["row"]["expert_shape"],
                result,
                design,
                config_id,
                merge,
                records_by_key[(design_id, config_id)]["key"],
                residual_us,
                cycle.transaction_bytes,
                memory,
                stage11_source,
                stage6_source["cache_artifacts"]["expert"]["cache_key"],
            )
            rows.append(summary)
            layer_rows.extend(details)

    metadata = {
        "schema_version": 1,
        "stage": "kv-cxl-pim-chunk-stage12-v1",
        "case": CASE,
        "method": "exact chunked CXL-PIM Ramulator milestones plus explicit analytic GPU merge in the calibrated 48-layer Decode graph",
        "classification": "mixed A800 extrapolation, exact Ramulator simulation, controlled routing, and analytic merge sensitivity",
        "design_ids": list(DESIGN_IDS),
        "pipeline_configs": [
            {
                "config_id": config_id,
                "chunk_count": chunk_count,
                "buffer_slots": buffer_slots,
            }
            for config_id, chunk_count, buffer_slots in PIPELINE_CONFIGS
        ],
        "exact_runs": len(exact_rows),
        "new_runs_executed": new_runs,
        "failed_runs": 0,
        "remaining_runs": 0,
        "interpolation": 0,
        "rows": len(rows),
        "layer_rows": len(layer_rows),
        "one_chunk_identity_fields_per_design": 10,
        "one_chunk_identity": identity_checks,
        "execution": execution,
        "input_hashes": input_hashes,
        "cache_records": cache_records,
        "merge_models": merge_models,
        "non_attention_calibration": {
            "a800_mean_us_per_decode": non_attention_total_us,
            "modeled_explicit_us_per_layer": modeled_us,
            "residual_us_per_layer": residual_us,
            "interpretation": "aggregate residual only; Router/Expert/Combine are not separately measured A800 components",
        },
        "capacity_domains": {
            "a800_local_capacity_bytes": 80_000_000_000,
            "simulated_local_capacity_bytes": 96_000_000_000,
            "cxl_capacity_bytes": int(capacity["cxl_capacity_bytes"]),
            "domains_are_not_combined": True,
        },
        "limitations": [
            "B8/C32k routing is a controlled B8/C4k template assigned C32k context, not a new A800 capture.",
            "B8/C32k local Attention is extrapolated from A800 B8/C4k-C16k measurements.",
            "GPU merge is an analytic roofline assumption using the configured simulated GPU and is not A800 measured.",
            "The scheduler is deterministic and non-preemptive; local GPU Attention is listed before merge kernels on GPU_COMPUTE.",
            "Each chunk returns a full FP32 partial state, so eight chunks multiply PIM READ and result-link traffic by eight.",
            "Link and PIM media controllers are separate simulated resources, not a unified physical CXL controller.",
            "No paging, eviction, recovery, physical CXL protocol, Shared CXL-PIM, multi-GPU arbitration or energy model is implemented.",
        ],
    }

    _write_csv(output / "exact_workloads.csv", exact_rows)
    (output / "exact_workloads.json").write_text(
        json.dumps({"metadata": metadata, "rows": exact_rows}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    _write_csv(output / "comparison.csv", rows)
    comparison_path = output / "comparison.json"
    comparison_path.write_text(
        json.dumps({"metadata": metadata, "rows": rows}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    _write_csv(output / "layer_results.csv", layer_rows)
    _write_report(output, rows, metadata)
    output_names = (
        "comparison.csv",
        "comparison.json",
        "exact_workloads.csv",
        "exact_workloads.json",
        "layer_results.csv",
        "stage12_report.md",
    )
    manifest = {
        "schema_version": 1,
        "stage": metadata["stage"],
        "case": CASE,
        "design_ids": list(DESIGN_IDS),
        "exact_runs": len(exact_rows),
        "new_runs_executed": new_runs,
        "failed_runs": 0,
        "remaining_runs": 0,
        "interpolation": 0,
        "rows": len(rows),
        "layer_rows": len(layer_rows),
        "execution": execution,
        "input_hashes": input_hashes,
        "cache_records": cache_records,
        "one_chunk_identity": identity_checks,
        "output_hashes": {
            name: stage5._sha256(output / name) for name in output_names
        },
    }
    (output / "stage_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"metadata": metadata, "rows": rows, "stage_manifest": manifest}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    result = run_stage12(args.output)
    print(json.dumps(result["stage_manifest"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
