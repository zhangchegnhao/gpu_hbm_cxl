#!/usr/bin/env python3
"""Replay exact Stage-10 CXL-PIM timing in a calibrated 48-layer Decode graph.

Stage 11 performs no new Ramulator runs.  It binds the full-count Stage-6
Expert milestone, the Stage-7 A800/Local-HBM bridge, and the three exact
Stage-10 unified CXL-PIM pipelines.  The B8/C32k route is still the controlled
B8/C4k template with its context field changed to C32k.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_kv_cxl_decode_stage5 as stage5
import run_kv_cxl_phase_stage6 as stage6
import run_kv_cxl_pim_pipeline_stage10 as stage10
from sieve_replay.ramulator.cxl_kv_workload import CXLKVWorkloadResult
from sieve_replay.ramulator.cxl_pim_pipeline import CXLPIMPipelineResult
from sieve_replay.simulation import EventEngine
from sieve_replay.simulation.event import Event
from sieve_replay.simulation.layer_graph import build_layer_graph
from sieve_replay.timing import AnalyticTimingModel, CoupledExpertTiming
from sieve_replay.trace import TraceBatch
from sieve_replay.types import AttentionTarget, ExpertLoad, PlacementDecision, TimingEstimate


STAGE6_DIR = ROOT / "results/kv_cxl_phase_stage6_b8_c32k_formal_v1"
STAGE6_COMPARISON = STAGE6_DIR / "comparison.json"
STAGE6_MANIFEST = STAGE6_DIR / "stage_manifest.json"
STAGE6_VALIDATION = STAGE6_DIR / "validation.json"
STAGE6_LAYERS = STAGE6_DIR / "layer_results.csv"
STAGE7_DIR = ROOT / "results/kv_cxl_a800_bridge_stage7_v1"
STAGE7_COMPARISON = STAGE7_DIR / "comparison.json"
STAGE7_CALIBRATION = STAGE7_DIR / "calibration.json"
STAGE7_MANIFEST = STAGE7_DIR / "stage_manifest.json"
STAGE7_VALIDATION = STAGE7_DIR / "validation.json"
STAGE10_DIR = ROOT / "results/kv_cxl_pim_pipeline_stage10_v1"
STAGE10_COMPARISON = STAGE10_DIR / "pipeline_comparison.json"
STAGE10_MANIFEST = STAGE10_DIR / "stage_manifest.json"
STAGE10_VALIDATION = STAGE10_DIR / "validation.json"
DEFAULT_OUTPUT = ROOT / "results/kv_cxl_pim_decode_stage11_v1"

CASE = "b8_c32k"
BATCH_SIZE = 8
CONTEXT_LENGTH = 32_768
LAYERS = 48
POLICY = "gpu-only-expert-placement"
DESIGN_IDS = stage10.DESIGN_IDS
SCHEDULE_MODES = ("ideal-overlap-bound", "fully-serialized-bound")

ATTENTION_CATEGORIES = {
    "qkv",
    "rope",
    "attention",
    "attention_output",
    "cxl_memory_penalty",
    "cxl_pim_attention",
}
ROUTING_CATEGORIES = {"router", "metadata", "dispatch", "scheduler"}
EXPERT_CATEGORIES = {
    "gpu_weight_load",
    "gpu_expert",
    "pim_write",
    "pim_expert",
    "pim_read",
}
COMBINE_CATEGORIES = {"combine"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _require_validation(path: Path, stage: str) -> None:
    validation = _read_json(path)
    recorded_stage = validation.get("stage")
    if validation.get("status") != "passed" or recorded_stage not in {None, stage}:
        raise ValueError(f"Stage 11 requires passed validation for {stage}")


def _load_stage6() -> dict[str, Any]:
    _require_validation(STAGE6_VALIDATION, "kv-cxl-phase-stage6-v1")
    raw = _read_json(STAGE6_COMPARISON)
    metadata = raw["metadata"]
    if metadata.get("stage") != "kv-cxl-phase-stage6-v1":
        raise ValueError("Stage-6 comparison has an unexpected stage identifier")
    rows = [
        row
        for row in raw["rows"]
        if int(row["request_scale"]) == 1 and row["policy"] == "gpu-only"
    ]
    if len(rows) != 1 or metadata["interpolation"] != 0:
        raise ValueError("Stage 11 requires one exact full-count Stage-6 GPU row")
    row = rows[0]
    records = {record["cache_key"]: record for record in metadata["cache_records"]}
    cache_artifacts: dict[str, dict[str, Any]] = {}
    cached_results: dict[str, CXLKVWorkloadResult] = {}
    for phase in ("attention", "expert"):
        key = str(row[f"{phase}_cache_key"])
        record = records.get(key)
        path = STAGE6_DIR / "exact_cache" / f"{key}.json"
        if record is None or record["phase"] != phase:
            raise ValueError(f"Stage-6 {phase} cache record is missing")
        if stage5._sha256(path) != record["cache_sha256"]:
            raise ValueError(f"Stage-6 {phase} cache hash mismatch")
        cached = _read_json(path)
        if cached.get("cache_input", {}).get("phase") != phase:
            raise ValueError(f"Stage-6 {phase} cache identity mismatch")
        result = CXLKVWorkloadResult(**cached["result"])
        shape = row[f"{phase}_shape"]
        if phase == "expert" and (
            result.gpu_completed_requests != int(shape["gpu_read_transactions"])
            or result.pim_completed_requests != 0
        ):
            raise ValueError("Stage-6 Expert completion counts are inconsistent")
        cached_results[phase] = result
        cache_artifacts[phase] = {
            "kind": f"stage6-{phase}",
            "path": _relative(path),
            "sha256": stage5._sha256(path),
            "cache_key": key,
        }
    return {
        "raw": raw,
        "row": row,
        "expert_result": cached_results["expert"],
        "cache_artifacts": cache_artifacts,
    }


def _load_stage7() -> dict[str, Any]:
    _require_validation(STAGE7_VALIDATION, "kv-cxl-a800-bridge-stage7-v1")
    raw = _read_json(STAGE7_COMPARISON)
    rows = {row["scenario"]: row for row in raw["rows"]}
    expected = {"resident-local-counterfactual", "nominal-memory-only-cxl-spill"}
    if set(rows) != expected or raw["metadata"]["interpolation"] != 0:
        raise ValueError("Stage-7 baseline matrix is incomplete")
    resident_record = raw["metadata"]["exact_cache"]
    resident_path = STAGE7_DIR / "exact_cache" / f"{resident_record['key']}.json"
    if stage5._sha256(resident_path) != resident_record["sha256"]:
        raise ValueError("Stage-7 resident cache hash mismatch")
    return {
        "raw": raw,
        "rows": rows,
        "resident_cache": {
            "kind": "stage7-resident-local-attention",
            "path": _relative(resident_path),
            "sha256": stage5._sha256(resident_path),
            "cache_key": resident_record["key"],
        },
    }


def _load_stage10() -> dict[str, Any]:
    _require_validation(STAGE10_VALIDATION, "kv-cxl-pim-pipeline-stage10-v1")
    raw = _read_json(STAGE10_COMPARISON)
    manifest = _read_json(STAGE10_MANIFEST)
    rows = {row["design_id"]: row for row in raw["rows"]}
    records = {row["design_id"]: row for row in manifest["cache_records"]}
    if set(rows) != set(DESIGN_IDS) or set(records) != set(DESIGN_IDS):
        raise ValueError("Stage-10 boundary design matrix is incomplete")
    results: dict[str, CXLPIMPipelineResult] = {}
    artifacts: list[dict[str, Any]] = []
    for design_id in DESIGN_IDS:
        record = records[design_id]
        path = ROOT / record["cache_path"]
        if stage5._sha256(path) != record["sha256"]:
            raise ValueError(f"Stage-10 cache hash mismatch: {design_id}")
        cached = _read_json(path)
        if cached.get("cache_input") != record["cache_input"]:
            raise ValueError(f"Stage-10 cache identity mismatch: {design_id}")
        result = CXLPIMPipelineResult(**cached["result"])
        if not math.isclose(
            result.pipeline_completion_us,
            float(rows[design_id]["unified_branch_us_per_layer"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Stage-10 pipeline timing mismatch: {design_id}")
        results[design_id] = result
        artifacts.append(
            {
                "kind": "stage10-cxl-pim-pipeline",
                "design_id": design_id,
                "path": record["cache_path"],
                "sha256": record["sha256"],
                "cache_key": record["key"],
            }
        )
    return {"raw": raw, "rows": rows, "results": results, "cache_artifacts": artifacts}


def _input_hashes(configuration: Any) -> dict[str, str]:
    assert configuration.experiment.trace_manifest_path is not None
    return {
        "stage6_comparison": stage5._sha256(STAGE6_COMPARISON),
        "stage6_manifest": stage5._sha256(STAGE6_MANIFEST),
        "stage6_validation": stage5._sha256(STAGE6_VALIDATION),
        "stage6_layer_results": stage5._sha256(STAGE6_LAYERS),
        "stage7_comparison": stage5._sha256(STAGE7_COMPARISON),
        "stage7_calibration": stage5._sha256(STAGE7_CALIBRATION),
        "stage7_manifest": stage5._sha256(STAGE7_MANIFEST),
        "stage7_validation": stage5._sha256(STAGE7_VALIDATION),
        "stage10_comparison": stage5._sha256(STAGE10_COMPARISON),
        "stage10_manifest": stage5._sha256(STAGE10_MANIFEST),
        "stage10_validation": stage5._sha256(STAGE10_VALIDATION),
        "source_experiment": stage5._sha256(stage5.SOURCE_EXPERIMENT),
        "source_trace": stage5._sha256(configuration.experiment.trace_path),
        "source_manifest": stage5._sha256(configuration.experiment.trace_manifest_path),
        "model": stage5._sha256(configuration.experiment.model_path),
        "hardware": stage5._sha256(configuration.experiment.hardware_path),
        "capacity_source": stage5._sha256(stage5.CAPACITY),
        "cycle_config": stage5._sha256(stage5.CYCLE),
    }


class Stage11Timing(AnalyticTimingModel):
    """Calibrated Attention module plus exact Stage-6 Expert memory milestone."""

    def __init__(
        self,
        model: Any,
        hardware: Any,
        local_attention_module_us_per_layer: float,
        expert_result: CXLKVWorkloadResult,
        expert_shape: dict[str, int],
        transaction_bytes: int,
    ) -> None:
        super().__init__(model, hardware)
        self.local_attention_module_us_per_layer = local_attention_module_us_per_layer
        self.expert_result = expert_result
        self.expert_shape = expert_shape
        self.transaction_bytes = transaction_bytes
        fixed_attention_us = (
            super().qkv_projection(BATCH_SIZE).duration_us
            + super().rope(BATCH_SIZE).duration_us
            + super().output_projection(BATCH_SIZE).duration_us
        )
        self.local_attention_core_us = local_attention_module_us_per_layer - fixed_attention_us
        if self.local_attention_core_us <= 0.0:
            raise ValueError("A800-calibrated Attention core must be positive")

    def gpu_attention(self, context_lengths: tuple[int, ...]) -> TimingEstimate:
        return TimingEstimate(
            self.local_attention_core_us,
            description="A800-trend-calibrated local GPU Attention core",
        )

    def coupled_expert_timing(
        self,
        gpu_loads: tuple[ExpertLoad, ...],
        pim_loads: tuple[ExpertLoad, ...],
    ) -> CoupledExpertTiming:
        gpu_bytes = int(self.expert_shape["gpu_read_transactions"]) * self.transaction_bytes
        gpu_us = (
            self.expert_result.gpu_completion_cycles
            * self.expert_result.tick_ps
            / 1_000_000.0
        )
        pim_us = (
            self.expert_result.pim_completion_cycles
            * self.expert_result.tick_ps
            / 1_000_000.0
        )
        return CoupledExpertTiming(
            gpu_weight_load=TimingEstimate(
                gpu_us,
                bytes_accessed=float(gpu_bytes),
                description="exact Stage-6 full-count GPU Expert READ milestone",
            ),
            pim_token_write=TimingEstimate(0.0, description="no PIM experts in fixed placement"),
            pim_expert_compute=TimingEstimate(0.0, description="no PIM experts in fixed placement"),
            pim_result_read=TimingEstimate(
                pim_us, description="exact Stage-6 full-count PIM Expert milestone"
            ),
            report={
                "model": "stage11-stage6-expert-reuse-v1",
                "gpu_completion_cycles": self.expert_result.gpu_completion_cycles,
                "pim_completion_cycles": self.expert_result.pim_completion_cycles,
            },
        )


def _decision(stage6_row: dict[str, Any]) -> PlacementDecision:
    placement = stage6_row["placement"]
    if placement["pim_experts"]:
        raise ValueError("Stage 11 expects the frozen Stage-6 GPU-only Expert placement")
    return PlacementDecision(
        policy="gpu-only",
        attention_target=AttentionTarget.GPU,
        gpu_experts=tuple(int(value) for value in placement["gpu_experts"]),
        pim_experts=(),
        estimated_objective_us=0.0,
        search_report={"source": "Stage-6 full-count gpu-only placement"},
    )


def _build_layer_events(
    trace: TraceBatch,
    decision: PlacementDecision,
    timing: Stage11Timing,
    previous_tail: str | None,
    *,
    external_kind: str,
    schedule_mode: str,
    external_us: float,
    non_attention_residual_us: float,
) -> tuple[tuple[Event, ...], str]:
    base = build_layer_graph(trace, decision, timing)
    expanded: list[Event] = []
    for event in base:
        if event.name == "o_proj":
            if external_kind == "cxl-pim":
                dependencies = (
                    ("rope",)
                    if schedule_mode == "ideal-overlap-bound"
                    else ("attention",)
                )
                expanded.append(
                    Event(
                        name="cxl_pim_pipeline",
                        category="cxl_pim_attention",
                        resources=("CXL_PIM_PIPELINE",),
                        dependencies=dependencies,
                        duration_us=external_us,
                        description="exact Stage-10 unified CXL-PIM pipeline milestone",
                    )
                )
                join_dependencies = ("attention", "cxl_pim_pipeline")
            elif external_kind == "memory-only-cxl":
                expanded.append(
                    Event(
                        name="cxl_memory_penalty",
                        category="cxl_memory_penalty",
                        resources=("CXL_MEM_PATH",),
                        dependencies=("attention",),
                        duration_us=external_us,
                        description="paired exact Stage-7 CXL-minus-resident completion delta",
                    )
                )
                join_dependencies = ("attention", "cxl_memory_penalty")
            elif external_kind == "none":
                join_dependencies = ("attention",)
            else:
                raise ValueError(f"unsupported Stage-11 external branch: {external_kind}")
            expanded.append(
                Event(
                    name="attention_join",
                    category="attention_join",
                    resources=(),
                    dependencies=join_dependencies,
                    duration_us=0.0,
                    description="join local GPU and external Attention branches",
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
                    resources=("GPU_COMPUTE",),
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


def _category_totals(
    events: tuple[Any, ...], names: set[str] | None = None
) -> dict[str, float]:
    totals: dict[str, float] = {}
    for item in events:
        if names is not None and item.event.name not in names:
            continue
        totals[item.event.category] = totals.get(item.event.category, 0.0) + item.event.duration_us
    return totals


def _sum_categories(values: dict[str, float], categories: set[str]) -> float:
    return sum(values.get(category, 0.0) for category in categories)


def _calibration_residual_us(
    trace: TraceBatch,
    decision: PlacementDecision,
    timing: Stage11Timing,
    target_non_attention_us_per_layer: float,
) -> tuple[float, float]:
    events, _ = _build_layer_events(
        trace,
        decision,
        timing,
        None,
        external_kind="none",
        schedule_mode="resident",
        external_us=0.0,
        non_attention_residual_us=0.0,
    )
    scheduled = EventEngine().run(events)
    critical = set(EventEngine.critical_path(scheduled))
    values = _category_totals(scheduled, critical)
    attention_us = _sum_categories(values, ATTENTION_CATEGORIES)
    total_us = max(item.end_us for item in scheduled)
    modeled_non_attention_us = total_us - attention_us
    residual = target_non_attention_us_per_layer - modeled_non_attention_us
    if residual < -1e-9:
        raise ValueError("modeled non-Attention path exceeds the A800 calibration")
    return max(0.0, residual), modeled_non_attention_us


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


def _replay(
    configuration: Any,
    template: TraceBatch,
    decision: PlacementDecision,
    expert_result: CXLKVWorkloadResult,
    expert_shape: dict[str, int],
    scenario: dict[str, Any],
    non_attention_residual_us: float,
    transaction_bytes: int,
    memory: dict[str, Any],
    source_keys: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    timing = Stage11Timing(
        configuration.model,
        configuration.hardware,
        float(scenario["local_attention_module_us_per_layer"]),
        expert_result,
        expert_shape,
        transaction_bytes,
    )
    graph: list[Event] = []
    previous_tail: str | None = None
    for layer in range(LAYERS):
        events, previous_tail = _build_layer_events(
            stage6._layer_trace(template, layer),
            decision,
            timing,
            previous_tail,
            external_kind=str(scenario["external_kind"]),
            schedule_mode=str(scenario["schedule_mode"]),
            external_us=float(scenario["external_us_per_layer"]),
            non_attention_residual_us=non_attention_residual_us,
        )
        graph.extend(events)
    scheduled = EventEngine().run(tuple(graph))
    critical_path = EventEngine.critical_path(scheduled)
    critical_names = set(critical_path)
    critical = _category_totals(scheduled, critical_names)
    all_durations = _category_totals(scheduled)
    total_us = max(item.end_us for item in scheduled)
    attention_us = _sum_categories(critical, ATTENTION_CATEGORIES)
    routing_us = _sum_categories(critical, ROUTING_CATEGORIES)
    expert_us = _sum_categories(critical, EXPERT_CATEGORIES)
    combine_us = _sum_categories(critical, COMBINE_CATEGORIES)
    raw_router_us = critical.get("router", 0.0)
    other_us = total_us - attention_us - routing_us - expert_us - combine_us
    if other_us < -1e-8:
        raise ValueError("Stage-11 component accounting is negative")

    summary = {
        "scenario": scenario["scenario"],
        "design_id": scenario["design_id"],
        "schedule_mode": scenario["schedule_mode"],
        "policy": POLICY,
        "case": CASE,
        "batch_size": BATCH_SIZE,
        "context_length": CONTEXT_LENGTH,
        "layer_count": LAYERS,
        "decode_steps": 1,
        "trace_classification": "controlled B8/C4k Router template tiled to B8/C32k; not an A800 B8/C32k Router capture",
        "timing_classification": scenario["timing_classification"],
        "a800_capacity_state": scenario["a800_capacity_state"],
        "a800_capacity_feasible": scenario["a800_capacity_feasible"],
        "simulated_96gb_capacity_state": "resident",
        "simulated_96gb_capacity_feasible": True,
        "kv_cache_bytes": memory["kv_cache_bytes"],
        "peak_memory_bytes": memory["peak_memory_bytes"],
        "local_gpu_attention_us": float(scenario["local_attention_module_us_per_layer"]) * LAYERS,
        "external_attention_us_per_layer": float(scenario["external_us_per_layer"]),
        "cxl_pim_pipeline_us_per_layer": float(scenario["cxl_pim_pipeline_us_per_layer"]),
        "attention_latency_us": attention_us,
        "attention_latency_share": attention_us / total_us,
        "router_latency_us": raw_router_us,
        "routing_control_latency_us": routing_us,
        "expert_latency_us": expert_us,
        "combine_latency_us": combine_us,
        "other_non_attention_latency_us": max(0.0, other_us),
        "total_latency_us": total_us,
        "throughput_tokens_per_s": BATCH_SIZE * 1_000_000.0 / total_us,
        "expert_exact_us_per_layer": expert_us / LAYERS,
        "non_attention_calibration_residual_us_per_layer": non_attention_residual_us,
        "all_branch_work_us": sum(all_durations.values()),
        "stage6_expert_cache_key": source_keys["stage6_expert"],
        "stage10_pipeline_cache_key": scenario["pipeline_cache_key"],
        "interpolation": 0,
    }
    layer_rows: list[dict[str, Any]] = []
    for layer in range(LAYERS):
        layer_items = tuple(item for item in scheduled if item.event.layer == layer)
        layer_names = {
            name for name in critical_names if name.startswith(f"step0.layer{layer}.")
        }
        layer_critical = _category_totals(layer_items, layer_names)
        start_us = min(item.start_us for item in layer_items)
        end_us = max(item.end_us for item in layer_items)
        layer_rows.append(
            {
                "scenario": scenario["scenario"],
                "design_id": scenario["design_id"],
                "schedule_mode": scenario["schedule_mode"],
                "policy": POLICY,
                "layer": layer,
                "layer_start_us": start_us,
                "layer_end_us": end_us,
                "layer_span_us": end_us - start_us,
                "attention_latency_us": _sum_categories(layer_critical, ATTENTION_CATEGORIES),
                "routing_control_latency_us": _sum_categories(layer_critical, ROUTING_CATEGORIES),
                "expert_latency_us": _sum_categories(layer_critical, EXPERT_CATEGORIES),
                "combine_latency_us": _sum_categories(layer_critical, COMBINE_CATEGORIES),
                "cxl_pim_pipeline_us": all_durations.get("cxl_pim_attention", 0.0) / LAYERS,
                "interpolation": 0,
            }
        )
    return summary, layer_rows


def _scenarios(stage7_source: dict[str, Any], stage10_source: dict[str, Any]) -> list[dict[str, Any]]:
    resident = stage7_source["rows"]["resident-local-counterfactual"]
    memory_only = stage7_source["rows"]["nominal-memory-only-cxl-spill"]
    stage10_metadata = stage10_source["raw"]["metadata"]
    full_local_us = float(resident["estimated_attention_ms"]) * 1000.0 / LAYERS
    partial_local_us = float(stage10_metadata["baselines"]["local_gpu_attention_ms"]) * 1000.0 / LAYERS
    memory_penalty_us = (
        float(memory_only["estimated_attention_ms"])
        - float(resident["estimated_attention_ms"])
    ) * 1000.0 / LAYERS
    rows: list[dict[str, Any]] = [
        {
            "scenario": "resident-local-counterfactual",
            "design_id": "none",
            "schedule_mode": "resident",
            "external_kind": "none",
            "external_us_per_layer": 0.0,
            "cxl_pim_pipeline_us_per_layer": 0.0,
            "local_attention_module_us_per_layer": full_local_us,
            "pipeline_cache_key": "",
            "a800_capacity_state": "oom/infeasible-counterfactual",
            "a800_capacity_feasible": False,
            "timing_classification": "A800 C4k-C16k trend extrapolation plus exact Local-HBM counterfactual and exact Stage-6 Expert timing",
        },
        {
            "scenario": "memory-only-cxl",
            "design_id": "memory-only-64GBps",
            "schedule_mode": "paired-memory-delta",
            "external_kind": "memory-only-cxl",
            "external_us_per_layer": memory_penalty_us,
            "cxl_pim_pipeline_us_per_layer": 0.0,
            "local_attention_module_us_per_layer": full_local_us,
            "pipeline_cache_key": "",
            "a800_capacity_state": "spill",
            "a800_capacity_feasible": True,
            "timing_classification": "A800 C4k-C16k trend extrapolation plus paired exact Ramulator memory delta and exact Stage-6 Expert timing",
        },
    ]
    records = {
        row["design_id"]: row
        for row in _read_json(STAGE10_MANIFEST)["cache_records"]
    }
    for design_id in DESIGN_IDS:
        pipeline_us = stage10_source["results"][design_id].pipeline_completion_us
        for mode in SCHEDULE_MODES:
            rows.append(
                {
                    "scenario": "cxl-pim-attention",
                    "design_id": design_id,
                    "schedule_mode": mode,
                    "external_kind": "cxl-pim",
                    "external_us_per_layer": pipeline_us,
                    "cxl_pim_pipeline_us_per_layer": pipeline_us,
                    "local_attention_module_us_per_layer": partial_local_us,
                    "pipeline_cache_key": records[design_id]["key"],
                    "a800_capacity_state": "spill",
                    "a800_capacity_feasible": True,
                    "timing_classification": "A800 local-branch trend extrapolation plus exact Stage-10 unified CXL-PIM pipeline and exact Stage-6 Expert timing",
                }
            )
    return rows


def _write_report(output: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    memory = next(row for row in rows if row["scenario"] == "memory-only-cxl")
    lines = [
        "# KV CXL-PIM Decode Stage 11",
        "",
        "Stage 11 replays three exact Stage-10 CXL-PIM pipelines in a calibrated 48-layer Decode event graph. It performs no new Ramulator runs.",
        "",
        "| scenario | design | schedule | Attention ms | Expert ms | total ms | tok/s | Attention share |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['design_id']} | {row['schedule_mode']} | "
            f"{row['attention_latency_us'] / 1000.0:.3f} | "
            f"{row['expert_latency_us'] / 1000.0:.3f} | "
            f"{row['total_latency_us'] / 1000.0:.3f} | "
            f"{row['throughput_tokens_per_s']:.3f} | "
            f"{row['attention_latency_share']:.6f} |"
        )
    lines += [
        "",
        f"The memory-only CXL comparison is {memory['total_latency_us'] / 1000.0:.3f} ms per Decode step. Ideal overlap is a scheduling bound, not an implemented runtime policy.",
        f"The event graph uses `{metadata['non_attention_calibration']['residual_us_per_layer']:.6f}` us/layer of calibration residual so that the aggregate non-Attention path matches the independent A800 mean. Router, Expert and Combine component values remain model-derived and must not be read as separate A800 measurements.",
        "",
        "B8/C32k routing is controlled/tiled from the validated B8/C4k Router template. B8/C32k Attention is extrapolated from A800 B8/C4k-C16k measurements. CXL-PIM and memory milestones are Ramulator results for the stated assumptions.",
        "",
        "This experiment does not implement paging, eviction, physical CXL protocol, Shared CXL-PIM, multi-GPU arbitration, energy, or a deployable two-byte partial format.",
        "",
    ]
    (output / "stage11_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_stage11(output_dir: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Stage-11 output already exists and is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    stage6_source = _load_stage6()
    stage7_source = _load_stage7()
    stage10_source = _load_stage10()
    configuration, source = stage5._source_configuration()
    template = stage5._template(source, BATCH_SIZE, CONTEXT_LENGTH)
    decision = _decision(stage6_source["row"])
    cycle = stage6.SieveCycleV1Config.load(stage5.CYCLE)

    resident = stage7_source["rows"]["resident-local-counterfactual"]
    non_attention_total_us = float(
        stage10_source["raw"]["metadata"]["baselines"]["non_attention_ms"]
    ) * 1000.0
    calibration_timing = Stage11Timing(
        configuration.model,
        configuration.hardware,
        float(resident["estimated_attention_ms"]) * 1000.0 / LAYERS,
        stage6_source["expert_result"],
        stage6_source["row"]["expert_shape"],
        cycle.transaction_bytes,
    )
    residual_us, modeled_us = _calibration_residual_us(
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
    source_keys = {
        "stage6_expert": stage6_source["cache_artifacts"]["expert"]["cache_key"]
    }
    rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    for scenario in _scenarios(stage7_source, stage10_source):
        summary, details = _replay(
            configuration,
            template,
            decision,
            stage6_source["expert_result"],
            stage6_source["row"]["expert_shape"],
            scenario,
            residual_us,
            cycle.transaction_bytes,
            memory,
            source_keys,
        )
        rows.append(summary)
        layer_rows.extend(details)

    exact_sources = [
        stage6_source["cache_artifacts"]["attention"],
        stage6_source["cache_artifacts"]["expert"],
        stage7_source["resident_cache"],
        *stage10_source["cache_artifacts"],
    ]
    metadata = {
        "schema_version": 1,
        "stage": "kv-cxl-pim-decode-stage11-v1",
        "case": CASE,
        "method": "A800-calibrated 48-layer Decode event graph with exact reused Ramulator milestones",
        "classification": "mixed-source sensitivity analysis with explicit A800 extrapolation, controlled routing, and Ramulator timing labels",
        "layer_count": LAYERS,
        "decode_steps": 1,
        "policy": POLICY,
        "design_ids": list(DESIGN_IDS),
        "schedule_modes": list(SCHEDULE_MODES),
        "rows": len(rows),
        "layer_rows": len(layer_rows),
        "new_ramulator_runs": 0,
        "reused_exact_cache_artifacts": len(exact_sources),
        "interpolation": 0,
        "trace": {
            "classification": "controlled/tiled routing; not A800 B8/C32k capture",
            "source_case": "b8_c4k",
            "target_case": CASE,
            "source_trace_sha256": stage5._sha256(configuration.experiment.trace_path),
            "source_manifest_sha256": stage5._sha256(configuration.experiment.trace_manifest_path),
        },
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
        "input_hashes": _input_hashes(configuration),
        "exact_sources": exact_sources,
        "limitations": [
            "B8/C32k routing is a controlled B8/C4k template assigned C32k context, not a new A800 capture.",
            "B8/C32k Attention is extrapolated from A800 B8/C4k-C16k measurements.",
            "Ideal overlap is a scheduling bound and is not an implemented runtime policy.",
            "The A800 aggregate non-Attention anchor is distributed with a calibration residual; component-level Router, Expert and Combine values remain modeled.",
            "CXL-PIM topology, link, PIM media and exact request streams are simulator assumptions, not a physical device measurement.",
            "No paging, eviction, recovery, physical CXL protocol, Shared CXL-PIM, multi-GPU arbitration or energy model is implemented.",
        ],
    }

    _write_csv(output / "comparison.csv", rows)
    _write_csv(output / "layer_results.csv", layer_rows)
    comparison_path = output / "comparison.json"
    comparison_path.write_text(
        json.dumps({"metadata": metadata, "rows": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(output, rows, metadata)
    manifest = {
        "schema_version": 1,
        "stage": metadata["stage"],
        "case": CASE,
        "rows": len(rows),
        "layer_rows": len(layer_rows),
        "design_ids": list(DESIGN_IDS),
        "schedule_modes": list(SCHEDULE_MODES),
        "new_ramulator_runs": 0,
        "reused_exact_cache_artifacts": len(exact_sources),
        "interpolation": 0,
        "input_hashes": metadata["input_hashes"],
        "exact_sources": exact_sources,
        "output_hashes": {
            "comparison.csv": stage5._sha256(output / "comparison.csv"),
            "comparison.json": stage5._sha256(comparison_path),
            "layer_results.csv": stage5._sha256(output / "layer_results.csv"),
            "stage11_report.md": stage5._sha256(output / "stage11_report.md"),
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
    result = run_stage11(args.output)
    print(json.dumps(result["stage_manifest"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
