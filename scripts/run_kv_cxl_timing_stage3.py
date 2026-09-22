#!/usr/bin/env python3
"""Run the stage-3 analytic CXL memory-only KV timing sensitivity study.

The study reuses one manifest-validated real pilot layer-batch as a controlled
routing template, tiles it to the requested Batch x Context points, and runs
the complete 48-layer Decode event graph for analytic policies. CXL profiles
are explicit sensitivity assumptions, not hardware measurements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from sieve_replay.config import load_configuration
from sieve_replay.model import classify_capacity, estimate_memory_footprint
from sieve_replay.policy import create_policy
from sieve_replay.report.writer import sha256_file
from sieve_replay.simulation import EventEngine
from sieve_replay.simulation.decode_graph import build_decode_layer_graph
from sieve_replay.timing import AnalyticTimingModel, CXLReadConfig
from sieve_replay.trace import RouterTraceRecord, TraceBatch, load_trace_set
from sieve_replay.trace_manifest import TraceManifest


PRESSURE_MATRIX: tuple[tuple[str, int, int], ...] = (
    ("b8_c32k", 8, 32768),
    ("b16_c16k", 16, 16384),
    ("b16_c32k", 16, 32768),
    ("b32_c16k", 32, 16384),
)
DOMAINS: tuple[tuple[str, float], ...] = (("a800", 80.0), ("simulated", 96.0))
CXL_BUDGET_GB: tuple[int, ...] = (0, 16, 32, 64)
# Architecture sensitivity uses one GPU baseline and the analytic Sieve policy.
# The full seven-policy cycle-v1 comparison remains frozen in prior results.
POLICIES: tuple[str, ...] = ("gpu-only", "sieve")
MODES: tuple[str, ...] = ("local-only-serial", "cxl-memory-only-serial", "cxl-memory-only-overlap")
CXL_PROFILES: dict[str, dict[str, float | str]] = {
    "conservative-assumption": {
        "bandwidth_gb_s": 32.0,
        "latency_us": 0.5,
        "source": "unbound sensitivity assumption; no target device selected",
    },
    "nominal-assumption": {
        "bandwidth_gb_s": 64.0,
        "latency_us": 0.25,
        "source": "unbound sensitivity assumption; no target device selected",
    },
    "optimistic-assumption": {
        "bandwidth_gb_s": 128.0,
        "latency_us": 0.1,
        "source": "unbound sensitivity assumption; no target device selected",
    },
}
SOURCE_EXPERIMENT = Path("configs/experiments/full_decode_real_pilot_cycle_v0.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _template(source: TraceBatch, batch_size: int, context_length: int) -> TraceBatch:
    records = tuple(
        RouterTraceRecord(
            step=0,
            layer=source.layer,
            token_id=f"kv-cxl-stage3-{batch_size}-{context_length}-{index}",
            context_length=context_length,
            expert_ids=source.records[index % source.batch_size].expert_ids,
            expert_weights=source.records[index % source.batch_size].expert_weights,
        )
        for index in range(batch_size)
    )
    return TraceBatch(step=0, layer=source.layer, records=records)


def _memory_row(configuration: Any, template: TraceBatch, local_capacity_bytes: int, cxl_capacity_bytes: int) -> dict[str, Any]:
    memory = estimate_memory_footprint(configuration.model, template)
    admission = classify_capacity(memory.total_bytes, local_capacity_bytes, cxl_capacity_bytes)
    return {
        "kv_cache_bytes": memory.kv_cache_bytes,
        "peak_memory_bytes": memory.total_bytes,
        "model_weights_bytes": memory.model_weights_bytes,
        "activation_bytes": memory.activation_bytes,
        "local_capacity_bytes": local_capacity_bytes,
        "cxl_capacity_bytes": cxl_capacity_bytes,
        "capacity_state": admission["state"],
        "feasible": admission["feasible"],
        "spill_bytes": admission["spill_bytes"],
        "unallocated_bytes": admission["unallocated_bytes"],
    }


def _timing_row(
    configuration: Any,
    template: TraceBatch,
    policy_name: str,
    mode: str,
    cxl_config: CXLReadConfig,
    profile: str,
    domain: str,
    budget_gb: int,
) -> dict[str, Any]:
    memory_row = _memory_row(
        configuration,
        template,
        cxl_config.local_capacity_bytes,
        cxl_config.capacity_bytes,
    )
    row: dict[str, Any] = {
        "case": f"b{template.batch_size}_c{template.context_lengths[0] // 1024}k",
        "batch_size": template.batch_size,
        "context_length": template.context_lengths[0],
        "domain": domain,
        "cxl_budget_gb": budget_gb,
        "cxl_profile": profile,
        "mode": mode,
        "policy": policy_name,
        "layers": configuration.model.num_hidden_layers,
        "decode_steps": 1,
        "timing_classification": "analytic-control-event-graph; CXL memory-only assumptions; no Ramulator or A800 timing",
        **memory_row,
        "cxl_bandwidth_gb_s": cxl_config.bandwidth_bytes_per_second / 1e9,
        "cxl_latency_us": cxl_config.latency_us,
    }
    if not memory_row["feasible"]:
        row.update(
            {
                "replayed": False,
                "total_latency_us": None,
                "throughput_request_tokens_per_s": None,
                "local_kv_read_bytes": None,
                "cxl_kv_read_bytes": None,
                "cxl_kv_read_latency_us": None,
                "cxl_kv_read_critical_path_us": None,
                "attention_critical_path_us": None,
                "expert_critical_path_us": None,
            }
        )
        return row

    effective_cxl = cxl_config if mode != "local-only-serial" else CXLReadConfig()
    kv_mode = "serial-local-hbm-v1" if mode != "cxl-memory-only-overlap" else "overlap-local-hbm-v1"
    timing = AnalyticTimingModel(configuration.model, configuration.hardware, cxl_config=effective_cxl)
    policy = create_policy(policy_name, timing)
    events = []
    previous_tail: str | None = None
    traces: list[TraceBatch] = []
    for layer in range(configuration.model.num_hidden_layers):
        trace = TraceBatch(step=0, layer=layer, records=template.records)
        decision = policy.place(trace)
        layer_events, previous_tail = build_decode_layer_graph(
            trace,
            decision,
            timing,
            previous_tail,
            kv_mode,
        )
        events.extend(layer_events)
        traces.append(trace)
    scheduled = EventEngine().run(tuple(events))
    total = max((event.end_us for event in scheduled), default=0.0)
    duration_by_category: dict[str, float] = {}
    for event in scheduled:
        duration_by_category[event.event.category] = duration_by_category.get(event.event.category, 0.0) + event.event.duration_us
    critical = EventEngine.critical_path(scheduled)
    critical_set = set(critical)
    critical_by_category: dict[str, float] = {}
    for event in scheduled:
        if event.event.name in critical_set:
            category = event.event.category
            critical_by_category[category] = critical_by_category.get(category, 0.0) + event.event.duration_us
    cxl_bytes = sum(
        event.event.duration_us * 0.0 + event.event.duration_us
        for event in scheduled
        if event.event.category == "cxl_kv_read"
    )
    local_bytes = sum(
        event.event.duration_us * 0.0 + event.event.duration_us
        for event in scheduled
        if event.event.category == "kv_read"
    )
    layer_kv_bytes = (
        2
        * sum(template.context_lengths)
        * configuration.model.num_key_value_heads
        * configuration.model.head_dim
        * configuration.model.dtype_bytes
    )
    total_kv_bytes = layer_kv_bytes * configuration.model.num_hidden_layers
    spill_fraction = (
        float(memory_row["spill_bytes"]) / float(memory_row["kv_cache_bytes"])
        if memory_row["kv_cache_bytes"]
        else 0.0
    )
    row.update(
        {
            "replayed": True,
            "total_latency_us": total,
            "throughput_request_tokens_per_s": template.batch_size * 1_000_000.0 / total if total else 0.0,
            "local_kv_read_bytes": total_kv_bytes * (1.0 - spill_fraction),
            "cxl_kv_read_bytes": total_kv_bytes * spill_fraction,
            "cxl_kv_read_latency_us": duration_by_category.get("cxl_kv_read", 0.0),
            "cxl_kv_read_critical_path_us": critical_by_category.get("cxl_kv_read", 0.0),
            "attention_critical_path_us": critical_by_category.get("attention", 0.0),
            "expert_critical_path_us": sum(
                critical_by_category.get(category, 0.0)
                for category in ("gpu_weight_load", "gpu_expert", "pim_write", "pim_expert", "pim_read")
            ),
            "event_duration_kv_read_us": local_bytes,
            "event_duration_cxl_kv_read_us": cxl_bytes,
            "critical_path": ";".join(critical),
        }
    )
    return row


def _write_report(output: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output / "comparison.json").write_text(
        json.dumps({"metadata": metadata, "rows": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# KV CXL timing stage 3",
        "",
        "本阶段是受控解析事件图实验：将 CXL 建模为独立 memory-only KV READ 队列，",
        "使用 manifest 校验的旧 pilot layer-batch 作为路由模板，扩展到四个压力点并重复",
        "完整 48 层、1 个 Decode step。CXL profile 是明确的敏感性假设，不是目标设备实测。",
        "",
        "| domain | case | policy | mode | CXL budget (GB) | profile | state | total latency (us) | CXL READ (GB) | CXL critical path (us) |",
        "|---|---|---|---|---:|---|---|---:|---:|---:|",
    ]
    for row in rows:
        if row["replayed"]:
            latency = f"{float(row['total_latency_us']):.6f}"
            cxl_bytes = f"{float(row['cxl_kv_read_bytes']) / 1e9:.6f}"
            cxl_path = f"{float(row['cxl_kv_read_critical_path_us']):.6f}"
        else:
            latency = cxl_bytes = cxl_path = "NA"
        lines.append(
            f"| {row['domain']} | {row['case']} | {row['policy']} | {row['mode']} | {row['cxl_budget_gb']} | {row['cxl_profile']} | {row['capacity_state']} | {latency} | {cxl_bytes} | {cxl_path} |"
        )
    lines += [
        "",
        "`oom` 行没有被回放，因而不包含可实现的吞吐量。`spill` 只表示容量准入后有",
        "静态 KV 字节放到 CXL；本阶段没有分页、eviction、恢复、CXL-PIM 或真实链路观测。",
        "`event_duration_cxl_kv_read_us` 是 CXL READ 事件时长总和，`cxl_kv_read_critical_path_us`",
        "只统计最终关键路径上的 CXL READ。",
        "",
        "路由模板来自已校验的真实 pilot，但本阶段不是新的 A800 捕获，也不是 Ramulator timing。",
        "",
    ]
    (output / "stage3_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_stage3(
    experiment_path: str | Path,
    output_dir: str | Path,
    *,
    pressure_matrix: Iterable[tuple[str, int, int]] = PRESSURE_MATRIX,
    domains: Iterable[tuple[str, float]] = DOMAINS,
    budgets: Iterable[int] = CXL_BUDGET_GB,
    profiles: dict[str, dict[str, float | str]] = CXL_PROFILES,
    policies: Iterable[str] = POLICIES,
    modes: Iterable[str] = MODES,
) -> dict[str, Any]:
    pressure_matrix = tuple(pressure_matrix)
    domains = tuple(domains)
    budgets = tuple(int(value) for value in budgets)
    profiles = dict(profiles)
    policies = tuple(policies)
    modes = tuple(modes)
    if not pressure_matrix or not domains or not budgets or not policies or not modes:
        raise ValueError("stage-3 inputs must contain at least one case, domain, budget, policy and mode")
    if any(value < 0 for value in budgets) or len(set(budgets)) != len(budgets):
        raise ValueError("stage-3 CXL budgets must be unique nonnegative integers")
    configuration = load_configuration(experiment_path)
    trace_set = load_trace_set(
        configuration.experiment.trace_path,
        configuration.model,
        configuration.experiment.layers,
        configuration.experiment.steps,
    )
    manifest_path = configuration.experiment.trace_manifest_path
    if manifest_path is None:
        raise ValueError("stage-3 timing requires a manifest-bound source trace")
    manifest = TraceManifest.load_and_validate(
        manifest_path,
        configuration.experiment.trace_path,
        configuration.model,
        trace_set,
    )
    if manifest.raw["schema_version"] != 2:
        raise ValueError("stage-3 timing requires schema-v2 prompt-bound provenance")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    source = trace_set.batches[0]
    for case, batch_size, context_length in pressure_matrix:
        template = _template(source, batch_size, context_length)
        for domain, local_gb in domains:
            local_bytes = int(local_gb * 1_000_000_000)
            for budget_gb in budgets:
                cxl_bytes = int(budget_gb * 1_000_000_000)
                memory = _memory_row(configuration, template, local_bytes, cxl_bytes)
                for policy_name in policies:
                    if "local-only-serial" in modes and budget_gb == budgets[0]:
                        row = _timing_row(
                            configuration,
                            template,
                            policy_name,
                            "local-only-serial",
                            CXLReadConfig(
                                mode="disabled",
                                local_capacity_bytes=local_bytes,
                                capacity_bytes=0,
                            ),
                            "none",
                            domain,
                            0,
                        )
                        row["case"] = case
                        row["local_capacity_bytes"] = local_bytes
                        row["cxl_capacity_bytes"] = 0
                        row["capacity_state"] = memory["capacity_state"]
                        row["feasible"] = memory["capacity_state"] != "oom"
                        row["spill_bytes"] = memory["spill_bytes"]
                        row["unallocated_bytes"] = memory["unallocated_bytes"]
                        if memory["capacity_state"] == "oom":
                            row["replayed"] = False
                            row["total_latency_us"] = None
                            row["throughput_request_tokens_per_s"] = None
                        rows.append(row)
                    for profile_name, profile in profiles.items():
                        if "cxl-memory-only-serial" in modes:
                            cxl_mode = "memory-only-v1"
                            config = CXLReadConfig(
                                mode=cxl_mode,
                                local_capacity_bytes=local_bytes,
                                capacity_bytes=cxl_bytes,
                                bandwidth_bytes_per_second=float(profile["bandwidth_gb_s"]) * 1e9,
                                latency_us=float(profile["latency_us"]),
                            )
                            row = _timing_row(configuration, template, policy_name, "cxl-memory-only-serial", config, profile_name, domain, budget_gb)
                            row["case"] = case
                            rows.append(row)
                        if "cxl-memory-only-overlap" in modes:
                            config = CXLReadConfig(
                                mode="memory-only-v1",
                                local_capacity_bytes=local_bytes,
                                capacity_bytes=cxl_bytes,
                                bandwidth_bytes_per_second=float(profile["bandwidth_gb_s"]) * 1e9,
                                latency_us=float(profile["latency_us"]),
                            )
                            row = _timing_row(configuration, template, policy_name, "cxl-memory-only-overlap", config, profile_name, domain, budget_gb)
                            row["case"] = case
                            rows.append(row)

    input_paths = {
        "experiment": Path(experiment_path).resolve(),
        "trace": configuration.experiment.trace_path,
        "trace_manifest": manifest_path,
        "prompts": manifest_path.parent / manifest.raw["prompts"]["file"],
        "model": configuration.experiment.model_path,
        "hardware": configuration.experiment.hardware_path,
    }
    metadata: dict[str, Any] = {
        "method": "analytic-cxl-memory-only-timing-sensitivity-v1",
        "result_classification": "解析控制事件图 + CXL memory-only 参数敏感性；不是Ramulator或A800 timing",
        "source_manifest_validated": True,
        "source_trace_sha256": _sha256(configuration.experiment.trace_path),
        "routing_template": {
            "source_step": source.step,
            "source_layer": source.layer,
            "source_batch_size": source.batch_size,
            "context_assumption": "uniform context; expert routes tiled from the source pilot batch",
        },
        "pressure_matrix": [
            {"case": case, "batch_size": batch, "context_length": context}
            for case, batch, context in pressure_matrix
        ],
        "domains": [{"name": name, "local_capacity_gb": capacity} for name, capacity in domains],
        "cxl_budget_gb": list(budgets),
        "profiles": profiles,
        "policies": list(policies),
        "modes": list(modes),
        "limitations": [
            "Routing is a controlled template, not a new A800 capture.",
            "CXL bandwidth and latency profiles are explicit assumptions with no target device binding.",
            "No CXL link protocol, paging, eviction, recovery, CXL-PIM, or Ramulator CXL model.",
            "Spill bytes are uniformly distributed across model KV layers for this sensitivity study.",
            "OOM rows are not replayed and have no realizable throughput.",
        ],
        "input_hashes": {name: _sha256(path) for name, path in input_paths.items()},
        "rows": len(rows),
    }
    _write_report(output, rows, metadata)
    metadata["output_hashes"] = {
        "comparison.csv": sha256_file(output / "comparison.csv"),
        "comparison.json": sha256_file(output / "comparison.json"),
        "stage3_report.md": sha256_file(output / "stage3_report.md"),
    }
    # Rewrite comparison.json after adding output hashes, then bind all artifacts.
    (output / "comparison.json").write_text(
        json.dumps({"metadata": metadata, "rows": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stage_manifest = {
        "schema_version": 1,
        "method": metadata["method"],
        "input_hashes": metadata["input_hashes"],
        "output_hashes": {
            "comparison.csv": sha256_file(output / "comparison.csv"),
            "comparison.json": sha256_file(output / "comparison.json"),
            "stage3_report.md": sha256_file(output / "stage3_report.md"),
        },
        "rows": len(rows),
        "replayed_rows": sum(bool(row["replayed"]) for row in rows),
        "oom_rows": sum(row["capacity_state"] == "oom" for row in rows),
    }
    (output / "stage_manifest.json").write_text(
        json.dumps(stage_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"metadata": metadata, "rows": rows, "stage_manifest": stage_manifest}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=str(SOURCE_EXPERIMENT))
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = run_stage3(args.experiment, args.output)
    print(json.dumps(result["stage_manifest"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
