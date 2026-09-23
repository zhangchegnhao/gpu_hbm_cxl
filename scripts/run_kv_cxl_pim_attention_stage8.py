#!/usr/bin/env python3
"""Run the first isolated CXL-PIM Attention comparison for B8/C32k."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_kv_cxl_decode_stage5 as stage5
from sieve_replay.ramulator.cxl_kv_workload import (
    CXLKVWorkloadResult,
    cxl_kv_cache_context,
    run_cxl_kv_workload,
)
from sieve_replay.ramulator.cxl_pim_attention import (
    CXLPIMAttentionShape,
    CXLPIMTopology,
    build_cxl_pim_attention_shape,
    cxl_pim_attention_context,
    run_cxl_pim_operation,
)
from sieve_replay.ramulator.microbenchmark import MicrobenchmarkResult
from sieve_replay.ramulator.mixed_workload import SieveCycleV1Config


STAGE7 = ROOT / "results/kv_cxl_a800_bridge_stage7_v1/comparison.json"
STAGE7_VALIDATION = ROOT / "results/kv_cxl_a800_bridge_stage7_v1/validation.json"
DEFAULT_OUTPUT = ROOT / "results/kv_cxl_pim_attention_stage8_v1"
CASE = "b8_c32k"
BATCH_SIZE = 8
CONTEXT_LENGTH = 32768
LAYERS = 48
TOPOLOGY = CXLPIMTopology()
PIM_OPERATIONS = (
    ("query-gwrite", "PIM_GWRITE", "pim_gwrite_waves"),
    ("attention-mac", "PIM_MAC", "pim_mac_waves"),
    ("partial-read", "PIM_READ", "pim_read_waves"),
)


def _key(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_stage7() -> dict[str, Any]:
    validation = json.loads(STAGE7_VALIDATION.read_text(encoding="utf-8"))
    if validation.get("status") != "passed":
        raise ValueError("Stage 8 requires a passed Stage-7 validation")
    raw = json.loads(STAGE7.read_text(encoding="utf-8"))
    if raw["metadata"]["case"] != CASE or raw["metadata"]["interpolation"] != 0:
        raise ValueError("Stage-7 source is not the exact B8/C32k bridge")
    names = {row["scenario"] for row in raw["rows"]}
    if names != {
        "resident-local-counterfactual",
        "nominal-memory-only-cxl-spill",
    }:
        raise ValueError("Stage-7 scenario matrix is incomplete")
    return raw


def _cache_input(
    kind: str,
    payload: dict[str, Any],
    shape: CXLPIMAttentionShape,
    cycle: SieveCycleV1Config,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": "kv-cxl-pim-attention-stage8-v1",
        "case": CASE,
        "kind": kind,
        "payload": payload,
        "topology": asdict(TOPOLOGY),
        "profile": {"name": stage5.PROFILE_NAME, **stage5.PROFILE},
        "shape": asdict(shape),
        "cxl_context": cxl_kv_cache_context(ROOT, stage5.CYCLE),
        "cxl_pim_context": cxl_pim_attention_context(ROOT, stage5.CYCLE),
        "cycle_tick_ps": cycle.expected_tick_ps,
        "input_hashes": {
            "stage7": stage5._sha256(STAGE7),
            "stage7_validation": stage5._sha256(STAGE7_VALIDATION),
            "model": stage5._sha256(ROOT / "configs/models/qwen3_30b_a3b.json"),
            "cycle": stage5._sha256(stage5.CYCLE),
        },
    }


def _run_or_load(
    output: Path,
    kind: str,
    payload: dict[str, Any],
    shape: CXLPIMAttentionShape,
    cycle: SieveCycleV1Config,
    transfer_runner: Callable[..., CXLKVWorkloadResult],
    pim_runner: Callable[..., MicrobenchmarkResult],
) -> tuple[str, Path, CXLKVWorkloadResult | MicrobenchmarkResult, bool]:
    cache_input = _cache_input(kind, payload, shape, cycle)
    key = _key(cache_input)
    cache_dir = output / "exact_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    if path.is_file():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("cache_input") != cache_input:
            raise ValueError(f"Stage-8 cache provenance mismatch: {path}")
        result_type = cached["result_type"]
        raw_result = dict(cached["result"])
        if result_type == "cxl-transfer":
            result = CXLKVWorkloadResult(**raw_result)
        else:
            raw_result["duration_us_by_waves"] = {
                int(waves): duration
                for waves, duration in raw_result["duration_us_by_waves"].items()
            }
            raw_result["cycles_by_waves"] = {
                int(waves): cycles
                for waves, cycles in raw_result["cycles_by_waves"].items()
            }
            result = MicrobenchmarkResult(**raw_result)
        return key, path, result, False

    if kind in {"query-link", "partial-result-link"}:
        transactions = int(payload["transactions"])
        result = transfer_runner(
            stage5.RAMULATOR,
            cycle,
            gpu_read_transactions=0,
            local_kv_read_transactions=0,
            cxl_kv_read_transactions=transactions,
            pim_gwrite_waves=0,
            pim_mac_waves=0,
            pim_read_waves=0,
            cxl_bandwidth_bytes_per_second=float(stage5.PROFILE["bandwidth_gb_s"])
            * 1e9,
            cxl_latency_us=float(stage5.PROFILE["latency_us"]),
            cxl_channels=int(stage5.PROFILE["channel_count"]),
        )
        result_type = "cxl-transfer"
    else:
        result = pim_runner(
            stage5.RAMULATOR,
            cycle,
            TOPOLOGY,
            str(payload["operation"]),
            int(payload["waves"]),
        )
        result_type = "pim-operation"
    path.write_text(
        json.dumps(
            {
                "cache_input": cache_input,
                "result_type": result_type,
                "result": asdict(result),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return key, path, result, True


def _transfer_us(result: CXLKVWorkloadResult) -> float:
    return result.cxl_kv_completion_cycles * result.tick_ps / 1_000_000.0


def _pim_us(result: MicrobenchmarkResult) -> float:
    return result.duration_us_by_waves[result.max_waves]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_report(output: Path, rows: list[dict[str, Any]], derived: dict[str, Any]) -> None:
    lines = [
        "# KV CXL-PIM Attention Stage 8",
        "",
        "Stage 8 is an optimistic, decomposed CXL-PIM Attention model. Query transfer, PIM GWRITE/MAC/READ, and partial-result transfer are five separate exact Ramulator runs connected in dependency order. The local GPU Attention branch and CXL-PIM branch overlap ideally.",
        "",
        "| scenario | estimated Decode ms | throughput tok/s | change vs memory-only CXL |",
        "|---|---:|---:|---:|",
    ]
    memory_only = next(row for row in rows if row["scenario"] == "nominal-memory-only-cxl-spill")
    for row in rows:
        change = row["estimated_throughput_tokens_per_s"] / memory_only["estimated_throughput_tokens_per_s"] - 1.0
        lines.append(
            f"| {row['scenario']} | {row['estimated_decode_ms']:.3f} | "
            f"{row['estimated_throughput_tokens_per_s']:.3f} | {100.0 * change:.3f}% |"
        )
    lines += [
        "",
        f"CXL-PIM transfers {derived['cxl_pim_link_bytes_per_layer']:,} bytes per layer versus {derived['memory_only_cxl_link_bytes_per_layer']:,} raw spilled-KV bytes, a {100.0 * derived['link_traffic_reduction_fraction']:.3f}% reduction.",
        f"Its exact decomposed branch takes {derived['cxl_pim_branch_us_per_layer']:.3f} us per layer; the A800-fit local GPU branch remains critical in the ideal-overlap model.",
        f"Ideal overlap and serialized branches bound estimated Decode at {derived['ideal_overlap_decode_ms']:.3f}--{derived['serialized_decode_ms']:.3f} ms under the remaining model assumptions.",
        "",
        "This is not a physical CXL-PIM result. The four-channel topology, PIM throughput, FP32 partial-state format, ideal branch overlap, and zero merge cost are assumptions. Softmax scalar operations, protocol overhead, paging, eviction, energy, multi-GPU sharing, and contention between the five isolated phases are absent.",
        "",
    ]
    (output / "stage8_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_stage8(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    transfer_runner: Callable[..., CXLKVWorkloadResult] = run_cxl_kv_workload,
    pim_runner: Callable[..., MicrobenchmarkResult] = run_cxl_pim_operation,
) -> dict[str, Any]:
    output = Path(output_dir)
    if output.exists():
        existing = {item.name for item in output.iterdir()}
        if existing and not existing.issubset({"exact_cache"}):
            raise ValueError(f"Stage-8 output already exists and is not resumable: {output}")
    output.mkdir(parents=True, exist_ok=True)

    stage7 = _load_stage7()
    configuration, _ = stage5._source_configuration()
    cycle = SieveCycleV1Config.load(stage5.CYCLE)
    spilled_bytes = int(stage7["derived"]["cxl_payload_bytes_per_layer"])
    shape = build_cxl_pim_attention_shape(
        configuration.model,
        cycle,
        batch_size=BATCH_SIZE,
        spilled_kv_bytes_per_layer=spilled_bytes,
        topology=TOPOLOGY,
    )
    plan: list[tuple[str, dict[str, Any]]] = [
        ("query-link", {"transactions": shape.query_link_transactions, "direction": "gpu-to-cxl-pim-write-proxied-by-fifo-read"}),
        *[
            (name, {"operation": operation, "waves": int(getattr(shape, field))})
            for name, operation, field in PIM_OPERATIONS
        ],
        ("partial-result-link", {"transactions": shape.result_link_transactions, "direction": "cxl-pim-to-gpu-read"}),
    ]
    results: dict[str, CXLKVWorkloadResult | MicrobenchmarkResult] = {}
    cache_records: list[dict[str, Any]] = []
    new_runs = 0
    for kind, payload in plan:
        key, path, result, executed = _run_or_load(
            output,
            kind,
            payload,
            shape,
            cycle,
            transfer_runner,
            pim_runner,
        )
        results[kind] = result
        new_runs += int(executed)
        cache_records.append(
            {
                "kind": kind,
                "key": key,
                "sha256": stage5._sha256(path),
                "payload": payload,
                "executed": executed,
            }
        )

    for kind, transactions in (
        ("query-link", shape.query_link_transactions),
        ("partial-result-link", shape.result_link_transactions),
    ):
        result = results[kind]
        if not isinstance(result, CXLKVWorkloadResult) or result.cxl_kv_completed_requests != transactions:
            raise ValueError(f"Stage-8 incomplete transfer: {kind}")
    for name, _, field in PIM_OPERATIONS:
        result = results[name]
        waves = int(getattr(shape, field))
        expected = waves * TOPOLOGY.total_pseudo_channels
        if not isinstance(result, MicrobenchmarkResult) or result.completed_requests != expected:
            raise ValueError(f"Stage-8 incomplete PIM operation: {name}")

    component_us = {
        "query_link_us": _transfer_us(results["query-link"]),
        "pim_gwrite_us": _pim_us(results["query-gwrite"]),
        "pim_mac_us": _pim_us(results["attention-mac"]),
        "pim_read_us": _pim_us(results["partial-read"]),
        "result_link_us": _transfer_us(results["partial-result-link"]),
    }
    branch_us = sum(component_us.values())
    spill_fraction = float(stage7["derived"]["spill_fraction_of_kv_bytes"])
    calibration = stage7["calibration"]
    local_context_equivalent = CONTEXT_LENGTH * (1.0 - spill_fraction)
    local_gpu_attention_ms = (
        calibration["fit"]["intercept_ms"]
        + calibration["fit"]["slope_ms_per_context_token"]
        * local_context_equivalent
    )
    cxl_pim_branch_ms = branch_us * LAYERS / 1000.0
    cxl_pim_attention_ms = max(local_gpu_attention_ms, cxl_pim_branch_ms)
    serialized_attention_ms = local_gpu_attention_ms + cxl_pim_branch_ms
    non_attention_ms = calibration["target"]["non_attention_mean_ms"]
    cxl_pim_decode_ms = cxl_pim_attention_ms + non_attention_ms
    serialized_decode_ms = serialized_attention_ms + non_attention_ms

    prior_rows = {row["scenario"]: dict(row) for row in stage7["rows"]}
    resident = prior_rows["resident-local-counterfactual"]
    memory_only = prior_rows["nominal-memory-only-cxl-spill"]
    cxl_pim_row = {
        "scenario": "optimistic-cxl-pim-attention",
        "evidence_class": "A800 local-branch extrapolation plus five isolated exact Ramulator phase results",
        "capacity_state": "spill",
        "estimated_attention_ms": cxl_pim_attention_ms,
        "estimated_decode_ms": cxl_pim_decode_ms,
        "estimated_throughput_tokens_per_s": BATCH_SIZE * 1000.0 / cxl_pim_decode_ms,
        "memory_completion_us_per_layer": branch_us,
        "memory_delta_vs_resident_ms_per_decode": cxl_pim_decode_ms
        - float(resident["estimated_decode_ms"]),
        "cxl_bandwidth_gb_s": float(stage5.PROFILE["bandwidth_gb_s"]),
        "cxl_latency_us": float(stage5.PROFILE["latency_us"]),
        "local_kv_read_transactions": int(memory_only["local_kv_read_transactions"]),
        "cxl_kv_read_transactions": 0,
    }
    serialized_row = {
        **cxl_pim_row,
        "scenario": "serialized-cxl-pim-attention",
        "evidence_class": "A800 local-branch extrapolation plus five isolated exact Ramulator phases serialized as a scheduling bound",
        "estimated_attention_ms": serialized_attention_ms,
        "estimated_decode_ms": serialized_decode_ms,
        "estimated_throughput_tokens_per_s": BATCH_SIZE * 1000.0
        / serialized_decode_ms,
        "memory_delta_vs_resident_ms_per_decode": serialized_decode_ms
        - float(resident["estimated_decode_ms"]),
    }
    rows = [resident, memory_only, cxl_pim_row, serialized_row]
    link_bytes = (shape.query_link_transactions + shape.result_link_transactions) * cycle.transaction_bytes
    derived = {
        **component_us,
        "cxl_pim_branch_us_per_layer": branch_us,
        "cxl_pim_branch_ms_per_decode": cxl_pim_branch_ms,
        "local_gpu_context_equivalent": local_context_equivalent,
        "local_gpu_attention_ms": local_gpu_attention_ms,
        "critical_attention_branch": "local-gpu" if local_gpu_attention_ms >= cxl_pim_branch_ms else "cxl-pim",
        "ideal_overlap_decode_ms": cxl_pim_decode_ms,
        "serialized_decode_ms": serialized_decode_ms,
        "memory_only_cxl_link_bytes_per_layer": spilled_bytes,
        "cxl_pim_link_bytes_per_layer": link_bytes,
        "link_traffic_reduction_fraction": 1.0 - link_bytes / spilled_bytes,
        "throughput_change_vs_memory_only_fraction": cxl_pim_row["estimated_throughput_tokens_per_s"]
        / float(memory_only["estimated_throughput_tokens_per_s"])
        - 1.0,
        "throughput_change_vs_resident_fraction": cxl_pim_row["estimated_throughput_tokens_per_s"]
        / float(resident["estimated_throughput_tokens_per_s"])
        - 1.0,
        "serialized_throughput_change_vs_memory_only_fraction": serialized_row["estimated_throughput_tokens_per_s"]
        / float(memory_only["estimated_throughput_tokens_per_s"])
        - 1.0,
    }
    _write_csv(output / "comparison.csv", rows)
    _write_report(output, rows, derived)
    metadata = {
        "schema_version": 1,
        "stage": "kv-cxl-pim-attention-stage8-v1",
        "case": CASE,
        "method": "decomposed isolated request-level CXL-PIM Attention lower bound",
        "classification": "A800-anchored analytic sensitivity plus exact isolated Ramulator phases",
        "exact_runs": len(plan),
        "new_runs_executed": new_runs,
        "interpolation": 0,
        "layers": LAYERS,
        "topology_assumption": asdict(TOPOLOGY),
        "shape": asdict(shape),
        "phase_order": [kind for kind, _ in plan],
        "component_us_per_layer": component_us,
        "cache_records": cache_records,
        "execution": stage5._execution_provenance(),
        "input_hashes": {
            "stage7": stage5._sha256(STAGE7),
            "stage7_validation": stage5._sha256(STAGE7_VALIDATION),
            "model": stage5._sha256(configuration.experiment.model_path),
            "cycle": stage5._sha256(stage5.CYCLE),
        },
        "limitations": [
            "CXL-PIM topology and PIM timing are model assumptions, not a measured device.",
            "Query writes are represented by the existing FIFO link READ proxy with identical byte service.",
            "The five phases are isolated exact runs connected serially; their controllers do not contend in one Ramulator simulation.",
            "The comparison reports ideal overlap and full serialization of the GPU/CXL-PIM branches; partial-result merge cost is zero in both bounds.",
            "Softmax scalar operations, protocol overhead, paging, eviction, energy, multi-GPU sharing, and physical CXL are absent.",
        ],
    }
    comparison_path = output / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {"metadata": metadata, "derived": derived, "rows": rows},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "stage": metadata["stage"],
        "case": CASE,
        "rows": len(rows),
        "exact_runs": len(plan),
        "new_runs_executed": new_runs,
        "interpolation": 0,
        "execution": metadata["execution"],
        "input_hashes": metadata["input_hashes"],
        "cache_records": cache_records,
        "output_hashes": {
            "comparison.csv": stage5._sha256(output / "comparison.csv"),
            "comparison.json": stage5._sha256(comparison_path),
            "stage8_report.md": stage5._sha256(output / "stage8_report.md"),
        },
    }
    (output / "stage_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"metadata": metadata, "derived": derived, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    result = run_stage8(args.output)
    print(json.dumps({"metadata": result["metadata"], "derived": result["derived"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
