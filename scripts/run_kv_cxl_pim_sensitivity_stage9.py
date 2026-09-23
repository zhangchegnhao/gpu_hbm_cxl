#!/usr/bin/env python3
"""Run Stage-9 CXL-PIM topology, compute, precision and scheduling sensitivity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from dataclasses import asdict, replace
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
    CXLPIMTopology,
    build_cxl_pim_attention_shape,
    cxl_pim_attention_context,
    run_cxl_pim_operation,
)
from sieve_replay.ramulator.microbenchmark import MicrobenchmarkResult
from sieve_replay.ramulator.mixed_workload import SieveCycleV1Config


STAGE8 = ROOT / "results/kv_cxl_pim_attention_stage8_v1/comparison.json"
STAGE8_VALIDATION = ROOT / "results/kv_cxl_pim_attention_stage8_v1/validation.json"
DEFAULT_OUTPUT = ROOT / "results/kv_cxl_pim_sensitivity_stage9_v1"
CHANNELS = (2, 4, 8)
MAC_INTERVALS_PS = (12_288, 24_576, 49_152)
PARTIAL_SCALAR_BYTES = (2, 4)
OVERLAP_FRACTIONS = (0.0, 0.5, 1.0)
MERGE_US_PER_LAYER = (0.0, 5.0, 20.0)
PSEUDO_CHANNELS_PER_CHANNEL = 2
BANKS_PER_PSEUDO_CHANNEL = 24
BATCH_SIZE = 8
LAYERS = 48


def _key(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_stage8() -> dict[str, Any]:
    validation = json.loads(STAGE8_VALIDATION.read_text(encoding="utf-8"))
    if validation.get("status") != "passed":
        raise ValueError("Stage 9 requires a passed Stage-8 validation")
    raw = json.loads(STAGE8.read_text(encoding="utf-8"))
    metadata = raw["metadata"]
    if metadata["interpolation"] != 0 or metadata["exact_runs"] != 5:
        raise ValueError("Stage-8 source is not the validated exact result")
    return raw


def _identity_context() -> dict[str, Any]:
    return {
        "cxl": cxl_kv_cache_context(ROOT, stage5.CYCLE),
        "cxl_pim": cxl_pim_attention_context(ROOT, stage5.CYCLE),
        "profile": {
            "bandwidth_gb_s": stage5.PROFILE["bandwidth_gb_s"],
            "latency_us": stage5.PROFILE["latency_us"],
            "channel_count": stage5.PROFILE["channel_count"],
        },
    }


def _transfer_identity(transactions: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "cxl-fifo-transfer-read-proxy",
        "transactions": transactions,
        "context": _identity_context(),
    }


def _pim_identity(
    operation: str,
    waves: int,
    topology: CXLPIMTopology,
    cycle: SieveCycleV1Config,
) -> dict[str, Any]:
    if operation == "PIM_MAC":
        interval_name = "pim_mac_interval_ps"
        interval_ps = cycle.pim_mac_interval_ps
    else:
        interval_name = "pim_io_interval_ps"
        interval_ps = cycle.pim_io_interval_ps
    return {
        "schema_version": 1,
        "kind": "isolated-cxl-pim-operation",
        "operation": operation,
        "waves": waves,
        "topology": {
            "channels": topology.channels,
            "pseudo_channels_per_channel": topology.pseudo_channels_per_channel,
            "banks_per_pseudo_channel": topology.banks_per_pseudo_channel,
        },
        "timing": {
            interval_name: interval_ps,
            "row_span_waves": cycle.row_span_waves,
            "tick_ps": cycle.expected_tick_ps,
        },
        "context": _identity_context(),
    }


def _stage8_cache_index(
    stage8: dict[str, Any], cycle: SieveCycleV1Config
) -> dict[str, dict[str, Any]]:
    metadata = stage8["metadata"]
    topology = CXLPIMTopology(**metadata["topology_assumption"])
    records: dict[str, dict[str, Any]] = {}
    for record in metadata["cache_records"]:
        kind = record["kind"]
        payload = record["payload"]
        if kind in {"query-link", "partial-result-link"}:
            identity = _transfer_identity(int(payload["transactions"]))
        else:
            identity = _pim_identity(
                str(payload["operation"]),
                int(payload["waves"]),
                topology,
                cycle,
            )
        path = STAGE8.parent / "exact_cache" / f"{record['key']}.json"
        if stage5._sha256(path) != record["sha256"]:
            raise ValueError(f"Stage-8 cache hash mismatch: {kind}")
        records[_key(identity)] = {
            "path": path,
            "sha256": record["sha256"],
            "source": "stage8-reuse",
        }
    if len(records) != 5:
        raise ValueError("Stage-8 cache identity matrix is not unique")
    return records


def _deserialize(path: Path) -> CXLKVWorkloadResult | MicrobenchmarkResult:
    cached = json.loads(path.read_text(encoding="utf-8"))
    raw = dict(cached["result"])
    if cached["result_type"] == "cxl-transfer":
        return CXLKVWorkloadResult(**raw)
    raw["duration_us_by_waves"] = {
        int(key): value for key, value in raw["duration_us_by_waves"].items()
    }
    raw["cycles_by_waves"] = {
        int(key): value for key, value in raw["cycles_by_waves"].items()
    }
    return MicrobenchmarkResult(**raw)


def _designs(model: Any, cycle: SieveCycleV1Config, spilled_bytes: int) -> list[dict[str, Any]]:
    designs: list[dict[str, Any]] = []
    for channels in CHANNELS:
        for mac_interval_ps in MAC_INTERVALS_PS:
            for scalar_bytes in PARTIAL_SCALAR_BYTES:
                topology = CXLPIMTopology(
                    channels=channels,
                    pseudo_channels_per_channel=PSEUDO_CHANNELS_PER_CHANNEL,
                    banks_per_pseudo_channel=BANKS_PER_PSEUDO_CHANNEL,
                    partial_scalar_bytes=scalar_bytes,
                )
                design_cycle = replace(cycle, pim_mac_interval_ps=mac_interval_ps)
                shape = build_cxl_pim_attention_shape(
                    model,
                    design_cycle,
                    batch_size=BATCH_SIZE,
                    spilled_kv_bytes_per_layer=spilled_bytes,
                    topology=topology,
                )
                designs.append(
                    {
                        "design_id": f"c{channels}_mac{mac_interval_ps}_p{scalar_bytes}",
                        "channels": channels,
                        "total_pseudo_channels": topology.total_pseudo_channels,
                        "mac_interval_ps": mac_interval_ps,
                        "partial_scalar_bytes": scalar_bytes,
                        "topology": topology,
                        "cycle": design_cycle,
                        "shape": shape,
                    }
                )
    return designs


def _component_plan(designs: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    planned: dict[str, dict[str, Any]] = {}
    design_keys: dict[str, dict[str, str]] = {}
    for design in designs:
        topology = design["topology"]
        cycle = design["cycle"]
        shape = design["shape"]
        components = {
            "query_link": _transfer_identity(shape.query_link_transactions),
            "pim_gwrite": _pim_identity("PIM_GWRITE", shape.pim_gwrite_waves, topology, cycle),
            "pim_mac": _pim_identity("PIM_MAC", shape.pim_mac_waves, topology, cycle),
            "pim_read": _pim_identity("PIM_READ", shape.pim_read_waves, topology, cycle),
            "result_link": _transfer_identity(shape.result_link_transactions),
        }
        keys: dict[str, str] = {}
        for name, identity in components.items():
            key = _key(identity)
            keys[name] = key
            entry = planned.setdefault(
                key,
                {"identity": identity, "component": name, "consumers": []},
            )
            entry["consumers"].append(design["design_id"])
        design_keys[design["design_id"]] = keys
    return planned, design_keys


def _run_or_load_new(
    output: Path,
    key: str,
    identity: dict[str, Any],
    cycle: SieveCycleV1Config,
    topology: CXLPIMTopology,
    transfer_runner: Callable[..., CXLKVWorkloadResult],
    pim_runner: Callable[..., MicrobenchmarkResult],
) -> tuple[Path, CXLKVWorkloadResult | MicrobenchmarkResult, bool]:
    cache_dir = output / "exact_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    cache_input = {
        "schema_version": 1,
        "stage": "kv-cxl-pim-sensitivity-stage9-v1",
        "simulation_identity": identity,
    }
    if path.is_file():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("cache_input") != cache_input:
            raise ValueError(f"Stage-9 cache provenance mismatch: {path}")
        return path, _deserialize(path), False

    if identity["kind"] == "cxl-fifo-transfer-read-proxy":
        result = transfer_runner(
            stage5.RAMULATOR,
            cycle,
            gpu_read_transactions=0,
            local_kv_read_transactions=0,
            cxl_kv_read_transactions=int(identity["transactions"]),
            pim_gwrite_waves=0,
            pim_mac_waves=0,
            pim_read_waves=0,
            cxl_bandwidth_bytes_per_second=float(stage5.PROFILE["bandwidth_gb_s"]) * 1e9,
            cxl_latency_us=float(stage5.PROFILE["latency_us"]),
            cxl_channels=int(stage5.PROFILE["channel_count"]),
        )
        result_type = "cxl-transfer"
    else:
        result = pim_runner(
            stage5.RAMULATOR,
            cycle,
            topology,
            str(identity["operation"]),
            int(identity["waves"]),
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
    return path, result, True


def _duration_us(result: CXLKVWorkloadResult | MicrobenchmarkResult) -> float:
    if isinstance(result, CXLKVWorkloadResult):
        return result.cxl_kv_completion_cycles * result.tick_ps / 1_000_000.0
    return result.duration_us_by_waves[result.max_waves]


def _manifest_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _factor_summary(rows: list[dict[str, Any]], memory_only_throughput: float, resident_throughput: float) -> list[dict[str, Any]]:
    factors = (
        "channels",
        "total_pseudo_channels",
        "mac_interval_ps",
        "partial_scalar_bytes",
        "overlap_fraction",
        "merge_us_per_layer",
    )
    result: list[dict[str, Any]] = []
    for factor in factors:
        for value in sorted({row[factor] for row in rows}):
            group = [row for row in rows if row[factor] == value]
            decode = [float(row["estimated_decode_ms"]) for row in group]
            throughput = [float(row["estimated_throughput_tokens_per_s"]) for row in group]
            result.append(
                {
                    "factor": factor,
                    "value": value,
                    "rows": len(group),
                    "mean_decode_ms": statistics.fmean(decode),
                    "median_decode_ms": statistics.median(decode),
                    "min_decode_ms": min(decode),
                    "max_decode_ms": max(decode),
                    "mean_throughput_tokens_per_s": statistics.fmean(throughput),
                    "fraction_beating_memory_only": sum(value > memory_only_throughput for value in throughput) / len(group),
                    "fraction_beating_resident": sum(value > resident_throughput for value in throughput) / len(group),
                }
            )
    return result


def _report(
    output: Path,
    rows: list[dict[str, Any]],
    design_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    best = min(rows, key=lambda row: row["estimated_decode_ms"])
    worst = max(rows, key=lambda row: row["estimated_decode_ms"])
    robust = [row for row in design_rows if row["robustly_beats_memory_only"]]
    lines = [
        "# KV CXL-PIM sensitivity Stage 9",
        "",
        "Stage 9 scans 18 hardware/data-format designs and nine scheduling/merge assumptions per design. Exact Ramulator components are deduplicated by simulation inputs; Stage-8 baseline components are reused by SHA-256.",
        "",
        f"Design points: `{metadata['design_points']}`; scheduling rows: `{metadata['rows']}`; exact components: `{metadata['exact_components']}`; Stage-8 reused: `{metadata['reused_stage8_components']}`; new exact runs: `{metadata['new_exact_components']}`; interpolation: `0`.",
        "",
        f"Best row: `{best['design_id']}`, overlap={best['overlap_fraction']:.1f}, merge={best['merge_us_per_layer']:.1f} us/layer, Decode={best['estimated_decode_ms']:.3f} ms.",
        f"Worst row: `{worst['design_id']}`, overlap={worst['overlap_fraction']:.1f}, merge={worst['merge_us_per_layer']:.1f} us/layer, Decode={worst['estimated_decode_ms']:.3f} ms.",
        f"Designs that beat memory-only CXL even at their worst scheduling/merge row: `{len(robust)}/{len(design_rows)}`.",
        "",
        "These are model-sensitivity results, not physical CXL-PIM measurements. The scan still uses isolated phase runs, an A800-fit B8/C32k local branch, a fixed 64 GB/s FIFO link, and zero paging/protocol/energy modeling.",
        "",
    ]
    (output / "stage9_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_stage9(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    transfer_runner: Callable[..., CXLKVWorkloadResult] = run_cxl_kv_workload,
    pim_runner: Callable[..., MicrobenchmarkResult] = run_cxl_pim_operation,
) -> dict[str, Any]:
    output = Path(output_dir)
    if output.exists():
        existing = {item.name for item in output.iterdir()}
        if existing and not existing.issubset({"exact_cache"}):
            raise ValueError(f"Stage-9 output already exists and is not resumable: {output}")
    output.mkdir(parents=True, exist_ok=True)

    stage8 = _load_stage8()
    configuration, _ = stage5._source_configuration()
    base_cycle = SieveCycleV1Config.load(stage5.CYCLE)
    spilled_bytes = int(stage8["derived"]["memory_only_cxl_link_bytes_per_layer"])
    designs = _designs(configuration.model, base_cycle, spilled_bytes)
    planned, design_keys = _component_plan(designs)
    stage8_index = _stage8_cache_index(stage8, base_cycle)

    results: dict[str, CXLKVWorkloadResult | MicrobenchmarkResult] = {}
    cache_records: list[dict[str, Any]] = []
    new_runs = 0
    reused = 0
    design_by_id = {design["design_id"]: design for design in designs}
    for key, entry in planned.items():
        if key in stage8_index:
            source = stage8_index[key]
            path = source["path"]
            result = _deserialize(path)
            executed = False
            reused += 1
            source_name = "stage8-reuse"
        else:
            consumer = design_by_id[entry["consumers"][0]]
            path, result, executed = _run_or_load_new(
                output,
                key,
                entry["identity"],
                consumer["cycle"],
                consumer["topology"],
                transfer_runner,
                pim_runner,
            )
            new_runs += int(executed)
            source_name = "stage9-exact-cache"
        results[key] = result
        cache_records.append(
            {
                "key": key,
                "component": entry["component"],
                "consumers": sorted(entry["consumers"]),
                "source": source_name,
                "cache_path": _manifest_path(path),
                "sha256": stage5._sha256(path),
                "executed_this_invocation": executed,
                "simulation_identity": entry["identity"],
            }
        )

    resident = next(row for row in stage8["rows"] if row["scenario"] == "resident-local-counterfactual")
    memory_only = next(row for row in stage8["rows"] if row["scenario"] == "nominal-memory-only-cxl-spill")
    local_attention_ms = float(stage8["derived"]["local_gpu_attention_ms"])
    non_attention_ms = float(resident["estimated_decode_ms"]) - float(resident["estimated_attention_ms"])
    memory_only_throughput = float(memory_only["estimated_throughput_tokens_per_s"])
    resident_throughput = float(resident["estimated_throughput_tokens_per_s"])
    rows: list[dict[str, Any]] = []
    design_rows: list[dict[str, Any]] = []
    for design in designs:
        keys = design_keys[design["design_id"]]
        components = {name: _duration_us(results[key]) for name, key in keys.items()}
        branch_us = sum(components.values())
        branch_ms = branch_us * LAYERS / 1000.0
        design_result_rows: list[dict[str, Any]] = []
        for overlap in OVERLAP_FRACTIONS:
            for merge_us in MERGE_US_PER_LAYER:
                attention_ms = (
                    local_attention_ms
                    + branch_ms
                    - overlap * min(local_attention_ms, branch_ms)
                    + merge_us * LAYERS / 1000.0
                )
                decode_ms = attention_ms + non_attention_ms
                row = {
                    "design_id": design["design_id"],
                    "channels": design["channels"],
                    "total_pseudo_channels": design["total_pseudo_channels"],
                    "mac_interval_ps": design["mac_interval_ps"],
                    "partial_scalar_bytes": design["partial_scalar_bytes"],
                    "overlap_fraction": overlap,
                    "merge_us_per_layer": merge_us,
                    "query_link_us_per_layer": components["query_link"],
                    "pim_gwrite_us_per_layer": components["pim_gwrite"],
                    "pim_mac_us_per_layer": components["pim_mac"],
                    "pim_read_us_per_layer": components["pim_read"],
                    "result_link_us_per_layer": components["result_link"],
                    "cxl_pim_branch_us_per_layer": branch_us,
                    "local_gpu_attention_ms": local_attention_ms,
                    "cxl_pim_branch_ms_per_decode": branch_ms,
                    "estimated_attention_ms": attention_ms,
                    "estimated_decode_ms": decode_ms,
                    "estimated_throughput_tokens_per_s": BATCH_SIZE * 1000.0 / decode_ms,
                    "beats_memory_only_cxl": BATCH_SIZE * 1000.0 / decode_ms > memory_only_throughput,
                    "beats_resident_counterfactual": BATCH_SIZE * 1000.0 / decode_ms > resident_throughput,
                }
                rows.append(row)
                design_result_rows.append(row)
        design_rows.append(
            {
                "design_id": design["design_id"],
                "channels": design["channels"],
                "total_pseudo_channels": design["total_pseudo_channels"],
                "mac_interval_ps": design["mac_interval_ps"],
                "partial_scalar_bytes": design["partial_scalar_bytes"],
                "query_bytes": design["shape"].query_bytes,
                "partial_result_bytes": design["shape"].partial_result_bytes,
                "pim_mac_waves": design["shape"].pim_mac_waves,
                "branch_us_per_layer": branch_us,
                "best_decode_ms": min(row["estimated_decode_ms"] for row in design_result_rows),
                "worst_decode_ms": max(row["estimated_decode_ms"] for row in design_result_rows),
                "best_throughput_tokens_per_s": max(row["estimated_throughput_tokens_per_s"] for row in design_result_rows),
                "worst_throughput_tokens_per_s": min(row["estimated_throughput_tokens_per_s"] for row in design_result_rows),
                "robustly_beats_memory_only": all(row["beats_memory_only_cxl"] for row in design_result_rows),
                "robustly_beats_resident": all(row["beats_resident_counterfactual"] for row in design_result_rows),
            }
        )

    factor_rows = _factor_summary(rows, memory_only_throughput, resident_throughput)
    _write_csv(output / "sensitivity.csv", rows)
    _write_csv(output / "design_summary.csv", design_rows)
    _write_csv(output / "factor_summary.csv", factor_rows)
    metadata = {
        "schema_version": 1,
        "stage": "kv-cxl-pim-sensitivity-stage9-v1",
        "method": "factorial isolated exact-component sensitivity with A800-fit Decode composition",
        "classification": "Ramulator component sensitivity plus analytic scheduling composition; no physical CXL-PIM timing",
        "design_points": len(designs),
        "rows": len(rows),
        "exact_components": len(planned),
        "reused_stage8_components": reused,
        "new_exact_components": len(planned) - reused,
        "new_runs_executed": new_runs,
        "interpolation": 0,
        "factors": {
            "channels": list(CHANNELS),
            "pseudo_channels_per_channel": PSEUDO_CHANNELS_PER_CHANNEL,
            "mac_intervals_ps": list(MAC_INTERVALS_PS),
            "partial_scalar_bytes": list(PARTIAL_SCALAR_BYTES),
            "overlap_fractions": list(OVERLAP_FRACTIONS),
            "merge_us_per_layer": list(MERGE_US_PER_LAYER),
        },
        "baseline": {
            "memory_only_decode_ms": float(memory_only["estimated_decode_ms"]),
            "memory_only_throughput_tokens_per_s": memory_only_throughput,
            "resident_decode_ms": float(resident["estimated_decode_ms"]),
            "resident_throughput_tokens_per_s": resident_throughput,
        },
        "cache_records": cache_records,
        "execution": stage5._execution_provenance(),
        "input_hashes": {
            "stage8": stage5._sha256(STAGE8),
            "stage8_validation": stage5._sha256(STAGE8_VALIDATION),
            "model": stage5._sha256(configuration.experiment.model_path),
            "cycle": stage5._sha256(stage5.CYCLE),
        },
        "limitations": [
            "Topology, MAC interval, partial precision, overlap, and merge values are model assumptions.",
            "Exact phases are isolated and combined analytically; there is no unified CXL-PIM controller contention model.",
            "The B8/C32k local GPU branch is extrapolated from A800 B8/C4k-C16k measurements.",
            "The link remains a fixed 64 GB/s FIFO READ proxy; physical CXL protocol and direction effects are absent.",
            "Paging, eviction, recovery, softmax scalar latency, energy, multi-GPU sharing, and Shared CXL-PIM are absent.",
        ],
    }
    comparison_path = output / "sensitivity.json"
    comparison_path.write_text(
        json.dumps(
            {"metadata": metadata, "rows": rows, "design_summary": design_rows, "factor_summary": factor_rows},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _report(output, rows, design_rows, metadata)
    manifest = {
        "schema_version": 1,
        "stage": metadata["stage"],
        "design_points": len(designs),
        "rows": len(rows),
        "exact_components": len(planned),
        "reused_stage8_components": reused,
        "new_exact_components": len(planned) - reused,
        "new_runs_executed": new_runs,
        "interpolation": 0,
        "execution": metadata["execution"],
        "input_hashes": metadata["input_hashes"],
        "cache_records": cache_records,
        "output_hashes": {
            "design_summary.csv": stage5._sha256(output / "design_summary.csv"),
            "factor_summary.csv": stage5._sha256(output / "factor_summary.csv"),
            "sensitivity.csv": stage5._sha256(output / "sensitivity.csv"),
            "sensitivity.json": stage5._sha256(comparison_path),
            "stage9_report.md": stage5._sha256(output / "stage9_report.md"),
        },
    }
    (output / "stage_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"metadata": metadata, "rows": rows, "design_summary": design_rows, "factor_summary": factor_rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    result = run_stage9(args.output)
    print(json.dumps(result["metadata"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
