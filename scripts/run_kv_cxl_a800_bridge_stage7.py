#!/usr/bin/env python3
"""Bridge independent A800 timing to exact B8/C32k CXL KV simulation.

Stage 7 keeps the evidence classes separate.  It fits an A800 Attention trend
from the independently measured B8 anchors, executes one exact full-count
Local-HBM Attention request stream as a capacity-unconstrained counterfactual,
and pairs that result with the frozen Stage-6 nominal CXL spill result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
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
from sieve_replay.ramulator.mixed_workload import SieveCycleV1Config


A800_SUMMARY = ROOT / "results/a800_timing_memory_v1/summary.json"
STAGE6 = ROOT / "results/kv_cxl_phase_stage6_b8_c32k_formal_v1/comparison.json"
DEFAULT_OUTPUT = ROOT / "results/kv_cxl_a800_bridge_stage7_v1"
TARGET_CASE = "b8_c32k"
TARGET_BATCH = 8
TARGET_CONTEXT = 32768
LAYERS = 48
TRANSACTION_BYTES = 32


def _key(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _linear_fit(points: list[tuple[float, float]]) -> dict[str, float]:
    if len(points) < 2:
        raise ValueError("linear calibration requires at least two points")
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    if denominator == 0:
        raise ValueError("calibration contexts must differ")
    slope = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in points
    ) / denominator
    intercept = y_mean - slope * x_mean
    residual_sum = sum(
        (y_value - (intercept + slope * x_value)) ** 2
        for x_value, y_value in points
    )
    total_sum = sum((value - y_mean) ** 2 for value in ys)
    return {
        "intercept_ms": intercept,
        "slope_ms_per_context_token": slope,
        "r_squared": 1.0 - residual_sum / total_sum if total_sum else 1.0,
    }


def _load_a800_calibration() -> dict[str, Any]:
    raw = json.loads(A800_SUMMARY.read_text(encoding="utf-8"))
    anchors = sorted(
        (
            row
            for row in raw["cases"]
            if int(row["batch_size"]) == TARGET_BATCH
            and int(row["context_length"]) in {4096, 8192, 16384}
        ),
        key=lambda row: int(row["context_length"]),
    )
    if [int(row["context_length"]) for row in anchors] != [4096, 8192, 16384]:
        raise ValueError("Stage 7 requires A800 B8/C4k, C8k and C16k anchors")
    points = [
        (float(row["context_length"]), float(row["attention_decode_mean_ms"]))
        for row in anchors
    ]
    fit = _linear_fit(points)
    forward_fit = _linear_fit(points[:2])
    predicted_c16k = (
        forward_fit["intercept_ms"]
        + forward_fit["slope_ms_per_context_token"] * points[2][0]
    )
    observed_c16k = points[2][1]
    non_attention = [
        float(row["decode_mean_ms"]) - float(row["attention_decode_mean_ms"])
        for row in anchors
    ]
    target_attention = (
        fit["intercept_ms"]
        + fit["slope_ms_per_context_token"] * TARGET_CONTEXT
    )
    return {
        "source": str(A800_SUMMARY.relative_to(ROOT)),
        "source_sha256": stage5._sha256(A800_SUMMARY),
        "anchors": [
            {
                "case": row["case"],
                "context_length": int(row["context_length"]),
                "attention_decode_mean_ms": float(row["attention_decode_mean_ms"]),
                "decode_mean_ms": float(row["decode_mean_ms"]),
                "non_attention_decode_ms": float(row["decode_mean_ms"])
                - float(row["attention_decode_mean_ms"]),
            }
            for row in anchors
        ],
        "fit": fit,
        "forward_holdout": {
            "train_contexts": [4096, 8192],
            "held_out_context": 16384,
            "predicted_attention_ms": predicted_c16k,
            "observed_attention_ms": observed_c16k,
            "absolute_percentage_error": abs(predicted_c16k - observed_c16k)
            / observed_c16k,
        },
        "target": {
            "case": TARGET_CASE,
            "context_length": TARGET_CONTEXT,
            "attention_extrapolated_ms": target_attention,
            "non_attention_mean_ms": statistics.fmean(non_attention),
            "non_attention_range_ms": max(non_attention) - min(non_attention),
        },
    }


def _load_stage6_spill() -> tuple[dict[str, Any], CXLKVWorkloadResult, Path]:
    raw = json.loads(STAGE6.read_text(encoding="utf-8"))
    metadata = raw["metadata"]
    if metadata["case"] != TARGET_CASE or metadata["interpolation"] != 0:
        raise ValueError("Stage-6 source is not the exact B8/C32k result")
    rows = [
        row
        for row in raw["rows"]
        if int(row["request_scale"]) == 1 and row["policy"] == "gpu-only"
    ]
    if len(rows) != 1:
        raise ValueError("Stage 7 requires one Stage-6 scale=1 gpu-only row")
    row = rows[0]
    cache_records = {
        record["cache_key"]: record for record in metadata["cache_records"]
    }
    cache_key = row["attention_cache_key"]
    record = cache_records.get(cache_key)
    if record is None or record["phase"] != "attention":
        raise ValueError("Stage-6 Attention cache binding is missing")
    cache_path = STAGE6.parent / "exact_cache" / f"{cache_key}.json"
    if stage5._sha256(cache_path) != record["cache_sha256"]:
        raise ValueError("Stage-6 Attention cache hash mismatch")
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    result = CXLKVWorkloadResult(**cached["result"])
    shape = row["attention_shape"]
    if (
        result.local_kv_completed_requests != int(shape["local_kv_read_transactions"])
        or result.cxl_kv_completed_requests != int(shape["cxl_kv_read_transactions"])
    ):
        raise ValueError("Stage-6 Attention completion counts are inconsistent")
    return row, result, cache_path


def _resident_cache_input(
    shape: dict[str, int], calibration: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": "kv-cxl-a800-bridge-stage7-v1",
        "case": TARGET_CASE,
        "phase": "attention-resident-counterfactual",
        "request_scale": 1,
        "shape": shape,
        "local_memory_profile": {
            "cycle_config": str(stage5.CYCLE.relative_to(ROOT)),
            "transaction_bytes": TRANSACTION_BYTES,
            "channels": 128,
        },
        "unused_cxl_controller_profile": {
            "bandwidth_gb_s": stage5.PROFILE["bandwidth_gb_s"],
            "latency_us": stage5.PROFILE["latency_us"],
            "channel_count": stage5.PROFILE["channel_count"],
        },
        "simulation_context": cxl_kv_cache_context(ROOT, stage5.CYCLE),
        "input_hashes": {
            "a800_summary": calibration["source_sha256"],
            "stage6_comparison": stage5._sha256(STAGE6),
            "cycle_config": stage5._sha256(stage5.CYCLE),
        },
    }


def _run_or_load_resident(
    output: Path,
    shape: dict[str, int],
    calibration: dict[str, Any],
    cycle: SieveCycleV1Config,
    workload_runner: Callable[..., CXLKVWorkloadResult],
) -> tuple[str, Path, CXLKVWorkloadResult, bool]:
    cache_input = _resident_cache_input(shape, calibration)
    cache_key = _key(cache_input)
    cache_dir = output / "exact_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key}.json"
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("cache_input") != cache_input:
            raise ValueError("Stage-7 resident cache provenance mismatch")
        return cache_key, cache_path, CXLKVWorkloadResult(**cached["result"]), False
    result = workload_runner(
        stage5.RAMULATOR,
        cycle,
        **shape,
        cxl_bandwidth_bytes_per_second=float(stage5.PROFILE["bandwidth_gb_s"])
        * 1e9,
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


def _completion_us(result: CXLKVWorkloadResult, field: str) -> float:
    return getattr(result, field) * result.tick_ps / 1_000_000.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    output: Path,
    calibration: dict[str, Any],
    rows: list[dict[str, Any]],
    derived: dict[str, Any],
) -> None:
    resident, spill = rows
    holdout = calibration["forward_holdout"]
    lines = [
        "# KV CXL A800 bridge Stage 7",
        "",
        "Stage 7 combines evidence only through an explicit delta model: the B8/C32k A800 baseline is a measured-trend extrapolation, while the Local-HBM/CXL memory delta comes from paired exact Ramulator request streams.",
        "",
        "| scenario | capacity state | estimated Attention ms | estimated Decode ms | throughput tok/s | memory completion us/layer |",
        "|---|---|---:|---:|---:|---:|",
        f"| {resident['scenario']} | {resident['capacity_state']} | {resident['estimated_attention_ms']:.3f} | {resident['estimated_decode_ms']:.3f} | {resident['estimated_throughput_tokens_per_s']:.3f} | {resident['memory_completion_us_per_layer']:.3f} |",
        f"| {spill['scenario']} | {spill['capacity_state']} | {spill['estimated_attention_ms']:.3f} | {spill['estimated_decode_ms']:.3f} | {spill['estimated_throughput_tokens_per_s']:.3f} | {spill['memory_completion_us_per_layer']:.3f} |",
        "",
        f"The forward holdout trained on C4k/C8k predicts C16k Attention with {100.0 * holdout['absolute_percentage_error']:.3f}% absolute error. The all-anchor linear fit has R^2={calibration['fit']['r_squared']:.6f}.",
        f"Under the nominal 64 GB/s memory-only CXL assumption, spill adds {derived['nominal_cxl_memory_penalty_ms_per_decode']:.3f} ms to the extrapolated Decode estimate and changes estimated throughput by {100.0 * derived['nominal_cxl_throughput_change_fraction']:.3f}%.",
        f"Serving the spilled payload by the exact all-local completion deadline would require {derived['cxl_payload_rate_to_resident_deadline_gb_s']:.3f} GB/s before protocol and compute costs; this is a derived payload-rate threshold, not a physical CXL measurement.",
        "",
        "The resident B8/C32k row is capacity-infeasible on the A800 domain and was not run on A800. The CXL row is an A800-anchored extrapolation plus a Ramulator delta, not a physical CXL result. No paging, eviction, transfer state machine, CXL protocol, CXL-PIM compute, or multi-GPU execution is modeled.",
        "",
    ]
    (output / "stage7_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_stage7(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    workload_runner: Callable[..., CXLKVWorkloadResult] = run_cxl_kv_workload,
) -> dict[str, Any]:
    output = Path(output_dir)
    if output.exists():
        existing = {item.name for item in output.iterdir()}
        if existing and not existing.issubset({"exact_cache"}):
            raise ValueError(f"Stage-7 output already exists and is not resumable: {output}")
    output.mkdir(parents=True, exist_ok=True)

    calibration = _load_a800_calibration()
    spill_row, spill_result, spill_cache_path = _load_stage6_spill()
    spill_shape = {key: int(value) for key, value in spill_row["attention_shape"].items()}
    total_transactions = (
        spill_shape["local_kv_read_transactions"]
        + spill_shape["cxl_kv_read_transactions"]
    )
    resident_shape = {
        "gpu_read_transactions": 0,
        "local_kv_read_transactions": total_transactions,
        "cxl_kv_read_transactions": 0,
        "pim_gwrite_waves": 0,
        "pim_mac_waves": 0,
        "pim_read_waves": 0,
    }
    cycle = SieveCycleV1Config.load(stage5.CYCLE)
    cache_key, cache_path, resident_result, executed = _run_or_load_resident(
        output, resident_shape, calibration, cycle, workload_runner
    )
    if resident_result.local_kv_completed_requests != total_transactions:
        raise ValueError("Stage-7 resident Local-HBM request stream is incomplete")
    if resident_result.cxl_kv_completed_requests != 0:
        raise ValueError("Stage-7 resident counterfactual must not contain CXL requests")

    resident_completion_us = _completion_us(
        resident_result, "local_kv_completion_cycles"
    )
    spill_local_us = _completion_us(spill_result, "local_kv_completion_cycles")
    spill_cxl_us = _completion_us(spill_result, "cxl_kv_completion_cycles")
    spill_completion_us = max(spill_local_us, spill_cxl_us)
    memory_penalty_ms = LAYERS * (spill_completion_us - resident_completion_us) / 1000.0
    extrapolated_attention_ms = calibration["target"]["attention_extrapolated_ms"]
    non_attention_ms = calibration["target"]["non_attention_mean_ms"]
    resident_decode_ms = extrapolated_attention_ms + non_attention_ms
    spill_attention_ms = extrapolated_attention_ms + memory_penalty_ms
    spill_decode_ms = resident_decode_ms + memory_penalty_ms

    capacity = json.loads(stage5.CAPACITY.read_text(encoding="utf-8"))
    capacity_rows = [
        row
        for row in capacity["rows"]
        if row["domain"] == "a800"
        and int(row["batch_size"]) == TARGET_BATCH
        and int(row["context_length"]) == TARGET_CONTEXT
        and int(row["cxl_capacity_bytes"]) == 64_000_000_000
    ]
    if len(capacity_rows) != 1 or capacity_rows[0]["state"] != "spill":
        raise ValueError("Stage-7 A800/CXL capacity source is inconsistent")
    capacity_row = capacity_rows[0]

    rows = [
        {
            "scenario": "resident-local-counterfactual",
            "evidence_class": "A800 trend extrapolation plus exact Ramulator Local-HBM counterfactual",
            "capacity_state": "infeasible-on-a800",
            "estimated_attention_ms": extrapolated_attention_ms,
            "estimated_decode_ms": resident_decode_ms,
            "estimated_throughput_tokens_per_s": TARGET_BATCH * 1000.0 / resident_decode_ms,
            "memory_completion_us_per_layer": resident_completion_us,
            "memory_delta_vs_resident_ms_per_decode": 0.0,
            "cxl_bandwidth_gb_s": 0.0,
            "cxl_latency_us": 0.0,
            "local_kv_read_transactions": total_transactions,
            "cxl_kv_read_transactions": 0,
        },
        {
            "scenario": "nominal-memory-only-cxl-spill",
            "evidence_class": "A800 trend extrapolation plus paired exact Ramulator CXL spill delta",
            "capacity_state": "spill",
            "estimated_attention_ms": spill_attention_ms,
            "estimated_decode_ms": spill_decode_ms,
            "estimated_throughput_tokens_per_s": TARGET_BATCH * 1000.0 / spill_decode_ms,
            "memory_completion_us_per_layer": spill_completion_us,
            "memory_delta_vs_resident_ms_per_decode": memory_penalty_ms,
            "cxl_bandwidth_gb_s": float(stage5.PROFILE["bandwidth_gb_s"]),
            "cxl_latency_us": float(stage5.PROFILE["latency_us"]),
            "local_kv_read_transactions": spill_shape["local_kv_read_transactions"],
            "cxl_kv_read_transactions": spill_shape["cxl_kv_read_transactions"],
        },
    ]
    cxl_payload_bytes = spill_shape["cxl_kv_read_transactions"] * TRANSACTION_BYTES
    derived = {
        "nominal_cxl_memory_penalty_ms_per_decode": memory_penalty_ms,
        "nominal_cxl_throughput_change_fraction": rows[1]["estimated_throughput_tokens_per_s"]
        / rows[0]["estimated_throughput_tokens_per_s"]
        - 1.0,
        "cxl_payload_bytes_per_layer": cxl_payload_bytes,
        "nominal_cxl_effective_payload_rate_gb_s": cxl_payload_bytes
        / (spill_cxl_us * 1000.0),
        "cxl_payload_rate_to_resident_deadline_gb_s": cxl_payload_bytes
        / (resident_completion_us * 1000.0),
        "resident_local_completion_us_per_layer": resident_completion_us,
        "spill_local_completion_us_per_layer": spill_local_us,
        "spill_cxl_completion_us_per_layer": spill_cxl_us,
        "spill_fraction_of_kv_bytes": int(capacity_row["spill_bytes"])
        / int(capacity_row["kv_cache_bytes"]),
    }

    _write_csv(output / "comparison.csv", rows)
    (output / "calibration.json").write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(output, calibration, rows, derived)
    execution = stage5._execution_provenance()
    metadata = {
        "schema_version": 1,
        "stage": "kv-cxl-a800-bridge-stage7-v1",
        "case": TARGET_CASE,
        "method": "A800 measured-trend extrapolation plus paired exact Ramulator memory delta",
        "classification": "mixed-source sensitivity analysis with explicit evidence separation",
        "exact_new_runs": 1,
        "new_runs_executed": int(executed),
        "interpolation": 0,
        "layers": LAYERS,
        "execution": execution,
        "input_hashes": {
            "a800_summary": stage5._sha256(A800_SUMMARY),
            "stage6_comparison": stage5._sha256(STAGE6),
            "stage6_attention_cache": stage5._sha256(spill_cache_path),
            "capacity_source": stage5._sha256(stage5.CAPACITY),
            "cycle_config": stage5._sha256(stage5.CYCLE),
        },
        "exact_cache": {
            "key": cache_key,
            "sha256": stage5._sha256(cache_path),
            "shape": resident_shape,
        },
        "capacity": {
            "a800_local_capacity_bytes": int(capacity_row["local_capacity_bytes"]),
            "cxl_capacity_bytes": int(capacity_row["cxl_capacity_bytes"]),
            "peak_memory_bytes": int(capacity_row["peak_memory_bytes"]),
            "kv_cache_bytes": int(capacity_row["kv_cache_bytes"]),
            "spill_bytes": int(capacity_row["spill_bytes"]),
        },
        "limitations": [
            "B8/C32k A800 Attention and Decode values are linear extrapolations, not A800 measurements.",
            "The resident B8/C32k request stream is a capacity-unconstrained Ramulator counterfactual and is infeasible in the formal A800 80 GB admission domain.",
            "The CXL estimate adds an exact Ramulator memory-completion delta to the A800 trend; it is not a physical CXL measurement.",
            "No paging transfer state machine, eviction, recovery, physical CXL protocol, CXL-PIM computation, or multi-GPU execution is modeled.",
        ],
    }
    comparison_path = output / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {"metadata": metadata, "calibration": calibration, "derived": derived, "rows": rows},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "stage": metadata["stage"],
        "case": TARGET_CASE,
        "rows": len(rows),
        "exact_new_runs": 1,
        "new_runs_executed": int(executed),
        "interpolation": 0,
        "execution": execution,
        "input_hashes": metadata["input_hashes"],
        "output_hashes": {
            "calibration.json": stage5._sha256(output / "calibration.json"),
            "comparison.csv": stage5._sha256(output / "comparison.csv"),
            "comparison.json": stage5._sha256(comparison_path),
            "stage7_report.md": stage5._sha256(output / "stage7_report.md"),
        },
    }
    (output / "stage_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"metadata": metadata, "calibration": calibration, "derived": derived, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    result = run_stage7(args.output)
    print(json.dumps({"metadata": result["metadata"], "derived": result["derived"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
