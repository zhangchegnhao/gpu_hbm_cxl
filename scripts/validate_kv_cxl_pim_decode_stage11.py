#!/usr/bin/env python3
"""Validate Stage-11 CXL-PIM timing reuse and 48-layer Decode arithmetic."""

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
import run_kv_cxl_pim_decode_stage11 as stage11


def _close(left: float, right: float, tolerance: float = 1e-8) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def validate(output_dir: str | Path = stage11.DEFAULT_OUTPUT) -> dict[str, Any]:
    output = Path(output_dir)
    comparison_path = output / "comparison.json"
    manifest_path = output / "stage_manifest.json"
    layer_path = output / "layer_results.csv"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = comparison["metadata"]
    rows = comparison["rows"]
    errors: list[str] = []

    if metadata.get("stage") != "kv-cxl-pim-decode-stage11-v1":
        errors.append("unexpected Stage-11 identifier")
    if manifest.get("stage") != metadata.get("stage"):
        errors.append("manifest stage mismatch")
    if metadata.get("case") != stage11.CASE or manifest.get("case") != stage11.CASE:
        errors.append("unexpected Stage-11 case")
    if metadata.get("interpolation") != 0 or manifest.get("interpolation") != 0:
        errors.append("Stage 11 must not interpolate")
    if metadata.get("new_ramulator_runs") != 0 or manifest.get("new_ramulator_runs") != 0:
        errors.append("Stage 11 must not execute new Ramulator workloads")
    if metadata.get("reused_exact_cache_artifacts") != 6:
        errors.append("Stage 11 must bind six exact cache artifacts")

    configuration, _ = stage5._source_configuration()
    current_inputs = stage11._input_hashes(configuration)
    if metadata.get("input_hashes") != current_inputs:
        errors.append("current Stage-11 inputs differ from recorded hashes")
    if manifest.get("input_hashes") != current_inputs:
        errors.append("manifest input hashes mismatch")

    stage6_source = stage11._load_stage6()
    stage7_source = stage11._load_stage7()
    stage10_source = stage11._load_stage10()
    expected_sources = [
        stage6_source["cache_artifacts"]["attention"],
        stage6_source["cache_artifacts"]["expert"],
        stage7_source["resident_cache"],
        *stage10_source["cache_artifacts"],
    ]
    if metadata.get("exact_sources") != expected_sources:
        errors.append("comparison exact-source bindings mismatch")
    if manifest.get("exact_sources") != expected_sources:
        errors.append("manifest exact-source bindings mismatch")
    for record in expected_sources:
        path = ROOT / record["path"]
        if not path.is_file() or stage5._sha256(path) != record["sha256"]:
            errors.append(f"exact source hash mismatch: {record['kind']}")

    expected_keys = {
        ("resident-local-counterfactual", "none", "resident"),
        ("memory-only-cxl", "memory-only-64GBps", "paired-memory-delta"),
        *{
            ("cxl-pim-attention", design_id, mode)
            for design_id in stage11.DESIGN_IDS
            for mode in stage11.SCHEDULE_MODES
        },
    }
    seen: set[tuple[str, str, str]] = set()
    stage7_rows = stage7_source["rows"]
    stage10_rows = stage10_source["rows"]
    stage10_records = {
        record["design_id"]: record
        for record in json.loads(stage11.STAGE10_MANIFEST.read_text(encoding="utf-8"))[
            "cache_records"
        ]
    }
    expected_expert_us = float(stage6_source["row"]["expert_critical_path_us"])
    expected_memory = stage7_source["raw"]["metadata"]["capacity"]
    for row in rows:
        key = (row["scenario"], row["design_id"], row["schedule_mode"])
        if key not in expected_keys or key in seen:
            errors.append(f"duplicate or unexpected Stage-11 row: {key}")
            continue
        seen.add(key)
        if row["policy"] != stage11.POLICY or int(row["layer_count"]) != stage11.LAYERS:
            errors.append(f"policy/layer mismatch: {key}")
        if row["interpolation"] != 0:
            errors.append(f"row interpolation is nonzero: {key}")
        if int(row["kv_cache_bytes"]) != int(expected_memory["kv_cache_bytes"]):
            errors.append(f"KV byte mismatch: {key}")
        if int(row["peak_memory_bytes"]) != int(expected_memory["peak_memory_bytes"]):
            errors.append(f"peak-memory mismatch: {key}")
        if row["simulated_96gb_capacity_state"] != "resident":
            errors.append(f"simulated capacity mismatch: {key}")
        if not _close(row["expert_latency_us"], expected_expert_us):
            errors.append(f"Stage-6 Expert timing mismatch: {key}")
        total = float(row["total_latency_us"])
        expected_throughput = stage11.BATCH_SIZE * 1_000_000.0 / total
        if not _close(row["throughput_tokens_per_s"], expected_throughput, 1e-10):
            errors.append(f"throughput arithmetic mismatch: {key}")
        expected_share = float(row["attention_latency_us"]) / total
        if not _close(row["attention_latency_share"], expected_share, 1e-12):
            errors.append(f"Attention-share arithmetic mismatch: {key}")
        components = sum(
            float(row[name])
            for name in (
                "attention_latency_us",
                "routing_control_latency_us",
                "expert_latency_us",
                "combine_latency_us",
                "other_non_attention_latency_us",
            )
        )
        if not _close(total, components):
            errors.append(f"component accounting mismatch: {key}")

        if row["scenario"] == "resident-local-counterfactual":
            expected_ms = float(
                stage7_rows["resident-local-counterfactual"]["estimated_decode_ms"]
            )
            if row["a800_capacity_feasible"] or not row["a800_capacity_state"].startswith("oom"):
                errors.append("resident counterfactual must remain infeasible on A800")
        elif row["scenario"] == "memory-only-cxl":
            expected_ms = float(
                stage7_rows["nominal-memory-only-cxl-spill"]["estimated_decode_ms"]
            )
            if row["a800_capacity_state"] != "spill" or not row["a800_capacity_feasible"]:
                errors.append("memory-only CXL capacity state mismatch")
        else:
            design = stage10_rows[row["design_id"]]
            expected_pipeline = float(design["unified_branch_us_per_layer"])
            if not _close(row["cxl_pim_pipeline_us_per_layer"], expected_pipeline, 1e-12):
                errors.append(f"Stage-10 pipeline timing mismatch: {key}")
            if row["stage10_pipeline_cache_key"] != stage10_records[row["design_id"]]["key"]:
                errors.append(f"Stage-10 cache key mismatch: {key}")
            if row["schedule_mode"] == "ideal-overlap-bound":
                expected_ms = float(design["ideal_overlap_decode_ms"])
            else:
                expected_ms = float(design["serialized_decode_ms"])
            if row["a800_capacity_state"] != "spill" or not row["a800_capacity_feasible"]:
                errors.append(f"CXL-PIM capacity state mismatch: {key}")
        if not _close(total / 1000.0, expected_ms, 1e-9):
            errors.append(f"source Decode timing mismatch: {key}")

    if seen != expected_keys or len(rows) != len(expected_keys):
        errors.append("Stage-11 scenario/design/schedule matrix is incomplete")
    if metadata.get("rows") != len(rows) or manifest.get("rows") != len(rows):
        errors.append("Stage-11 summary row count mismatch")

    with layer_path.open(encoding="utf-8", newline="") as handle:
        layer_rows = list(csv.DictReader(handle))
    expected_layer_rows = len(expected_keys) * stage11.LAYERS
    if len(layer_rows) != expected_layer_rows:
        errors.append("Stage-11 layer row count mismatch")
    if metadata.get("layer_rows") != len(layer_rows) or manifest.get("layer_rows") != len(layer_rows):
        errors.append("Stage-11 recorded layer row count mismatch")
    by_key: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in layer_rows:
        key = (row["scenario"], row["design_id"], row["schedule_mode"])
        by_key.setdefault(key, []).append(row)
        if int(row["interpolation"]) != 0:
            errors.append(f"layer interpolation is nonzero: {key}")
    summary_by_key = {
        (row["scenario"], row["design_id"], row["schedule_mode"]): row
        for row in rows
    }
    for key in expected_keys:
        group = by_key.get(key, [])
        if len(group) != stage11.LAYERS or {int(row["layer"]) for row in group} != set(range(stage11.LAYERS)):
            errors.append(f"layer coverage mismatch: {key}")
            continue
        layer_total = sum(float(row["layer_span_us"]) for row in group)
        if not _close(layer_total, summary_by_key[key]["total_latency_us"]):
            errors.append(f"layer total mismatch: {key}")
        if key[0] == "cxl-pim-attention":
            expected_pipeline = float(stage10_rows[key[1]]["unified_branch_us_per_layer"])
        else:
            expected_pipeline = 0.0
        if any(
            not _close(float(row["cxl_pim_pipeline_us"]), expected_pipeline, 1e-9)
            for row in group
        ):
            errors.append(f"per-layer CXL-PIM timing mismatch: {key}")

    memory_row = next(row for row in rows if row["scenario"] == "memory-only-cxl")
    serialized = {
        row["design_id"]: row
        for row in rows
        if row["scenario"] == "cxl-pim-attention"
        and row["schedule_mode"] == "fully-serialized-bound"
    }
    if serialized["c2_mac49152_p4"]["total_latency_us"] <= memory_row["total_latency_us"]:
        errors.append("2-channel negative control unexpectedly beats memory-only CXL")
    for design_id in ("c4_mac24576_p4", "c8_mac24576_p4"):
        if serialized[design_id]["total_latency_us"] >= memory_row["total_latency_us"]:
            errors.append(f"expected serialized CXL-PIM design does not beat memory-only: {design_id}")

    for name, expected in manifest.get("output_hashes", {}).items():
        path = output / name
        if not path.is_file() or stage5._sha256(path) != expected:
            errors.append(f"output hash mismatch: {name}")

    validation = {
        "schema_version": 1,
        "stage": "kv-cxl-pim-decode-stage11-v1",
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "checks": {
            "rows": len(rows),
            "layer_rows": len(layer_rows),
            "designs": len(stage11.DESIGN_IDS),
            "reused_exact_cache_artifacts": len(expected_sources),
            "new_ramulator_runs": metadata.get("new_ramulator_runs"),
            "interpolation": metadata.get("interpolation"),
        },
    }
    (output / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if errors:
        raise ValueError("; ".join(errors))
    return validation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(stage11.DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    print(json.dumps(validate(args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
