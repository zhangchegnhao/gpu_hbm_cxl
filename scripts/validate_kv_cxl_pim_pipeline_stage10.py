#!/usr/bin/env python3
"""Validate Stage-10 unified CXL-PIM pipeline and precision results."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_kv_cxl_decode_stage5 as stage5
import run_kv_cxl_pim_pipeline_stage10 as stage10


def validate(output_dir: str | Path = stage10.DEFAULT_OUTPUT) -> dict[str, Any]:
    output = Path(output_dir)
    manifest = json.loads((output / "stage_manifest.json").read_text(encoding="utf-8"))
    comparison = json.loads((output / "pipeline_comparison.json").read_text(encoding="utf-8"))
    precision = json.loads((output / "partial_precision.json").read_text(encoding="utf-8"))
    errors: list[str] = []

    for name, expected in manifest["output_hashes"].items():
        path = output / name
        if not path.is_file() or stage5._sha256(path) != expected:
            errors.append(f"output hash mismatch: {name}")
    current_inputs = stage10._input_hashes()
    for name, expected in manifest["input_hashes"].items():
        if current_inputs.get(name) != expected:
            errors.append(f"input hash mismatch: {name}")

    metadata = comparison["metadata"]
    records = manifest["cache_records"]
    if metadata["cache_records"] != records:
        errors.append("manifest and comparison cache records disagree")
    if metadata["execution"] != manifest["execution"]:
        errors.append("manifest and comparison execution provenance disagree")
    if len(records) != 3 or {row["design_id"] for row in records} != set(stage10.DESIGN_IDS):
        errors.append("boundary cache matrix mismatch")
    cached_results: dict[str, dict[str, Any]] = {}
    for record in records:
        path = ROOT / record["cache_path"]
        if not path.is_file() or stage5._sha256(path) != record["sha256"]:
            errors.append(f"cache hash mismatch: {record['design_id']}")
            continue
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("cache_input") != record["cache_input"]:
            errors.append(f"cache identity mismatch: {record['design_id']}")
            continue
        result = cached["result"]
        cached_results[record["design_id"]] = result
        shape = record["cache_input"]["shape"]
        topology = record["cache_input"]["topology"]
        endpoints = int(topology["channels"]) * int(topology["pseudo_channels_per_channel"])
        expected_counts = {
            "query_link": int(shape["query_link_transactions"]),
            "pim_gwrite": int(shape["pim_gwrite_waves"]) * endpoints,
            "pim_mac": int(shape["pim_mac_waves"]) * endpoints,
            "pim_read": int(shape["pim_read_waves"]) * endpoints,
            "result_link": int(shape["result_link_transactions"]),
        }
        for phase, expected in expected_counts.items():
            if int(result[f"{phase}_injected_requests"]) != expected or int(result[f"{phase}_completed_requests"]) != expected:
                errors.append(f"request count mismatch: {record['design_id']} {phase}")
        if int(result["controller_link_requests"]) != expected_counts["query_link"] + expected_counts["result_link"]:
            errors.append(f"link controller count mismatch: {record['design_id']}")
        for operation in ("gwrite", "mac", "read"):
            if int(result[f"controller_pim_{operation}_requests"]) != expected_counts[f"pim_{operation}"]:
                errors.append(f"PIM controller count mismatch: {record['design_id']} {operation}")
        milestones = [
            int(result["query_link_completion_cycles"]),
            int(result["pim_gwrite_start_cycles"]),
            int(result["pim_gwrite_completion_cycles"]),
            int(result["pim_mac_start_cycles"]),
            int(result["pim_mac_completion_cycles"]),
            int(result["pim_read_start_cycles"]),
            int(result["pim_read_completion_cycles"]),
            int(result["result_link_start_cycles"]),
            int(result["result_link_completion_cycles"]),
        ]
        if milestones != sorted(milestones) or int(result["pipeline_completion_cycles"]) != milestones[-1]:
            errors.append(f"phase dependency mismatch: {record['design_id']}")

    rows = comparison["rows"]
    if len(rows) != 3 or {row["design_id"] for row in rows} != set(stage10.DESIGN_IDS):
        errors.append("pipeline comparison row matrix mismatch")
    if metadata["interpolation"] != 0 or manifest["interpolation"] != 0:
        errors.append("Stage 10 must not interpolate")
    stage9_source = json.loads(stage10.STAGE9.read_text(encoding="utf-8"))
    isolated = {
        row["design_id"]: float(row["branch_us_per_layer"])
        for row in stage9_source["design_summary"]
    }
    tick_ps_scale = 1_000_000.0
    for row in rows:
        result = cached_results.get(row["design_id"])
        if result is None:
            continue
        tick_ps = int(result["tick_ps"])
        unified_us = int(result["pipeline_completion_cycles"]) * tick_ps / tick_ps_scale
        phase_us = {
            "query_link_us": int(result["query_link_completion_cycles"]) * tick_ps / tick_ps_scale,
            "pim_gwrite_us": (int(result["pim_gwrite_completion_cycles"]) - int(result["pim_gwrite_start_cycles"])) * tick_ps / tick_ps_scale,
            "pim_mac_us": (int(result["pim_mac_completion_cycles"]) - int(result["pim_mac_start_cycles"])) * tick_ps / tick_ps_scale,
            "pim_read_us": (int(result["pim_read_completion_cycles"]) - int(result["pim_read_start_cycles"])) * tick_ps / tick_ps_scale,
            "result_link_us": (int(result["result_link_completion_cycles"]) - int(result["result_link_start_cycles"])) * tick_ps / tick_ps_scale,
        }
        expected_values = {
            "isolated_branch_us_per_layer": isolated[row["design_id"]],
            "unified_branch_us_per_layer": unified_us,
            "unified_minus_isolated_us_per_layer": unified_us - isolated[row["design_id"]],
            "unified_over_isolated": unified_us / isolated[row["design_id"]],
            **phase_us,
        }
        for name, expected_value in expected_values.items():
            if not math.isclose(float(row[name]), expected_value, rel_tol=0.0, abs_tol=1e-12):
                errors.append(f"pipeline derived value mismatch: {row['design_id']} {name}")
        branch_ms = unified_us * stage10.LAYERS / 1000.0
        local_attention_ms = float(metadata["baselines"]["local_gpu_attention_ms"])
        non_attention_ms = float(metadata["baselines"]["non_attention_ms"])
        expected_ideal = max(local_attention_ms, branch_ms) + non_attention_ms
        expected_serialized = local_attention_ms + branch_ms + non_attention_ms
        if not math.isclose(float(row["ideal_overlap_decode_ms"]), expected_ideal, rel_tol=0.0, abs_tol=1e-10):
            errors.append(f"ideal Decode arithmetic mismatch: {row['design_id']}")
        if not math.isclose(float(row["serialized_decode_ms"]), expected_serialized, rel_tol=0.0, abs_tol=1e-10):
            errors.append(f"serialized Decode arithmetic mismatch: {row['design_id']}")
        expected = stage10.BATCH_SIZE * 1000.0 / float(row["serialized_decode_ms"])
        if not math.isclose(expected, float(row["serialized_throughput_tokens_per_s"]), rel_tol=0.0, abs_tol=1e-10):
            errors.append(f"throughput arithmetic mismatch: {row['design_id']}")

    trials = precision["trials"]
    summaries = precision["summary"]
    expected_trials = (
        len(stage10.CONTEXT_LENGTHS)
        * len(stage10.TOTAL_PSEUDO_CHANNELS)
        * len(stage10.LOGIT_STDS)
        * len(stage10.SEEDS)
        * 3
    )
    if len(trials) != expected_trials or len(summaries) != len(stage10.TOTAL_PSEUDO_CHANNELS) * 3:
        errors.append("partial precision matrix mismatch")
    if summaries != stage10._precision_summary(trials):
        errors.append("partial precision summary derivation mismatch")
    trial_keys = {
        (
            row["context_length"],
            row["total_pseudo_channels"],
            row["logit_std"],
            row["seed"],
            row["scalar_format"],
        )
        for row in trials
    }
    if len(trial_keys) != expected_trials:
        errors.append("partial precision trials are duplicated")
    metric_names = (
        "max_abs_error",
        "mean_abs_error",
        "rmse",
        "relative_l2_error",
        "cosine_similarity",
        "vs_fp32_partial_max_abs_error",
        "vs_fp32_partial_relative_l2_error",
    )
    for row in trials:
        if not all(math.isfinite(float(row[name])) for name in metric_names):
            errors.append("partial precision metric is non-finite")
            break
        if row["scalar_format"] == "fp32" and (
            row["vs_fp32_partial_max_abs_error"] != 0.0
            or row["vs_fp32_partial_relative_l2_error"] != 0.0
        ):
            errors.append("FP32 incremental error must be zero")
            break

    validation = {
        "schema_version": 1,
        "stage": "kv-cxl-pim-pipeline-stage10-v1",
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "checks": {
            "boundary_designs": len(rows),
            "exact_runs": len(records),
            "interpolation": metadata["interpolation"],
            "precision_trials": len(trials),
            "precision_summary_rows": len(summaries),
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
    parser.add_argument("--output", default=str(stage10.DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    print(json.dumps(validate(args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
