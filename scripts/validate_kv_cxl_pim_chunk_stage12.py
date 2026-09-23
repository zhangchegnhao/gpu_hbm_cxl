#!/usr/bin/env python3
"""Independently validate Stage-12 chunk timing, provenance, and Decode math."""

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
import run_kv_cxl_pim_chunk_stage12 as stage12
import run_kv_cxl_pim_decode_stage11 as stage11
from sieve_replay.ramulator.cxl_pim_chunk_pipeline import CXLPIMChunkPipelineResult


def _close(left: float, right: float, tolerance: float = 1e-8) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(output_dir: str | Path = stage12.DEFAULT_OUTPUT) -> dict[str, Any]:
    output = Path(output_dir)
    comparison_path = output / "comparison.json"
    exact_path = output / "exact_workloads.json"
    layer_path = output / "layer_results.csv"
    manifest_path = output / "stage_manifest.json"
    comparison = _read_json(comparison_path)
    exact = _read_json(exact_path)
    manifest = _read_json(manifest_path)
    metadata = comparison["metadata"]
    rows = comparison["rows"]
    exact_rows = exact["rows"]
    errors: list[str] = []

    expected_stage = "kv-cxl-pim-chunk-stage12-v1"
    if metadata.get("stage") != expected_stage or manifest.get("stage") != expected_stage:
        errors.append("unexpected Stage-12 identifier")
    if exact.get("metadata") != metadata:
        errors.append("exact-workload metadata differs from comparison metadata")
    for container, label in ((metadata, "metadata"), (manifest, "manifest")):
        if container.get("interpolation") != 0:
            errors.append(f"{label} interpolation must be zero")
        if container.get("failed_runs") != 0 or container.get("remaining_runs") != 0:
            errors.append(f"{label} records incomplete exact runs")
        if container.get("exact_runs") != 9:
            errors.append(f"{label} exact-run count is not nine")

    configuration, _ = stage5._source_configuration()
    cycle = stage6.SieveCycleV1Config.load(stage5.CYCLE)
    current_inputs = stage12._input_hashes(configuration)
    current_execution = stage5._execution_provenance()
    if metadata.get("input_hashes") != current_inputs:
        errors.append("current Stage-12 inputs differ from recorded hashes")
    if manifest.get("input_hashes") != current_inputs:
        errors.append("manifest input hashes mismatch")
    if metadata.get("execution") != current_execution:
        errors.append("current Ramulator binary provenance differs from metadata")
    if manifest.get("execution") != current_execution:
        errors.append("manifest Ramulator binary provenance mismatch")

    designs = stage12._selected_designs(configuration, cycle)
    stage10_source = stage11._load_stage10()
    expected_keys = {
        (design_id, config_id)
        for design_id in stage12.DESIGN_IDS
        for config_id, _, _ in stage12.PIPELINE_CONFIGS
    }
    config_map = {
        config_id: (chunk_count, buffer_slots)
        for config_id, chunk_count, buffer_slots in stage12.PIPELINE_CONFIGS
    }
    records = manifest.get("cache_records", [])
    record_map = {
        (record["design_id"], record["config_id"]): record for record in records
    }
    if len(records) != 9 or set(record_map) != expected_keys:
        errors.append("cache-record design/config matrix is incomplete")
    if metadata.get("cache_records") != records:
        errors.append("comparison and manifest cache records differ")

    exact_map = {
        (row["design_id"], row["config_id"]): row for row in exact_rows
    }
    if len(exact_rows) != 9 or set(exact_map) != expected_keys:
        errors.append("exact-workload design/config matrix is incomplete")

    cached_results: dict[tuple[str, str], CXLPIMChunkPipelineResult] = {}
    identity_checks: dict[str, dict[str, Any]] = {}
    for key in sorted(expected_keys):
        design_id, config_id = key
        record = record_map.get(key)
        row = exact_map.get(key)
        if record is None or row is None:
            continue
        cache_path = ROOT / record["cache_path"]
        if not cache_path.is_file():
            errors.append(f"missing exact cache: {key}")
            continue
        if stage5._sha256(cache_path) != record["sha256"]:
            errors.append(f"exact cache SHA-256 mismatch: {key}")
            continue
        cached = _read_json(cache_path)
        cache_input = cached.get("cache_input", {})
        if cache_input != record.get("cache_input"):
            errors.append(f"cache identity differs from manifest: {key}")
        if stage12._key(cache_input) != record.get("key"):
            errors.append(f"cache key cannot be reconstructed: {key}")
        if cache_input.get("execution") != current_execution:
            errors.append(f"execution-time binary provenance mismatch: {key}")
        if cache_input.get("input_hashes") != current_inputs:
            errors.append(f"cache input hashes mismatch: {key}")

        result = CXLPIMChunkPipelineResult.from_dict(cached["result"])
        cached_results[key] = result
        chunk_count, buffer_slots = config_map[config_id]
        design = designs[design_id]
        shape = design["shape"]
        endpoints = design["topology"].total_pseudo_channels
        expected_counts = {
            "query_link": int(shape.query_link_transactions),
            "pim_gwrite": int(shape.pim_gwrite_waves) * endpoints,
            "pim_mac": int(shape.pim_mac_waves) * endpoints,
            "pim_read": chunk_count * int(shape.pim_read_waves) * endpoints,
            "result_link": chunk_count * int(shape.result_link_transactions),
        }
        if result.chunk_count != chunk_count or result.buffer_slots != buffer_slots:
            errors.append(f"chunk/buffer configuration mismatch: {key}")
        for phase, expected in expected_counts.items():
            for suffix in ("injected_requests", "completed_requests"):
                if getattr(result, f"{phase}_{suffix}") != expected:
                    errors.append(f"{phase} {suffix} mismatch: {key}")
        controller_counts = {
            "pim_gwrite": result.controller_pim_gwrite_requests,
            "pim_mac": result.controller_pim_mac_requests,
            "pim_read": result.controller_pim_read_requests,
            "result_link": result.controller_link_requests
            - result.query_link_completed_requests,
        }
        for phase, actual in controller_counts.items():
            if actual != expected_counts[phase]:
                errors.append(f"controller {phase} count mismatch: {key}")
        if result.max_active_buffers > buffer_slots or result.max_active_buffers < 1:
            errors.append(f"buffer-slot bound violated: {key}")
        if buffer_slots == 2 and result.max_active_buffers != 2:
            errors.append(f"double buffering was never active: {key}")
        if len(result.partial_ready_us) != chunk_count:
            errors.append(f"partial-ready vector length mismatch: {key}")
        if tuple(sorted(result.partial_ready_us)) != result.partial_ready_us:
            errors.append(f"partial-ready milestones are unordered: {key}")
        if result.chunk_result_link_completion_cycles[-1] != result.pipeline_completion_cycles:
            errors.append(f"last partial does not finish pipeline: {key}")
        if sum(result.chunk_mac_waves) != int(shape.pim_mac_waves):
            errors.append(f"chunk MAC waves do not preserve workload: {key}")
        if int(row["pipeline_completion_cycles"]) != result.pipeline_completion_cycles:
            errors.append(f"exact summary completion mismatch: {key}")
        if not _close(row["pipeline_completion_us"], result.pipeline_completion_us, 1e-12):
            errors.append(f"exact summary time mismatch: {key}")
        if int(row["max_active_buffers"]) != result.max_active_buffers:
            errors.append(f"exact summary buffer count mismatch: {key}")
        if int(row["interpolation"]) != 0:
            errors.append(f"exact row interpolation is nonzero: {key}")

        if chunk_count == 1:
            identity = stage12._stage10_identity(
                result, stage10_source["results"][design_id]
            )
            identity_checks[design_id] = identity
            if not identity["passed"] or identity["fields_checked"] != 10:
                errors.append(f"one-chunk Stage-10 identity failed: {design_id}")

    if metadata.get("one_chunk_identity") != identity_checks:
        errors.append("metadata one-chunk identity record mismatch")
    if manifest.get("one_chunk_identity") != identity_checks:
        errors.append("manifest one-chunk identity record mismatch")

    stage11_source = stage12._load_stage11()
    summary_map = {
        (row["design_id"], row["config_id"]): row for row in rows
    }
    if len(rows) != 9 or set(summary_map) != expected_keys:
        errors.append("Decode comparison design/config matrix is incomplete")
    expected_memory = stage11._load_stage7()["raw"]["metadata"]["capacity"]
    expected_expert_us = float(stage11._load_stage6()["row"]["expert_critical_path_us"])
    for key in sorted(expected_keys):
        row = summary_map.get(key)
        result = cached_results.get(key)
        if row is None or result is None:
            continue
        design_id, config_id = key
        chunk_count, buffer_slots = config_map[config_id]
        merge = stage12._merge_model(configuration, designs[design_id])
        if metadata.get("merge_models", {}).get(design_id) != merge:
            errors.append(f"merge-model provenance mismatch: {design_id}")
        if int(row["chunk_count"]) != chunk_count or int(row["buffer_slots"]) != buffer_slots:
            errors.append(f"Decode row chunk/buffer mismatch: {key}")
        if row["policy"] != stage12.POLICY or int(row["layer_count"]) != stage12.LAYERS:
            errors.append(f"Decode row policy/layer mismatch: {key}")
        if row["interpolation"] != 0:
            errors.append(f"Decode row interpolation is nonzero: {key}")
        if int(row["kv_cache_bytes"]) != int(expected_memory["kv_cache_bytes"]):
            errors.append(f"KV byte mismatch: {key}")
        if int(row["peak_memory_bytes"]) != int(expected_memory["peak_memory_bytes"]):
            errors.append(f"peak-memory mismatch: {key}")
        if row["a800_capacity_state"] != "spill" or not row["a800_capacity_feasible"]:
            errors.append(f"A800 capacity state mismatch: {key}")
        if row["simulated_96gb_capacity_state"] != "resident":
            errors.append(f"simulated capacity state mismatch: {key}")
        if not _close(row["cxl_pim_pipeline_us_per_layer"], result.pipeline_completion_us, 1e-12):
            errors.append(f"pipeline timing mismatch: {key}")
        if not _close(row["gpu_merge_us_per_chunk"], merge["duration_us_per_chunk"], 1e-12):
            errors.append(f"merge timing mismatch: {key}")
        if not _close(
            row["gpu_merge_us_per_layer"],
            merge["duration_us_per_chunk"] * chunk_count,
            1e-12,
        ):
            errors.append(f"merge total mismatch: {key}")
        if not _close(row["expert_latency_us"], expected_expert_us):
            errors.append(f"Stage-6 Expert timing mismatch: {key}")
        total = float(row["total_latency_us"])
        if not _close(
            row["throughput_tokens_per_s"],
            stage12.BATCH_SIZE * 1_000_000.0 / total,
            1e-10,
        ):
            errors.append(f"throughput arithmetic mismatch: {key}")
        if not _close(row["attention_latency_share"], row["attention_latency_us"] / total, 1e-12):
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
        references = stage11_source["references"][design_id]
        reference_fields = {
            "stage11_ideal_overlap_total_us": references["ideal-overlap-bound"]["total_latency_us"],
            "stage11_fully_serialized_total_us": references["fully-serialized-bound"]["total_latency_us"],
            "stage11_memory_only_total_us": stage11_source["memory"]["total_latency_us"],
        }
        for name, expected in reference_fields.items():
            if not _close(row[name], expected, 1e-12):
                errors.append(f"Stage-11 reference mismatch for {name}: {key}")
        expected_link_bytes = int(designs[design_id]["shape"].query_bytes) + chunk_count * int(
            designs[design_id]["shape"].partial_result_bytes
        )
        if int(row["total_cxl_link_bytes_per_layer"]) != expected_link_bytes:
            errors.append(f"link-byte accounting mismatch: {key}")

    with layer_path.open(encoding="utf-8", newline="") as handle:
        layer_rows = list(csv.DictReader(handle))
    expected_layer_rows = len(expected_keys) * stage12.LAYERS
    if len(layer_rows) != expected_layer_rows:
        errors.append("Stage-12 layer row count mismatch")
    if metadata.get("layer_rows") != len(layer_rows) or manifest.get("layer_rows") != len(layer_rows):
        errors.append("recorded layer row count mismatch")
    by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in layer_rows:
        key = (row["design_id"], row["config_id"])
        by_key.setdefault(key, []).append(row)
        if int(row["interpolation"]) != 0:
            errors.append(f"layer interpolation is nonzero: {key}")
    for key in expected_keys:
        group = by_key.get(key, [])
        if len(group) != stage12.LAYERS or {int(row["layer"]) for row in group} != set(
            range(stage12.LAYERS)
        ):
            errors.append(f"layer coverage mismatch: {key}")
            continue
        layer_total = sum(float(row["layer_span_us"]) for row in group)
        if not _close(layer_total, summary_map[key]["total_latency_us"]):
            errors.append(f"layer total mismatch: {key}")
        result = cached_results[key]
        if any(
            not _close(row["cxl_pim_pipeline_us"], result.pipeline_completion_us, 1e-9)
            for row in group
        ):
            errors.append(f"per-layer pipeline timing mismatch: {key}")
        if any(
            not _close(row["last_partial_ready_offset_us"], result.partial_ready_us[-1], 1e-9)
            for row in group
        ):
            errors.append(f"per-layer partial-ready milestone mismatch: {key}")

    for name, expected in manifest.get("output_hashes", {}).items():
        path = output / name
        if not path.is_file() or stage5._sha256(path) != expected:
            errors.append(f"output hash mismatch: {name}")

    validation = {
        "schema_version": 1,
        "stage": expected_stage,
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "checks": {
            "exact_runs": len(exact_rows),
            "one_chunk_identity_designs": len(identity_checks),
            "one_chunk_identity_fields_per_design": 10,
            "rows": len(rows),
            "layer_rows": len(layer_rows),
            "interpolation": metadata.get("interpolation"),
            "binary_provenance_bound": metadata.get("execution") == current_execution,
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
    parser.add_argument("--output", default=str(stage12.DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    print(json.dumps(validate(args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
