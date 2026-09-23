#!/usr/bin/env python3
"""Validate Stage-9 CXL-PIM sensitivity results and exact component caches."""

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
import run_kv_cxl_pim_sensitivity_stage9 as stage9


def validate(output_dir: str | Path = stage9.DEFAULT_OUTPUT) -> dict[str, Any]:
    output = Path(output_dir)
    manifest = json.loads((output / "stage_manifest.json").read_text(encoding="utf-8"))
    sensitivity = json.loads((output / "sensitivity.json").read_text(encoding="utf-8"))
    errors: list[str] = []

    for name, expected in manifest["output_hashes"].items():
        path = output / name
        if not path.is_file() or stage5._sha256(path) != expected:
            errors.append(f"output hash mismatch: {name}")
    current_inputs = {
        "stage8": stage5._sha256(stage9.STAGE8),
        "stage8_validation": stage5._sha256(stage9.STAGE8_VALIDATION),
        "model": stage5._sha256(ROOT / "configs/models/qwen3_30b_a3b.json"),
        "cycle": stage5._sha256(stage5.CYCLE),
    }
    for name, current in current_inputs.items():
        if manifest["input_hashes"].get(name) != current:
            errors.append(f"input hash mismatch: {name}")

    metadata = sensitivity["metadata"]
    records = manifest["cache_records"]
    expected_new_components = 18
    if len(records) != 23 or len({record["key"] for record in records}) != 23:
        errors.append("exact component matrix mismatch")
    if sum(record["source"] == "stage8-reuse" for record in records) != 5:
        errors.append("Stage-8 reuse count mismatch")
    if sum(record["source"] == "stage9-exact-cache" for record in records) != expected_new_components:
        errors.append("Stage-9 exact cache count mismatch")
    if metadata["cache_records"] != records:
        errors.append("manifest and sensitivity cache records disagree")
    if manifest["execution"] != metadata["execution"]:
        errors.append("manifest and sensitivity execution provenance disagree")

    stage8 = json.loads(stage9.STAGE8.read_text(encoding="utf-8"))
    cycle = stage9.SieveCycleV1Config.load(stage5.CYCLE)
    stage8_index = stage9._stage8_cache_index(stage8, cycle)
    for record in records:
        path = ROOT / record["cache_path"]
        if not path.is_file() or stage5._sha256(path) != record["sha256"]:
            errors.append(f"cache hash mismatch: {record['key']}")
            continue
        cached = json.loads(path.read_text(encoding="utf-8"))
        result = cached["result"]
        identity = record["simulation_identity"]
        if record["source"] == "stage8-reuse":
            source = stage8_index.get(record["key"])
            if source is None or source["path"].resolve() != path.resolve():
                errors.append(f"Stage-8 cache identity mismatch: {record['key']}")
        elif record["source"] == "stage9-exact-cache":
            expected_cache_input = {
                "schema_version": 1,
                "stage": "kv-cxl-pim-sensitivity-stage9-v1",
                "simulation_identity": identity,
            }
            if cached.get("cache_input") != expected_cache_input:
                errors.append(f"Stage-9 cache identity mismatch: {record['key']}")
        else:
            errors.append(f"unknown cache source: {record['key']}")
        if identity["kind"] == "cxl-fifo-transfer-read-proxy":
            if int(result["cxl_kv_completed_requests"]) != int(identity["transactions"]):
                errors.append(f"transfer completion mismatch: {record['key']}")
        else:
            topology = identity["topology"]
            expected = int(identity["waves"]) * int(topology["channels"]) * int(topology["pseudo_channels_per_channel"])
            if int(result["completed_requests"]) != expected:
                errors.append(f"PIM completion mismatch: {record['key']}")

    rows = sensitivity["rows"]
    designs = sensitivity["design_summary"]
    expected_designs = len(stage9.CHANNELS) * len(stage9.MAC_INTERVALS_PS) * len(stage9.PARTIAL_SCALAR_BYTES)
    expected_rows = expected_designs * len(stage9.OVERLAP_FRACTIONS) * len(stage9.MERGE_US_PER_LAYER)
    if len(designs) != expected_designs or len(rows) != expected_rows:
        errors.append("factorial row count mismatch")
    if metadata["interpolation"] != 0 or manifest["interpolation"] != 0:
        errors.append("Stage 9 must not interpolate")
    for row in rows:
        expected = stage9.BATCH_SIZE * 1000.0 / float(row["estimated_decode_ms"])
        if not math.isclose(expected, float(row["estimated_throughput_tokens_per_s"]), rel_tol=0.0, abs_tol=1e-10):
            errors.append(f"throughput arithmetic mismatch: {row['design_id']}")
            break

    baseline_ideal = next(row for row in rows if row["channels"] == 4 and row["mac_interval_ps"] == 24576 and row["partial_scalar_bytes"] == 4 and row["overlap_fraction"] == 1.0 and row["merge_us_per_layer"] == 0.0)
    baseline_serial = next(row for row in rows if row["channels"] == 4 and row["mac_interval_ps"] == 24576 and row["partial_scalar_bytes"] == 4 and row["overlap_fraction"] == 0.0 and row["merge_us_per_layer"] == 0.0)
    stage8_ideal = next(row for row in stage8["rows"] if row["scenario"] == "optimistic-cxl-pim-attention")
    stage8_serial = next(row for row in stage8["rows"] if row["scenario"] == "serialized-cxl-pim-attention")
    for label, actual, expected in (
        ("ideal", baseline_ideal, stage8_ideal),
        ("serialized", baseline_serial, stage8_serial),
    ):
        if not math.isclose(float(actual["estimated_decode_ms"]), float(expected["estimated_decode_ms"]), rel_tol=0.0, abs_tol=1e-9):
            errors.append(f"Stage-8 {label} baseline reproduction mismatch")

    validation = {
        "schema_version": 1,
        "stage": "kv-cxl-pim-sensitivity-stage9-v1",
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "checks": {
            "design_points": len(designs),
            "rows": len(rows),
            "exact_components": len(records),
            "reused_stage8_components": sum(record["source"] == "stage8-reuse" for record in records),
            "new_exact_components": sum(record["source"] == "stage9-exact-cache" for record in records),
            "interpolation": metadata["interpolation"],
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
    parser.add_argument("--output", default=str(stage9.DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    print(json.dumps(validate(args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
