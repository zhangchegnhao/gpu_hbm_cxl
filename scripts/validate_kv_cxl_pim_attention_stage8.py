#!/usr/bin/env python3
"""Validate Stage-8 CXL-PIM Attention results and provenance."""

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
import run_kv_cxl_pim_attention_stage8 as stage8


def validate(output_dir: str | Path = stage8.DEFAULT_OUTPUT) -> dict[str, Any]:
    output = Path(output_dir)
    manifest = json.loads((output / "stage_manifest.json").read_text(encoding="utf-8"))
    comparison = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    errors: list[str] = []

    for name, expected in manifest["output_hashes"].items():
        path = output / name
        if not path.is_file() or stage5._sha256(path) != expected:
            errors.append(f"output hash mismatch: {name}")
    current_inputs = {
        "stage7": stage5._sha256(stage8.STAGE7),
        "stage7_validation": stage5._sha256(stage8.STAGE7_VALIDATION),
        "model": stage5._sha256(ROOT / "configs/models/qwen3_30b_a3b.json"),
        "cycle": stage5._sha256(stage5.CYCLE),
    }
    for name, current in current_inputs.items():
        if manifest["input_hashes"].get(name) != current:
            errors.append(f"input hash mismatch: {name}")

    records = manifest["cache_records"]
    expected_cxl_context = stage8.cxl_kv_cache_context(ROOT, stage5.CYCLE)
    expected_pim_context = stage8.cxl_pim_attention_context(ROOT, stage5.CYCLE)
    if len(records) != 5 or len({record["kind"] for record in records}) != 5:
        errors.append("exact cache matrix mismatch")
    for record in records:
        path = output / "exact_cache" / f"{record['key']}.json"
        if not path.is_file() or stage5._sha256(path) != record["sha256"]:
            errors.append(f"cache hash mismatch: {record['kind']}")
            continue
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached["cache_input"].get("cxl_context") != expected_cxl_context:
            errors.append(f"CXL source context mismatch: {record['kind']}")
        if cached["cache_input"].get("cxl_pim_context") != expected_pim_context:
            errors.append(f"CXL-PIM source context mismatch: {record['kind']}")
        result = cached["result"]
        payload = record["payload"]
        if cached["result_type"] == "cxl-transfer":
            if int(result["cxl_kv_completed_requests"]) != int(payload["transactions"]):
                errors.append(f"transfer completion mismatch: {record['kind']}")
        else:
            expected = int(payload["waves"]) * stage8.TOPOLOGY.total_pseudo_channels
            if int(result["completed_requests"]) != expected:
                errors.append(f"PIM completion mismatch: {record['kind']}")

    metadata = comparison["metadata"]
    rows = comparison["rows"]
    if metadata["exact_runs"] != 5 or manifest["exact_runs"] != 5:
        errors.append("exact-run count mismatch")
    if metadata["interpolation"] != 0 or manifest["interpolation"] != 0:
        errors.append("Stage 8 must not interpolate")
    scenarios = {row["scenario"] for row in rows}
    if len(rows) != 4 or scenarios != {
        "resident-local-counterfactual",
        "nominal-memory-only-cxl-spill",
        "optimistic-cxl-pim-attention",
        "serialized-cxl-pim-attention",
    }:
        errors.append("comparison row matrix mismatch")
    else:
        for row in rows:
            expected_throughput = stage8.BATCH_SIZE * 1000.0 / row["estimated_decode_ms"]
            if not math.isclose(
                row["estimated_throughput_tokens_per_s"],
                expected_throughput,
                rel_tol=0.0,
                abs_tol=1e-10,
            ):
                errors.append(f"throughput arithmetic mismatch: {row['scenario']}")
    derived = comparison["derived"]
    if not math.isclose(
        derived["cxl_pim_branch_us_per_layer"],
        sum(metadata["component_us_per_layer"].values()),
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        errors.append("CXL-PIM component sum mismatch")
    if not 0.0 < derived["link_traffic_reduction_fraction"] < 1.0:
        errors.append("CXL-PIM link reduction is outside (0,1)")

    validation = {
        "schema_version": 1,
        "stage": "kv-cxl-pim-attention-stage8-v1",
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "checks": {
            "rows": len(rows),
            "exact_runs": metadata["exact_runs"],
            "interpolation": metadata["interpolation"],
            "cache_records": len(records),
            "link_traffic_reduction_fraction": derived["link_traffic_reduction_fraction"],
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
    parser.add_argument("--output", default=str(stage8.DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    print(json.dumps(validate(args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
