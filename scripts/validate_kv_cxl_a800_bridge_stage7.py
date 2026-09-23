#!/usr/bin/env python3
"""Validate Stage-7 A800/CXL bridge provenance and arithmetic."""

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

import run_kv_cxl_a800_bridge_stage7 as stage7
import run_kv_cxl_decode_stage5 as stage5


def validate(output_dir: str | Path = stage7.DEFAULT_OUTPUT) -> dict[str, Any]:
    output = Path(output_dir)
    manifest = json.loads((output / "stage_manifest.json").read_text(encoding="utf-8"))
    comparison = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    errors: list[str] = []

    for name, expected in manifest["output_hashes"].items():
        path = output / name
        if not path.is_file() or stage5._sha256(path) != expected:
            errors.append(f"output hash mismatch: {name}")
    _, _, stage6_attention_cache = stage7._load_stage6_spill()
    current_inputs = {
        "a800_summary": stage5._sha256(stage7.A800_SUMMARY),
        "stage6_comparison": stage5._sha256(stage7.STAGE6),
        "stage6_attention_cache": stage5._sha256(stage6_attention_cache),
        "capacity_source": stage5._sha256(stage5.CAPACITY),
        "cycle_config": stage5._sha256(stage5.CYCLE),
    }
    for name, current in current_inputs.items():
        if manifest["input_hashes"].get(name) != current:
            errors.append(f"input hash mismatch: {name}")

    metadata = comparison["metadata"]
    cache = metadata["exact_cache"]
    cache_path = output / "exact_cache" / f"{cache['key']}.json"
    if not cache_path.is_file() or stage5._sha256(cache_path) != cache["sha256"]:
        errors.append("resident exact cache hash mismatch")
    else:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        result = cached["result"]
        expected = int(cache["shape"]["local_kv_read_transactions"])
        if int(result["local_kv_completed_requests"]) != expected:
            errors.append("resident Local-HBM completion mismatch")
        if int(result["cxl_kv_completed_requests"]) != 0:
            errors.append("resident counterfactual contains CXL completions")

    rows = comparison["rows"]
    if len(rows) != 2 or {row["scenario"] for row in rows} != {
        "resident-local-counterfactual",
        "nominal-memory-only-cxl-spill",
    }:
        errors.append("scenario matrix mismatch")
    else:
        resident = next(row for row in rows if row["scenario"].startswith("resident"))
        spill = next(row for row in rows if row["scenario"].startswith("nominal"))
        delta = spill["estimated_decode_ms"] - resident["estimated_decode_ms"]
        expected_delta = comparison["derived"]["nominal_cxl_memory_penalty_ms_per_decode"]
        if not math.isclose(delta, expected_delta, rel_tol=0.0, abs_tol=1e-9):
            errors.append("Decode delta arithmetic mismatch")
        if not math.isclose(
            spill["estimated_attention_ms"] - resident["estimated_attention_ms"],
            expected_delta,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            errors.append("Attention delta arithmetic mismatch")

    if metadata["interpolation"] != 0 or manifest["interpolation"] != 0:
        errors.append("Stage 7 must not interpolate Ramulator results")
    if metadata["exact_new_runs"] != 1 or manifest["exact_new_runs"] != 1:
        errors.append("Stage 7 exact-run count mismatch")
    if comparison["calibration"]["forward_holdout"]["absolute_percentage_error"] >= 0.05:
        errors.append("A800 forward holdout error exceeds the recorded 5% gate")

    validation = {
        "schema_version": 1,
        "stage": "kv-cxl-a800-bridge-stage7-v1",
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "checks": {
            "rows": len(rows),
            "exact_new_runs": metadata["exact_new_runs"],
            "interpolation": metadata["interpolation"],
            "forward_holdout_absolute_percentage_error": comparison["calibration"]["forward_holdout"]["absolute_percentage_error"],
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
    parser.add_argument("--output", default=str(stage7.DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    print(json.dumps(validate(args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
