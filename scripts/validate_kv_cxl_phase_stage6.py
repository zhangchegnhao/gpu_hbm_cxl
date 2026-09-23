#!/usr/bin/env python3
"""Audit phase-faithful Stage-6 request-level CXL results."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_kv_cxl_decode_stage5 as stage5


EXPECTED_CASE = "b8_c32k"
EXPECTED_POLICIES = {"gpu-only", "sieve"}
EXPECTED_SCALES = {4096, 1}


def validate(output_dir: str | Path) -> dict[str, object]:
    output = Path(output_dir)
    comparison_path = output / "comparison.json"
    manifest_path = output / "stage_manifest.json"
    layer_path = output / "layer_results.csv"
    raw = json.loads(comparison_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = raw["metadata"]
    rows = raw["rows"]
    if metadata["stage"] != "kv-cxl-phase-stage6-v1":
        raise ValueError("unexpected Stage-6 identifier")
    if metadata["case"] != EXPECTED_CASE:
        raise ValueError("unexpected Stage-6 case")
    if set(metadata["policies"]) != EXPECTED_POLICIES:
        raise ValueError("Stage-6 policy set is incomplete")
    if set(metadata["request_scales"]) != EXPECTED_SCALES:
        raise ValueError("Stage-6 scale set is incomplete")
    expected_shapes = 2 * len(EXPECTED_SCALES)
    if metadata["unique_shapes"] != expected_shapes or metadata["exact_runs"] != expected_shapes:
        raise ValueError("Stage-6 must contain two unique exact phase shapes per scale")
    if metadata["interpolation"] != 0 or manifest["interpolation"] != 0:
        raise ValueError("Stage-6 contains interpolation")
    expected_rows = 2 * len(EXPECTED_SCALES)
    if len(rows) != expected_rows or manifest["rows"] != expected_rows:
        raise ValueError("Stage-6 summary must contain two policy rows per scale")
    if metadata["execution"] != manifest["execution"]:
        raise ValueError("execution provenance mismatch")
    if stage5._execution_provenance() != metadata["execution"]:
        raise ValueError("current CXL binaries differ from Stage-6 provenance")

    expected = {(scale, policy) for scale in EXPECTED_SCALES for policy in EXPECTED_POLICIES}
    seen: set[tuple[int, str]] = set()
    for row in rows:
        key = (int(row["request_scale"]), row["policy"])
        if key in seen or key not in expected:
            raise ValueError(f"duplicate or unexpected row: {key}")
        seen.add(key)
        if row["capacity_state"] != "spill" or not row["capacity_feasible"]:
            raise ValueError(f"unexpected capacity state: {key}")
        if row["interpolation"] != 0:
            raise ValueError(f"row interpolation is nonzero: {key}")
        if row["attention_shape"]["cxl_kv_read_transactions"] <= 0:
            raise ValueError(f"missing CXL attention traffic: {key}")
        if row["expert_shape"]["cxl_kv_read_transactions"] != 0:
            raise ValueError(f"Expert phase contains CXL traffic: {key}")
        if row["expert_shape"]["local_kv_read_transactions"] != 0:
            raise ValueError(f"Expert phase contains Local KV traffic: {key}")
        if row["cxl_link_busy_ratio"] <= 0 or row["cxl_queue_wait_us_all_requests"] <= 0:
            raise ValueError(f"CXL attention phase did not exercise the queue: {key}")
        for phase_key in ("attention_cache_key", "expert_cache_key"):
            cache_path = output / "exact_cache" / f"{row[phase_key]}.json"
            if not cache_path.is_file():
                raise ValueError(f"missing cache for {key}: {cache_path.name}")
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            if cache.get("cache_input", {}).get("request_scale") != int(row["request_scale"]):
                raise ValueError(f"cache scale mismatch: {cache_path.name}")

    with layer_path.open(encoding="utf-8", newline="") as handle:
        layer_rows = list(csv.DictReader(handle))
    if len(layer_rows) != expected_rows * 48 or manifest["layer_rows"] != len(layer_rows):
        raise ValueError("Stage-6 layer row count mismatch")
    if set(row["request_scale"] for row in layer_rows) != {"4096", "1"}:
        raise ValueError("layer rows do not cover both scales")
    for name, path in manifest["output_hashes"].items():
        if name == "comparison.csv":
            actual = stage5._sha256(output / "comparison.csv")
        elif name == "comparison.json":
            actual = stage5._sha256(comparison_path)
        elif name == "layer_results.csv":
            actual = stage5._sha256(layer_path)
        elif name == "stage6_report.md":
            actual = stage5._sha256(output / "stage6_report.md")
        else:
            raise ValueError(f"unknown output hash: {name}")
        if actual != path:
            raise ValueError(f"output hash mismatch: {name}")

    result = {
        "status": "passed",
        "case": EXPECTED_CASE,
        "rows": len(rows),
        "layer_rows": len(layer_rows),
        "unique_shapes": metadata["unique_shapes"],
        "exact_runs": metadata["exact_runs"],
        "interpolation": 0,
        "request_scales": sorted(EXPECTED_SCALES, reverse=True),
        "policies": sorted(EXPECTED_POLICIES),
        "comparison_sha256": stage5._sha256(comparison_path),
        "stage_manifest_sha256": stage5._sha256(manifest_path),
    }
    (output / "validation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "results/kv_cxl_phase_stage6_b8_c32k_v1"))
    args = parser.parse_args(argv)
    print(json.dumps(validate(args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
