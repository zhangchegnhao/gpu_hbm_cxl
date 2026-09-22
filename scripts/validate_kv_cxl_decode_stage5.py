#!/usr/bin/env python3
"""Audit Stage-5 request-level CXL Decode graph results."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sieve_replay.report.writer import sha256_file


RAMULATOR = ROOT / "third_party/ramulator2_cxl"
EXPECTED_CASES = {"b8_c32k", "b16_c16k", "b16_c32k", "b32_c16k"}
EXPECTED_POLICIES = {"gpu-only", "sieve"}


def _execution_provenance() -> dict[str, str]:
    revision = subprocess.check_output(
        ["git", "-C", str(RAMULATOR), "rev-parse", "HEAD"], text=True
    ).strip()
    bindings = sorted((RAMULATOR / "python/ramulator").glob("_ramulator.cpython-*.so"))
    if len(bindings) != 1:
        raise ValueError("expected exactly one CXL Ramulator Python binding")
    library = RAMULATOR / "libramulator.so"
    if not library.is_file():
        raise ValueError("CXL Ramulator shared library is missing")
    return {
        "ramulator_commit": revision,
        "binding_sha256": sha256_file(bindings[0]),
        "ramulator_library_sha256": sha256_file(library),
    }


def validate(
    output_dir: str | Path,
    *,
    request_scale: int = 4096,
    expected_cases: set[str] | None = None,
    expected_policies: set[str] | None = None,
) -> dict[str, object]:
    output = Path(output_dir)
    comparison_path = output / "comparison.json"
    layer_path = output / "layer_results.csv"
    manifest_path = output / "stage_manifest.json"
    raw = json.loads(comparison_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = raw["metadata"]
    rows = raw["rows"]
    if metadata["stage"] != "kv-cxl-decode-stage5-v1":
        raise ValueError("unexpected Stage-5 identifier")
    cases = expected_cases or EXPECTED_CASES
    policies = expected_policies or EXPECTED_POLICIES
    if metadata["request_scale"] != request_scale:
        raise ValueError(f"Stage-5 audit expects request_scale={request_scale}")
    if metadata["interpolation"] != 0 or manifest["interpolation"] != 0:
        raise ValueError("Stage-5 results must not contain interpolation")
    if set(metadata["cases"]) != cases or set(metadata["policies"]) != policies:
        raise ValueError("Stage-5 matrix is incomplete")
    expected_rows = len(cases) * len(policies)
    if len(rows) != expected_rows or metadata["rows"] != expected_rows or manifest["rows"] != expected_rows:
        raise ValueError(f"Stage-5 summary matrix must contain {expected_rows} rows")
    if metadata.get("execution") != manifest.get("execution"):
        raise ValueError("execution provenance differs between metadata and manifest")
    if metadata["execution"] != _execution_provenance():
        raise ValueError("Stage-5 CXL binary provenance differs from the current build")

    expected_keys = {(case, policy) for case in cases for policy in policies}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["case"], row["policy"])
        if key in seen or key not in expected_keys:
            raise ValueError(f"duplicate or unexpected summary row: {key}")
        seen.add(key)
        if "exact request-level CXL memory-only timing" not in row["classification"]:
            raise ValueError(f"classification mismatch: {key}")
        if not row["capacity_feasible"] or row["capacity_state"] != "spill":
            raise ValueError(f"A800+CXL capacity state is not spill: {key}")
        if row["exact_runs"] != 1 or row["interpolation"] != 0:
            raise ValueError(f"exact-run provenance mismatch: {key}")
        if row["cxl_kv_read_transactions_per_layer"] <= 0:
            raise ValueError(f"CXL traffic missing: {key}")
        if row["cxl_queue_wait_us_all_requests"] <= 0 or row["cxl_link_busy_ratio"] <= 0:
            raise ValueError(f"CXL queue/link was not exercised: {key}")
        cache_path = output / "exact_cache" / f"{row['cache_key']}.json"
        if not cache_path.is_file():
            raise ValueError(f"missing exact Stage-5 cache: {cache_path.name}")
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        if cache.get("cache_input", {}).get("request_scale") != request_scale:
            raise ValueError(f"cache request scale mismatch: {cache_path.name}")
        if cache.get("result", {}).get("cxl_kv_completed_requests") != row["cxl_kv_read_transactions_per_layer"]:
            raise ValueError(f"cache CXL completion mismatch: {key}")

    with layer_path.open(encoding="utf-8", newline="") as handle:
        header = handle.readline().strip().split(",")
        layer_rows = [line for line in handle if line.strip()]
    expected_layer_rows = expected_rows * 48
    if len(layer_rows) != expected_layer_rows:
        raise ValueError(f"expected {expected_layer_rows} layer rows, got {len(layer_rows)}")
    if "cxl_queue_wait_us_all_requests" not in header or "cxl_critical_path_us" not in header:
        raise ValueError("layer result accounting columns are missing")
    if manifest["layer_rows"] != len(layer_rows):
        raise ValueError("manifest layer row count mismatch")
    if manifest["output_hashes"]["comparison.csv"] != sha256_file(output / "comparison.csv"):
        raise ValueError("comparison.csv hash mismatch")
    if manifest["output_hashes"]["comparison.json"] != sha256_file(comparison_path):
        raise ValueError("comparison.json hash mismatch")
    if manifest["output_hashes"]["layer_results.csv"] != sha256_file(layer_path):
        raise ValueError("layer_results.csv hash mismatch")
    if manifest["output_hashes"]["stage5_report.md"] != sha256_file(output / "stage5_report.md"):
        raise ValueError("stage5_report.md hash mismatch")

    result = {
        "status": "passed",
        "rows": len(rows),
        "layer_rows": len(layer_rows),
        "cases": sorted(cases),
        "policies": sorted(policies),
        "request_scale": metadata["request_scale"],
        "exact_runs": metadata["exact_runs"],
        "interpolation": 0,
        "comparison_sha256": sha256_file(comparison_path),
        "stage_manifest_sha256": sha256_file(manifest_path),
    }
    (output / "validation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "results/kv_cxl_decode_stage5_v1"))
    parser.add_argument("--request-scale", type=int, default=4096)
    parser.add_argument("--cases", default=",".join(sorted(EXPECTED_CASES)))
    parser.add_argument("--policies", default=",".join(sorted(EXPECTED_POLICIES)))
    args = parser.parse_args(argv)
    cases = {value for value in args.cases.split(",") if value}
    policies = {value for value in args.policies.split(",") if value}
    print(json.dumps(validate(args.output, request_scale=args.request_scale, expected_cases=cases, expected_policies=policies), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
