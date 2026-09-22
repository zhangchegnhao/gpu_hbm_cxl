#!/usr/bin/env python3
"""Audit exact Stage-4 CXL request-level results."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sieve_replay.report.writer import sha256_file


EXPECTED_CASES = {"b8_c32k", "b16_c16k", "b16_c32k", "b32_c16k"}
EXPECTED_MODES = {
    "expert-only",
    "local-kv-only",
    "cxl-kv-only",
    "local-kv-plus-expert",
    "cxl-kv-plus-local-kv",
    "cxl-kv-plus-local-kv-plus-expert",
}
EXPECTED_PROFILES = {
    "conservative-assumption",
    "nominal-assumption",
    "optimistic-assumption",
}
RAMULATOR = ROOT / "third_party/ramulator2_cxl"


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


def validate(output_dir: str | Path) -> dict[str, object]:
    output = Path(output_dir)
    comparison_path = output / "comparison.json"
    manifest_path = output / "stage_manifest.json"
    raw = json.loads(comparison_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = raw["metadata"]
    rows = raw["rows"]
    execution = metadata.get("execution")
    if not isinstance(execution, dict) or len(execution.get("binding_sha256", "")) != 64:
        raise ValueError("Stage-4 result is missing CXL binary provenance")
    if manifest.get("execution") != execution:
        raise ValueError("Stage-4 manifest execution provenance differs from result metadata")
    if _execution_provenance() != execution:
        raise ValueError("Stage-4 CXL binary provenance differs from the current build")
    if metadata["request_scale"] != 4096:
        raise ValueError("this audit expects the recorded Stage-4 microbenchmark scale=4096")
    if metadata["interpolation"] != 0 or manifest["interpolation"] != 0:
        raise ValueError("Stage-4 results must not contain interpolation")
    if len(rows) != 72 or metadata["rows"] != 72 or manifest["rows"] != 72:
        raise ValueError("Stage-4 result matrix must contain exactly 72 rows")
    if set(metadata["profiles"]) != EXPECTED_PROFILES:
        raise ValueError("Stage-4 profile set is incomplete")

    seen: set[tuple[str, str, str]] = set()
    grouped: dict[tuple[str, str], dict[str, dict[str, object]]] = defaultdict(dict)
    cache_dir = output / "exact_cache"
    for row in rows:
        key = (row["case"], row["profile"], row["mode"])
        if key in seen:
            raise ValueError(f"duplicate result row: {key}")
        seen.add(key)
        if row["case"] not in EXPECTED_CASES or row["mode"] not in EXPECTED_MODES or row["profile"] not in EXPECTED_PROFILES:
            raise ValueError(f"unexpected result key: {key}")
        if row["classification"] != "Ramulator request-level memory-only CXL sensitivity; no A800 timing":
            raise ValueError("result classification changed")
        if row["cxl_kv_completed_requests"] != row["cxl_kv_read_transactions"]:
            raise ValueError(f"CXL completion mismatch: {key}")
        if row["gpu_completed_requests"] != row["gpu_read_transactions"]:
            raise ValueError(f"GPU completion mismatch: {key}")
        if row["local_kv_completed_requests"] != row["local_kv_read_transactions"]:
            raise ValueError(f"local KV completion mismatch: {key}")
        if row["pim_completed_requests"] != (
            row["pim_gwrite_waves"] + row["pim_mac_waves"] + row["pim_read_waves"]
        ) * 256:
            raise ValueError(f"PIM completion mismatch: {key}")
        cache_path = cache_dir / f"{row['cache_key']}.json"
        if not cache_path.is_file():
            raise ValueError(f"missing exact cache entry: {cache_path.name}")
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        if cache.get("cache_input", {}).get("request_scale") != 4096:
            raise ValueError(f"cache scale provenance mismatch: {cache_path.name}")
        if cache.get("result", {}).get("cxl_kv_completed_requests") != row["cxl_kv_completed_requests"]:
            raise ValueError(f"cache/result mismatch: {cache_path.name}")
        grouped[(row["case"], row["profile"])][row["mode"]] = row

    for group, modes in grouped.items():
        if set(modes) != EXPECTED_MODES:
            raise ValueError(f"incomplete mode group: {group}")
        expert = modes["expert-only"]
        local = modes["local-kv-only"]
        cxl = modes["cxl-kv-only"]
        local_expert = modes["local-kv-plus-expert"]
        cxl_local = modes["cxl-kv-plus-local-kv"]
        combined = modes["cxl-kv-plus-local-kv-plus-expert"]
        if expert["cxl_kv_read_transactions"] != 0 or local["cxl_kv_read_transactions"] != 0:
            raise ValueError(f"isolated Local modes contain CXL traffic: {group}")
        if cxl["local_kv_read_transactions"] != 0:
            raise ValueError(f"isolated CXL mode contains Local KV traffic: {group}")
        if cxl_local["cxl_kv_read_transactions"] != cxl["cxl_kv_read_transactions"]:
            raise ValueError(f"CXL traffic changed after adding Local KV: {group}")
        if combined["cxl_kv_read_transactions"] != cxl["cxl_kv_read_transactions"]:
            raise ValueError(f"CXL traffic changed after adding Expert: {group}")
        if cxl["cxl_queue_wait_cycles"] <= 0 or cxl["cxl_link_busy_cycles"] <= 0:
            raise ValueError(f"CXL-only workload did not exercise the CXL queue: {group}")

    validation = {
        "status": "passed",
        "rows": len(rows),
        "profiles": sorted(EXPECTED_PROFILES),
        "cases": sorted(EXPECTED_CASES),
        "modes": sorted(EXPECTED_MODES),
        "request_scale": metadata["request_scale"],
        "interpolation": 0,
        "comparison_sha256": sha256_file(comparison_path),
        "stage_manifest_sha256": sha256_file(manifest_path),
    }
    (output / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return validation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "results/kv_cxl_request_stage4_v1"))
    args = parser.parse_args(argv)
    print(json.dumps(validate(args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
