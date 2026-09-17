#!/usr/bin/env python3
"""Scan KV READ queue/controller assumptions with exact Ramulator runs.

This experiment reuses the verified representative consumers from
``results/kv_read_experiment_v1`` and varies only controller admission
parameters.  It is a request-level sensitivity study; it does not alter the
frozen baseline or the serial decode replay.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.freeze_kv_baseline import FROZEN, freeze_or_verify
from scripts.run_kv_read_experiments import (
    OUTPUT as BASE_OUTPUT,
    read_json,
    validate_result,
)
from sieve_replay.ramulator.kv_read_workload import (
    KVReadWorkloadResult,
    KVWorkloadShape,
    kv_read_cache_context,
    run_kv_read_workload,
)
from sieve_replay.ramulator.mixed_workload import SieveCycleV1Config
from sieve_replay.report.writer import sha256_file

CYCLE = ROOT / "configs/ramulator/sieve_hbm3e_cycle_v1.json"
OUTPUT = ROOT / "results/kv_read_sensitivity_v1"
CACHE_ROOT = ROOT / "ramulator/timing_tables/generated/.cache/kv_read_sensitivity_v1"
BASE_PLAN = BASE_OUTPUT / "kv_read_plan.json"
BASE_EXACT = BASE_OUTPUT / "kv_read_exact_results.json"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def variants(base: SieveCycleV1Config) -> list[dict[str, Any]]:
    """Controller-only variants; the first one is the unchanged baseline."""
    return [
        {"name": "baseline-rb256-dual", "read_buffer_size": base.read_buffer_size, "dual_row_buffer": base.dual_row_buffer},
        {"name": "rb32-dual", "read_buffer_size": 32, "dual_row_buffer": True},
        {"name": "rb64-dual", "read_buffer_size": 64, "dual_row_buffer": True},
        {"name": "rb128-dual", "read_buffer_size": 128, "dual_row_buffer": True},
        {"name": "rb256-single", "read_buffer_size": 256, "dual_row_buffer": False},
    ]


def _run_one(args: tuple[dict[str, int], dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    shape_raw, variant, cycle_raw = args
    base = SieveCycleV1Config(**cycle_raw)
    cycle = replace(base, read_buffer_size=int(variant["read_buffer_size"]), dual_row_buffer=bool(variant["dual_row_buffer"]))
    shape = KVWorkloadShape(**shape_raw)
    result = run_kv_read_workload(ROOT / "third_party/ramulator2", cycle, **shape_raw)
    validate_result(shape, result, cycle)
    return asdict(result)


def plan(output: Path = OUTPUT) -> dict[str, Any]:
    freeze_or_verify(verify_only=True)
    base_plan = read_json(BASE_PLAN)
    exact = read_json(BASE_EXACT)
    summary = exact.get("summary", {})
    if summary.get("remaining") or summary.get("failed") or summary.get("interpolated_workloads"):
        raise ValueError("base KV READ experiment is incomplete")
    base_cycle = SieveCycleV1Config.load(CYCLE)
    consumer_rows = base_plan.get("consumers", [])
    if not consumer_rows:
        raise ValueError("base KV READ plan has no consumers")
    shapes = {row["shape_key"]: row["shape"] for row in consumer_rows}
    if sorted(shapes) != sorted(exact.get("entries", {})):
        raise ValueError("base KV exact result does not cover all planned shapes")
    payload = {
        "schema_version": 1,
        "method": "kv-read-controller-sensitivity-v1",
        "classification": "exact Ramulator request-level sensitivity; not A800 timing or end-to-end replay",
        "base_plan_sha256": sha256_file(BASE_PLAN),
        "base_exact_results_sha256": sha256_file(BASE_EXACT),
        "frozen_baseline_sha256": sha256_file(FROZEN),
        "cycle_config_sha256": sha256_file(CYCLE),
        "cycle_config_snapshot": asdict(base_cycle),
        "runner_sha256": sha256_file(Path(__file__)),
        "simulation_context": kv_read_cache_context(ROOT, CYCLE),
        "variants": variants(base_cycle),
        "shapes": [{"shape_key": key, "shape": shapes[key]} for key in sorted(shapes)],
        "consumer_count": len(consumer_rows),
        "scope": "19 exact shapes reused by the four real A800 Router cases; controller read-buffer/row-buffer sensitivity",
    }
    atomic_json(output / "kv_read_sensitivity_plan.json", payload)
    return payload


def run(output: Path = OUTPUT, workers: int = 4) -> dict[str, Any]:
    plan_payload = read_json(output / "kv_read_sensitivity_plan.json")
    if plan_payload.get("runner_sha256") != sha256_file(Path(__file__)):
        raise ValueError("sensitivity plan was created by a different runner")
    freeze_or_verify(verify_only=True)
    if plan_payload.get("base_plan_sha256") != sha256_file(BASE_PLAN) or plan_payload.get("base_exact_results_sha256") != sha256_file(BASE_EXACT):
        raise ValueError("base KV result changed after sensitivity planning")
    if plan_payload.get("frozen_baseline_sha256") != sha256_file(FROZEN):
        raise ValueError("frozen baseline inventory changed")
    base_cycle = SieveCycleV1Config.load(CYCLE)
    if plan_payload.get("cycle_config_snapshot") != asdict(base_cycle):
        raise ValueError("cycle configuration changed after sensitivity planning")
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    context = {**plan_payload["simulation_context"], "sensitivity_plan_sha256": sha256_file(output / "kv_read_sensitivity_plan.json")}
    entries: dict[str, Any] = {}
    pending: list[tuple[str, dict[str, int], dict[str, Any]]] = []
    for variant in plan_payload["variants"]:
        for item in plan_payload["shapes"]:
            key = f"{variant['name']}:{item['shape_key']}"
            cache_input = {"context": context, "variant": variant, "shape": item["shape"]}
            cache_path = CACHE_ROOT / (digest(cache_input) + ".json")
            if cache_path.is_file():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("cache_input") != cache_input:
                    raise ValueError("sensitivity cache provenance mismatch")
                result = KVReadWorkloadResult(**cached["result"])
                validate_result(KVWorkloadShape(**item["shape"]), result, replace(base_cycle, read_buffer_size=int(variant["read_buffer_size"]), dual_row_buffer=bool(variant["dual_row_buffer"])))
                entries[key] = {"variant": variant, "shape_key": item["shape_key"], "shape": item["shape"], "result": cached["result"], "cache_file_sha256": sha256_file(cache_path), "origin": "exact-cache"}
            else:
                pending.append((key, item["shape"], variant))
    cycle_raw = asdict(base_cycle)
    failures: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, (shape, variant, cycle_raw)): (key, shape, variant) for key, shape, variant in pending}
        for future in as_completed(futures):
            key, shape, variant = futures[future]
            try:
                result = future.result()
                cache_input = {"context": context, "variant": variant, "shape": shape}
                cache_path = CACHE_ROOT / (digest(cache_input) + ".json")
                atomic_json(cache_path, {"cache_input": cache_input, "result": result})
                entries[key] = {"variant": variant, "shape_key": digest(shape), "shape": shape, "result": result, "cache_file_sha256": sha256_file(cache_path), "origin": "new-exact-run"}
            except Exception as exc:
                failures.append({"key": key, "error": str(exc)})
    result_payload = {
        "schema_version": 1,
        "method": plan_payload["method"],
        "classification": plan_payload["classification"],
        "plan_sha256": sha256_file(output / "kv_read_sensitivity_plan.json"),
        "frozen_baseline_sha256": sha256_file(FROZEN),
        "cache_context": context,
        "summary": {"required": len(plan_payload["variants"]) * len(plan_payload["shapes"]), "completed": len(entries), "failed": len(failures), "interpolated": 0},
        "entries": entries,
        "failures": failures,
    }
    atomic_json(output / "kv_read_sensitivity_exact_results.json", result_payload)
    if failures:
        return result_payload["summary"]
    _write_comparison(output, plan_payload, result_payload)
    return result_payload["summary"]


def _write_comparison(output: Path, plan_payload: dict[str, Any], exact: dict[str, Any]) -> None:
    baseline = {key.split(":", 1)[1]: value for key, value in exact["entries"].items() if key.startswith("baseline-rb256-dual:")}
    rows = []
    for key, entry in sorted(exact["entries"].items()):
        shape_key = entry["shape_key"]
        result = KVReadWorkloadResult(**entry["result"])
        base = KVReadWorkloadResult(**baseline[shape_key]["result"])
        rows.append({"variant": entry["variant"]["name"], "shape_key": shape_key, "shape": entry["shape"], "total_completion_us": result.total_completion_us, "baseline_total_completion_us": base.total_completion_us, "delta_total_completion_us": result.total_completion_us - base.total_completion_us, "gpu_accepted_to_column_issue_mean_us": result.gpu_accepted_to_column_issue_cycles * result.tick_ps / 1_000_000 / result.gpu_completed_requests if result.gpu_completed_requests else 0.0, "kv_accepted_to_column_issue_mean_us": result.kv_accepted_to_column_issue_cycles * result.tick_ps / 1_000_000 / result.kv_completed_requests if result.kv_completed_requests else 0.0, "gpu_injection_rejected_attempts": result.gpu_injection_rejected_attempts, "kv_injection_rejected_attempts": result.kv_injection_rejected_attempts})
    comparison = {"schema_version": 1, "classification": plan_payload["classification"], "plan_sha256": exact["plan_sha256"], "exact_results_sha256": sha256_file(output / "kv_read_sensitivity_exact_results.json"), "frozen_baseline_sha256": sha256_file(FROZEN), "rows": rows, "limitations": ["Controller read-buffer and dual-row assumptions only; no staggered/serial injection mode is implemented", "KV and Expert requests are abstract sequential streams, not A800 memory traces", "These are request-level microbenchmarks, not end-to-end decode timing"]}
    atomic_json(output / "kv_read_sensitivity_comparison.json", comparison)
    if rows:
        with (output / "kv_read_sensitivity_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if args.workers <= 0:
        parser.error("--workers must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    payload = plan(args.output)
    summary = run(args.output, args.workers) if args.run else {"required": len(payload["variants"]) * len(payload["shapes"]), "completed": 0, "failed": 0, "interpolated": 0}
    print(json.dumps(summary, sort_keys=True))
    return 1 if summary.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
