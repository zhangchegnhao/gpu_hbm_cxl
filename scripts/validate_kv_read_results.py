#!/usr/bin/env python3
"""Audit KV exact results and check Expert-only timings against the frozen cache."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.freeze_kv_baseline import FROZEN, freeze_or_verify
from scripts.run_kv_read_experiments import (
    CACHE_ROOT, CYCLE, OUTPUT, _binding_provenance, atomic_json, digest, read_json, validate_result,
)
from sieve_replay.ramulator.kv_read_workload import KVReadWorkloadResult, KVWorkloadShape, kv_read_cache_context
from sieve_replay.ramulator.mixed_workload import SieveCycleV1Config
from sieve_replay.ramulator.workload_cache import ContentionWorkloadCache, WorkloadShape
from sieve_replay.report.writer import sha256_file


def validate() -> dict:
    frozen = freeze_or_verify(verify_only=True)
    plan = read_json(OUTPUT / "kv_read_plan.json")
    exact_path = OUTPUT / "kv_read_exact_results.json"
    exact = read_json(exact_path)
    summary = exact["summary"]
    if (summary["remaining"] or summary["failed"] or summary["interpolated_workloads"]
            or summary["completed"] != len(plan["workloads"])
            or len(exact["entries"]) != summary["completed"]
            or exact["plan_sha256"] != sha256_file(OUTPUT / "kv_read_plan.json")):
        raise ValueError("KV result set is incomplete or does not match its plan")
    if exact.get("cache_context", {}).get("plan_sha256") != sha256_file(OUTPUT / "kv_read_plan.json"):
        raise ValueError("KV cache context is not bound to the current plan")
    simulation_context = plan.get("simulation_context", {})
    if any(exact["cache_context"].get(key) != value for key, value in simulation_context.items()):
        raise ValueError("KV cache context differs from the planned simulation context")
    current_context = kv_read_cache_context(ROOT, CYCLE)
    if any(exact["cache_context"].get(key) != value for key, value in current_context.items()):
        raise ValueError("KV request-model source provenance differs from the exact result")
    execution = _binding_provenance(SieveCycleV1Config.load(CYCLE))
    if any(exact["cache_context"].get(key) != value for key, value in execution.items() if key != "ramulator_commit"):
        raise ValueError("KV binary provenance differs from the exact result")
    program = exact.get("program_snapshot", {})
    if (program.get("runner_sha256") != sha256_file(ROOT / "scripts/run_kv_read_experiments.py")
            or program.get("request_model_sha256") != sha256_file(ROOT / "src/sieve_replay/ramulator/kv_read_workload.py")
            or program.get("frontend_sha256") != sha256_file(ROOT / "ramulator/extensions/sieve_kv_read/source/sieve_kv_read_frontend.cpp")):
        raise ValueError("KV exact result program snapshot differs from current sources")
    old_proof = read_json(ROOT / "results/kv_workload_timing_v1/exact_workloads.json")
    old_cache = ContentionWorkloadCache(
        ROOT / "ramulator/timing_tables/generated/.cache/kv_long_context_cycle_v1",
        old_proof["cache_context"],
    )
    cycle = SieveCycleV1Config.load(CYCLE)
    regression = []
    counters = {stream: 0 for stream in ("gpu", "kv", "pim")}
    for workload in plan["workloads"]:
        key = workload["shape_key"]
        entry = exact["entries"][key]
        if key != digest(workload["shape"]) or entry["shape"] != workload["shape"]:
            raise ValueError("KV shape identity mismatch")
        result = KVReadWorkloadResult(**entry["result"])
        shape = KVWorkloadShape(**entry["shape"])
        validate_result(shape, result, cycle)
        expected_input = {"context": exact["cache_context"], "shape": entry["shape"]}
        cache_path = CACHE_ROOT / (digest(expected_input) + ".json")
        cache = read_json(cache_path)
        if (sha256_file(cache_path) != entry["cache_file_sha256"]
                or cache["cache_input"] != expected_input or cache["result"] != entry["result"]):
            raise ValueError("KV exact cache differs from attested result")
        for stream in counters:
            counters[stream] += getattr(result, stream + "_completed_requests")
        if shape.kv_read_transactions == 0:
            old_shape = WorkloadShape(**{k: v for k, v in asdict(shape).items() if k != "kv_read_transactions"})
            old_key = old_cache.key(old_shape)
            old_path = old_cache.path(old_shape)
            old_raw = read_json(old_path)
            if (sha256_file(old_path) != old_proof["entries"][old_key]["cache_file_sha256"]
                    or old_raw["cache_input"] != {"context": old_proof["cache_context"], "shape": asdict(old_shape)}):
                raise ValueError("frozen Expert-only cache provenance mismatch")
            if any(entry["result"].get(k) != v for k, v in old_raw["result"].items()):
                raise ValueError("KV frontend changed the frozen Expert-only result")
            regression.append({"kv_shape_key": key, "baseline_shape_key": old_key,
                               "all_legacy_fields_equal": True})
    comparison = read_json(OUTPUT / "kv_read_comparison.json")
    for key, path in (("plan_sha256", OUTPUT / "kv_read_plan.json"),
                      ("exact_results_sha256", exact_path), ("frozen_baseline_sha256", FROZEN)):
        if comparison[key] != sha256_file(path):
            raise ValueError(f"comparison source hash mismatch: {key}")
    audit = {
        "classification": "exact Ramulator result audit; no additional hardware timing",
        "exact_results_sha256": sha256_file(exact_path),
        "comparison_sha256": sha256_file(OUTPUT / "kv_read_comparison.json"),
        "frozen_baseline_sha256": sha256_file(FROZEN),
        "frozen_files_verified": len(frozen["files"]),
        "summary": summary, "completed_requests_over_unique_runs": counters,
        "expert_only_regressions": regression,
    }
    atomic_json(OUTPUT / "validation.json", audit)
    return audit


if __name__ == "__main__":
    result = validate()
    print(json.dumps({"exact_shapes": result["summary"]["completed"],
                      "expert_only_regressions": len(result["expert_only_regressions"]),
                      "frozen_files_verified": result["frozen_files_verified"]}))
