#!/usr/bin/env python3
"""Audit saved KV sensitivity evidence and compare modes within each variant.

This script never runs Ramulator or rewrites the original plan/results. Cache
integrity and source identity are checked; missing historical binary hashes
remain an explicit limitation rather than a retrospectively supplied attestation.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.freeze_kv_baseline import FROZEN, freeze_or_verify
from scripts.run_kv_read_experiments import CASES, PLACEMENTS, MODES, validate_result
from scripts.run_kv_read_sensitivity import (
    BASE_PLAN, BASE_EXACT, CYCLE, OUTPUT, CACHE_ROOT, atomic_json, digest,
    variants, read_json,
)
from sieve_replay.ramulator.kv_read_workload import (
    KVReadWorkloadResult, KVWorkloadShape, kv_read_cache_context,
)
from sieve_replay.ramulator.mixed_workload import SieveCycleV1Config
from sieve_replay.report.writer import sha256_file


def check_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch")


def validate_matrix(plan: dict, exact: dict, base_plan: dict, cycle: SieveCycleV1Config) -> None:
    check_equal(plan["variants"], variants(cycle), "controller variants")
    expected_consumers = {(c, p, m) for c in CASES for p in PLACEMENTS for m in MODES}
    consumers = base_plan["consumers"]
    check_equal(len(consumers), len(expected_consumers), "consumer count")
    check_equal({(r["case"], r["placement"], r["mode"]) for r in consumers}, expected_consumers, "consumer matrix")
    shapes = {}
    for row in consumers:
        check_equal(row["shape_key"], digest(row["shape"]), "consumer shape digest")
        shapes[row["shape_key"]] = row["shape"]
    check_equal(len(shapes), 19, "unique shape count")
    check_equal(plan["consumer_count"], 36, "planned consumers")
    check_equal(plan["shapes"], [{"shape_key": k, "shape": shapes[k]} for k in sorted(shapes)], "planned shapes")
    expected_keys = {f"{v['name']}:{key}" for v in plan["variants"] for key in shapes}
    check_equal(set(exact["entries"]), expected_keys, "exact entry coverage")
    check_equal(exact["summary"], {"required": 95, "completed": 95, "failed": 0, "interpolated": 0}, "completion summary")
    check_equal(exact["failures"], [], "failures")


def verify_cache_entry(entry: dict, context: dict, cache_root: Path) -> None:
    cache_input = {"context": context, "variant": entry["variant"], "shape": entry["shape"]}
    path = cache_root / (digest(cache_input) + ".json")
    check_equal(sha256_file(path), entry["cache_file_sha256"], "cache SHA-256")
    check_equal(read_json(path), {"cache_input": cache_input, "result": entry["result"]}, "cache content")


def check_comparison(rows: list[dict], entries: dict) -> None:
    check_equal(len(rows), len(entries), "comparison size")
    check_equal({f"{r['variant']}:{r['shape_key']}" for r in rows}, set(entries), "comparison coverage")
    for row in rows:
        key = f"{row['variant']}:{row['shape_key']}"
        entry = entries[key]
        result = KVReadWorkloadResult(**entry["result"])
        base = KVReadWorkloadResult(**entries[f"baseline-rb256-dual:{row['shape_key']}"]["result"])
        check_equal(row["shape"], entry["shape"], "comparison shape")
        expected = {
            "total_completion_us": result.total_completion_us,
            "baseline_total_completion_us": base.total_completion_us,
            "delta_total_completion_us": result.total_completion_us - base.total_completion_us,
            "gpu_accepted_to_column_issue_mean_us": mean_preissue(result, "gpu") or 0.0,
            "kv_accepted_to_column_issue_mean_us": mean_preissue(result, "kv") or 0.0,
            "gpu_injection_rejected_attempts": result.gpu_injection_rejected_attempts,
            "kv_injection_rejected_attempts": result.kv_injection_rejected_attempts,
        }
        for field, value in expected.items():
            if not isinstance(row[field], (int, float)) or not math.isclose(row[field], value, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"comparison derived value mismatch: {field}")


def mean_preissue(result: KVReadWorkloadResult, stream: str) -> float | None:
    count = getattr(result, stream + "_completed_requests")
    if not count:
        return None
    return getattr(result, stream + "_accepted_to_column_issue_cycles") * result.tick_ps / 1_000_000 / count


def compare_modes(expert: KVReadWorkloadResult, kv: KVReadWorkloadResult, combined: KVReadWorkloadResult) -> dict:
    """Use isolated controls from the *same* controller variant."""
    isolated_max = max(expert.total_completion_us, kv.total_completion_us)
    expert_combined = max(combined.gpu_completion_us, combined.pim_completion_us)
    def ratio(num: float, den: float) -> float | None:
        return num / den if den else None
    return {
        "expert_only_us": expert.total_completion_us,
        "kv_only_us": kv.total_completion_us,
        "combined_us": combined.total_completion_us,
        "combined_minus_isolated_max_us": combined.total_completion_us - isolated_max,
        "combined_over_isolated_max": ratio(combined.total_completion_us, isolated_max),
        "expert_completion_delta_us": expert_combined - expert.total_completion_us,
        "expert_completion_ratio": ratio(expert_combined, expert.total_completion_us),
        "gpu_expert_completion_delta_us": combined.gpu_completion_us - expert.gpu_completion_us,
        "gpu_expert_completion_ratio": ratio(combined.gpu_completion_us, expert.gpu_completion_us),
        "pim_expert_completion_delta_us": combined.pim_completion_us - expert.pim_completion_us,
        "pim_expert_completion_ratio": ratio(combined.pim_completion_us, expert.pim_completion_us),
        "kv_completion_delta_us": combined.kv_completion_us - kv.kv_completion_us,
        "kv_completion_ratio": ratio(combined.kv_completion_us, kv.kv_completion_us),
        "gpu_accepted_to_column_issue_mean_us": mean_preissue(combined, "gpu"),
        "kv_accepted_to_column_issue_mean_us": mean_preissue(combined, "kv"),
        "gpu_isolated_accepted_to_column_issue_mean_us": mean_preissue(expert, "gpu"),
        "kv_isolated_accepted_to_column_issue_mean_us": mean_preissue(kv, "kv"),
        "gpu_rejected_attempts_per_request": ratio(combined.gpu_injection_rejected_attempts, combined.gpu_completed_requests),
        "kv_rejected_attempts_per_request": ratio(combined.kv_injection_rejected_attempts, combined.kv_completed_requests),
    }


def matched_rows(base_plan: dict, plan: dict, exact: dict) -> list[dict]:
    consumers = {(r["case"], r["placement"], r["mode"]): r for r in base_plan["consumers"]}
    rows = []
    for variant in plan["variants"]:
        for case in CASES:
            for placement in PLACEMENTS:
                selected = [consumers[(case, placement, mode)] for mode in MODES]
                results = [KVReadWorkloadResult(**exact["entries"][f"{variant['name']}:{r['shape_key']}"]["result"]) for r in selected]
                combined = selected[-1]
                rows.append({
                    "variant": variant["name"], "case": case, "placement": placement,
                    "batch_size": combined["batch_size"], "context_length": combined["context_lengths"][0],
                    "gpu_experts": len(combined["gpu_experts"]), "pim_experts": len(combined["pim_experts"]),
                    **{f"{mode.replace('-', '_')}_shape_key": r["shape_key"] for mode, r in zip(MODES, selected)},
                    **compare_modes(*results),
                })
    return rows


def audit() -> dict:
    frozen = freeze_or_verify(verify_only=True)
    paths = {name: OUTPUT / f"kv_read_sensitivity_{name}.json" for name in ("plan", "exact_results", "comparison")}
    plan, exact, comparison = (read_json(paths[n]) for n in ("plan", "exact_results", "comparison"))
    base_plan, base_exact = read_json(BASE_PLAN), read_json(BASE_EXACT)
    cycle = SieveCycleV1Config.load(CYCLE)
    source = ROOT / "scripts/run_kv_read_sensitivity.py"
    method = "kv-read-controller-sensitivity-v1"
    check_equal(plan["method"], method, "plan method")
    check_equal(exact["method"], method, "result method")
    for field, path in (("base_plan_sha256", BASE_PLAN), ("base_exact_results_sha256", BASE_EXACT),
                        ("frozen_baseline_sha256", FROZEN), ("cycle_config_sha256", CYCLE), ("runner_sha256", source)):
        check_equal(plan[field], sha256_file(path), field)
    check_equal(plan["cycle_config_snapshot"], asdict(cycle), "cycle snapshot")
    check_equal(plan["simulation_context"], kv_read_cache_context(ROOT, CYCLE), "request-model source context")
    check_equal(exact["plan_sha256"], sha256_file(paths["plan"]), "result plan hash")
    check_equal(exact["frozen_baseline_sha256"], sha256_file(FROZEN), "result frozen hash")
    context = {**plan["simulation_context"], "sensitivity_plan_sha256": sha256_file(paths["plan"])}
    check_equal(exact["cache_context"], context, "execution cache context")
    check_equal(base_exact["plan_sha256"], sha256_file(BASE_PLAN), "original KV plan hash")
    base_audit_path = BASE_EXACT.parent / "validation.json"
    base_audit = read_json(base_audit_path)
    check_equal(base_audit["exact_results_sha256"], sha256_file(BASE_EXACT), "original KV audit hash")
    validate_matrix(plan, exact, base_plan, cycle)
    baseline_equal = 0
    completed_requests = {s: 0 for s in ("gpu", "kv", "pim")}
    for variant in plan["variants"]:
        variant_cycle = replace(cycle, read_buffer_size=variant["read_buffer_size"], dual_row_buffer=variant["dual_row_buffer"])
        variant_cycle.validate()
        for shape in plan["shapes"]:
            entry = exact["entries"][f"{variant['name']}:{shape['shape_key']}"]
            for field, value in (("variant", variant), ("shape_key", shape["shape_key"]), ("shape", shape["shape"])):
                check_equal(entry[field], value, "entry " + field)
            if entry["origin"] not in {"new-exact-run", "exact-cache"}:
                raise ValueError("invalid exact origin")
            result = KVReadWorkloadResult(**entry["result"])
            validate_result(KVWorkloadShape(**shape["shape"]), result, variant_cycle)
            verify_cache_entry(entry, context, CACHE_ROOT)
            check_equal(result.gpu_column_issues, result.controller_read_completed_requests, "normal READ column issues")
            check_equal(result.pim_column_issues, result.pim_completed_requests, "PIM column issues")
            for stream in ("gpu", "kv"):
                count = getattr(result, stream + "_completed_requests")
                total = getattr(result, stream + "_request_residence_cycles")
                maximum = getattr(result, stream + "_max_request_residence_cycles")
                if not count and (total or maximum):
                    raise ValueError("empty stream has residence")
                if count and not result.read_latency_cycles <= maximum <= total <= count * maximum:
                    raise ValueError("inconsistent request residence maximum")
            for stream in completed_requests:
                completed_requests[stream] += getattr(result, stream + "_completed_requests")
            if variant["name"] == "baseline-rb256-dual":
                check_equal(entry["result"], base_exact["entries"][shape["shape_key"]]["result"], "baseline full-field regression")
                baseline_equal += 1
    for field, path in (("plan_sha256", paths["plan"]), ("exact_results_sha256", paths["exact_results"]), ("frozen_baseline_sha256", FROZEN)):
        check_equal(comparison[field], sha256_file(path), "comparison " + field)
    check_comparison(comparison["rows"], exact["entries"])
    csv_path = OUTPUT / "kv_read_sensitivity_comparison.csv"
    with csv_path.open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    check_equal(len(csv_rows), len(comparison["rows"]), "CSV row count")
    for actual, expected in zip(csv_rows, comparison["rows"]):
        check_equal(actual, {k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else str(v) for k, v in expected.items()}, "CSV row")
    # Account for old plan namespaces without treating them as additional cases.
    cache_namespaces: dict[str, int] = {}
    for path in CACHE_ROOT.glob("*.json"):
        key = read_json(path)["cache_input"]["context"]["sensitivity_plan_sha256"]
        cache_namespaces[key] = cache_namespaces.get(key, 0) + 1
    check_equal(cache_namespaces.get(sha256_file(paths["plan"])), 95, "current cache namespace size")
    rows = matched_rows(base_plan, plan, exact)
    hashes = {f"{name}_sha256": sha256_file(path) for name, path in paths.items()}
    hashes.update({"base_plan_sha256": sha256_file(BASE_PLAN), "base_exact_sha256": sha256_file(BASE_EXACT),
                   "base_audit_sha256": sha256_file(base_audit_path), "frozen_sha256": sha256_file(FROZEN),
                   "comparison_csv_sha256": sha256_file(csv_path), "auditor_sha256": sha256_file(Path(__file__))})
    limitations = [
        "Historical sensitivity runs did not record binding/library hashes; source/cache audit does not attest the binary used at execution time",
        "Cache namespaces reflect two plan versions, not additional workloads or independent hardware replicates",
        "Controller variants keep simultaneous injection and the original abstract address mapping; injection/address robustness is untested",
        "Each real trace contributes one layer-batch; timings are Ramulator microbenchmarks, not A800 or end-to-end measurements",
        "Accepted-to-column issue includes ACT/PRE and arbitration; rejected attempts count retries, not dropped requests",
    ]
    matched_path = OUTPUT / "matched_comparison.json"
    atomic_json(matched_path, {"classification": "Ramulator sensitivity with matched isolated controls", "input_hashes": hashes,
                              "rows": rows, "limitations": limitations})
    with (OUTPUT / "matched_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "status": "passed_with_provenance_limitation", "classification": "saved source/cache/counter audit; no new simulation",
        "input_hashes": hashes, "summary": exact["summary"], "frozen_files_verified": len(frozen["files"]),
        "baseline_full_field_matches": baseline_equal, "cache_namespaces": cache_namespaces,
        "matched_comparisons": len(rows), "matched_comparison_sha256": sha256_file(matched_path),
        "completed_requests_over_95_shapes": completed_requests,
        "binary_provenance_recorded_at_execution": False, "limitations": limitations,
        "e2e_extension_gate": "not established: injection/address sensitivity and A800 timing remain untested",
    }
    atomic_json(OUTPUT / "validation.json", result)
    return result


if __name__ == "__main__":
    result = audit()
    print(json.dumps({k: result[k] for k in ("status", "summary", "baseline_full_field_matches", "matched_comparisons", "frozen_files_verified")}, sort_keys=True))
