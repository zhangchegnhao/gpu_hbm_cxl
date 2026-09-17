#!/usr/bin/env python3
"""Plan or run the representative KV READ contention microbenchmark.

The default is planning only.  Four real traces contribute exactly their first
layer-batch (step 0, layer 0), under three fixed Expert placements and three
request modes.  Exact shapes are deduplicated across every consumer.  This is
independent of the frozen cycle-v1 tables and the serial decode replay.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.freeze_kv_baseline import FROZEN, freeze_or_verify
from sieve_replay.config import load_configuration
from sieve_replay.ramulator.kv_read_workload import (
    KVReadWorkloadResult, KVWorkloadShape, kv_read_cache_context, run_kv_read_workload,
)
from sieve_replay.ramulator.mixed_workload import SieveCycleV1Config
from sieve_replay.ramulator.workload_cache import shape_for_loads
from sieve_replay.report.writer import sha256_file
from sieve_replay.trace import load_trace_set
from sieve_replay.trace_manifest import TraceManifest

CASES = ("b8_c4k", "b8_c8k", "b8_c16k", "b16_c8k")
PLACEMENTS = ("gpu-only", "frozen-oracle", "fixed-half-prefix")
MODES = ("expert-only", "kv-only", "combined")
CYCLE = ROOT / "configs/ramulator/sieve_hbm3e_cycle_v1.json"
OUTPUT = ROOT / "results/kv_read_experiment_v1"
CACHE_ROOT = ROOT / "ramulator/timing_tables/generated/.cache/kv_read_experiment_v1"
SCOPE = "one representative layer-batch per case: step 0, layer 0; not all 384 layer-batches"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def kv_transactions(context_lengths: Iterable[int], model: Any, cycle: SieveCycleV1Config) -> tuple[int, int]:
    """K+V bytes and HBM transactions read by one decode Attention layer."""
    values = tuple(context_lengths)
    if not values or any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("context_lengths must contain positive integers")
    size = 2 * sum(values) * model.num_key_value_heads * model.head_dim * model.dtype_bytes
    return size, (size + cycle.transaction_bytes - 1) // cycle.transaction_bytes


def _frozen_oracle(case: str, cfg: Any, batch: Any) -> tuple[tuple[int, ...], tuple[int, ...], dict[str, str]]:
    directory = ROOT / "results/full_decode_real_kv_cycle_v1" / case / "sieve-cycle-v1"
    placement_path = directory / "placement.csv"
    manifest_path = directory / "run_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("policy") != "sieve-cycle-v1" or manifest.get("experiment") != cfg.experiment.raw:
        raise ValueError(f"{case}: frozen placement policy/configuration mismatch")
    paths = {
        "trace_sha256": cfg.experiment.trace_path,
        "trace_manifest_sha256": cfg.experiment.trace_manifest_path,
        "model_sha256": cfg.experiment.model_path,
        "hardware_sha256": cfg.experiment.hardware_path,
        "ramulator_cycle_config_sha256": CYCLE,
        "pim_timing_table_sha256": cfg.experiment.pim_timing_table_path,
        "contention_timing_table_sha256": cfg.experiment.contention_timing_table_path,
    }
    if any(path is None or manifest.get("input_hashes", {}).get(name) != sha256_file(path) for name, path in paths.items()):
        raise ValueError(f"{case}: frozen placement input hashes differ from current inputs")
    by_id = {load.expert_id: load.token_count for load in batch.expert_loads}
    seen: set[int] = set()
    gpu: list[int] = []
    pim: list[int] = []
    with placement_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (int(row["step"]), int(row["layer"])) != (batch.step, batch.layer):
                continue
            expert = int(row["expert_id"])
            if expert in seen or by_id.get(expert) != int(row["token_count"]):
                raise ValueError(f"{case}: frozen placement does not match routed Expert loads")
            seen.add(expert)
            target = row["target"]
            if target not in {"gpu", "pim"}:
                raise ValueError(f"{case}: invalid placement target")
            (gpu if target == "gpu" else pim).append(expert)
    if seen != set(by_id):
        raise ValueError(f"{case}: incomplete frozen placement")
    return tuple(sorted(gpu)), tuple(sorted(pim)), {
        "placement_path": placement_path.relative_to(ROOT).as_posix(),
        "placement_sha256": sha256_file(placement_path),
        "placement_manifest_path": manifest_path.relative_to(ROOT).as_posix(),
        "placement_manifest_sha256": sha256_file(manifest_path),
        "pim_timing_table_sha256": sha256_file(cfg.experiment.pim_timing_table_path),
        "contention_timing_table_sha256": sha256_file(cfg.experiment.contention_timing_table_path),
    }


def _case_plan(case: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if case not in CASES:
        raise ValueError(f"unsupported case: {case}")
    experiment = ROOT / f"configs/experiments/full_decode_real_kv_{case}.json"
    cfg = load_configuration(experiment)
    e = cfg.experiment
    traces = load_trace_set(e.trace_path, cfg.model, e.layers, e.steps)
    if e.trace_manifest_path is None:
        raise ValueError(f"{case}: a real Router manifest is required")
    manifest = TraceManifest.load_and_validate(e.trace_manifest_path, e.trace_path, cfg.model, traces)
    matches = [batch for batch in traces.batches if (batch.step, batch.layer) == (0, 0)]
    if len(matches) != 1:
        raise ValueError(f"{case}: expected exactly one step-0/layer-0 batch")
    batch = matches[0]
    gpu, pim, placement_meta = _frozen_oracle(case, cfg, batch)
    hot = tuple(load.expert_id for load in sorted(batch.expert_loads, key=lambda load: (-load.token_count, load.expert_id)))
    half = (len(hot) + 1) // 2
    placements = {
        "gpu-only": (tuple(sorted(hot)), ()),
        "frozen-oracle": (gpu, pim),
        "fixed-half-prefix": (tuple(sorted(hot[:half])), tuple(sorted(hot[half:]))),
    }
    cycle = SieveCycleV1Config.load(CYCLE)
    byte_count, transactions = kv_transactions(batch.context_lengths, cfg.model, cycle)
    by_id = {load.expert_id: load for load in batch.expert_loads}
    consumers: list[dict[str, Any]] = []
    for placement, (gpu_ids, pim_ids) in placements.items():
        expert_shape = asdict(shape_for_loads(tuple(by_id[x] for x in gpu_ids), tuple(by_id[x] for x in pim_ids), cfg.model, cycle))
        shapes = {
            "expert-only": KVWorkloadShape(**expert_shape, kv_read_transactions=0),
            "kv-only": KVWorkloadShape(0, 0, 0, 0, transactions),
            "combined": KVWorkloadShape(**expert_shape, kv_read_transactions=transactions),
        }
        for mode, shape in shapes.items():
            consumers.append({
                "case": case, "placement": placement, "mode": mode,
                "scope": SCOPE, "step": batch.step, "layer": batch.layer,
                "batch_size": batch.batch_size, "context_lengths": list(batch.context_lengths),
                "kv_read_bytes_per_layer": byte_count,
                "gpu_experts": list(gpu_ids), "pim_experts": list(pim_ids),
                "gpu_tokens": sum(by_id[x].token_count for x in gpu_ids),
                "pim_tokens": sum(by_id[x].token_count for x in pim_ids),
                "shape": asdict(shape), "shape_key": digest(asdict(shape)),
            })
    paths = {
        "experiment": experiment, "model": e.model_path, "hardware": e.hardware_path,
        "trace": e.trace_path, "trace_manifest": e.trace_manifest_path,
        "prompts": e.trace_manifest_path.parent / manifest.raw["prompts"]["file"],
        "cycle_config": CYCLE,
    }
    metadata = {
        "scope": SCOPE, "trace_layer_batches": len(traces.batches), "selected_layer_batches": 1,
        "input_hashes": {name + "_sha256": sha256_file(path) for name, path in paths.items()},
        "configuration_snapshot": e.raw, "model_snapshot": json.loads(e.model_path.read_text()),
        "hardware_snapshot": json.loads(e.hardware_path.read_text()),
        "frozen_placement": placement_meta,
        "representative": {"step": 0, "layer": 0, "context_lengths": list(batch.context_lengths), "expert_loads": [asdict(load) for load in batch.expert_loads]},
    }
    return metadata, consumers


def plan(output: Path = OUTPUT, cases: Iterable[str] = CASES) -> dict[str, Any]:
    freeze_or_verify(verify_only=True)
    cases = tuple(cases)
    if not cases or len(set(cases)) != len(cases):
        raise ValueError("select at least one case without duplicates")
    metadata: dict[str, Any] = {}
    consumers: list[dict[str, Any]] = []
    workloads: dict[str, dict[str, Any]] = {}
    for case in cases:
        metadata[case], rows = _case_plan(case)
        consumers.extend(rows)
        for row in rows:
            key = row["shape_key"]
            workloads.setdefault(key, {"shape_key": key, "shape": row["shape"], "consumers": []})["consumers"].append({"case": case, "placement": row["placement"], "mode": row["mode"]})
    payload = {
        "schema_version": 1, "method": "kv-read-representative-microbenchmark-v1", "scope": SCOPE,
        "classification": "planning only; real A800 routing is input, no A800 timing",
        "cases": metadata, "consumers": consumers,
        "workloads": [workloads[key] for key in sorted(workloads)],
        "simulation_context": kv_read_cache_context(ROOT, CYCLE),
        "cycle_config_snapshot": asdict(SieveCycleV1Config.load(CYCLE)),
        "planner_sha256": sha256_file(Path(__file__)),
        "frozen_baseline_sha256": sha256_file(FROZEN),
        "program_snapshot": {
            "runner_path": Path(__file__).relative_to(ROOT).as_posix(),
            "runner_sha256": sha256_file(Path(__file__)),
            "request_model_path": "src/sieve_replay/ramulator/kv_read_workload.py",
            "request_model_sha256": sha256_file(ROOT / "src/sieve_replay/ramulator/kv_read_workload.py"),
            "frontend_path": "ramulator/extensions/sieve_kv_read/source/sieve_kv_read_frontend.cpp",
            "frontend_sha256": sha256_file(ROOT / "ramulator/extensions/sieve_kv_read/source/sieve_kv_read_frontend.cpp"),
        },
        "request_model": {
            "kv_read_bytes": "2 * sum(context_lengths) * num_key_value_heads * head_dim * dtype_bytes",
            "kv_read_transactions": "ceil(kv_read_bytes / transaction_bytes)",
            "overlap_assumption": "simultaneous Expert+KV injection is a controlled stress microbenchmark; current decode Attention -> Router -> Expert remains serial",
            "placement_scope": "GPU-only, prior Expert-only Oracle placement, and fixed hot-prefix ceil(active Experts/2) on GPU; no placement is re-optimized for KV contention",
            "fixed_half_rule": "sort by (-token_count, expert_id), assign first ceil(active_count/2) to GPU and the rest to PIM; freeze IDs per representative batch",
        },
        "summary": {"cases": list(cases), "placements": list(PLACEMENTS), "modes": list(MODES), "consumers": len(consumers), "required_shapes": len(workloads), "completed": 0, "remaining": len(workloads), "failed": 0, "interpolated_workloads": 0},
    }
    atomic_json(output / "kv_read_plan.json", payload)
    _write_csv(output / "kv_read_plan.csv", consumers)
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows({name: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for name, value in row.items()} for row in rows)


def validate_result(shape: KVWorkloadShape, result: KVReadWorkloadResult, cycle: SieveCycleV1Config) -> None:
    if any(type(value) is not int or value < 0 for value in asdict(result).values()):
        raise ValueError("invalid exact result field")
    if any(getattr(result, key) != value for key, value in asdict(shape).items()):
        raise ValueError("result shape mismatch")
    if result.tick_ps != cycle.expected_tick_ps:
        raise ValueError("result clock mismatch")
    expected_pim = (shape.pim_gwrite_waves + shape.pim_mac_waves + shape.pim_read_waves) * cycle.total_pseudo_channels
    streams = (("gpu", shape.gpu_read_transactions), ("kv", shape.kv_read_transactions), ("pim", expected_pim))
    for name, expected in streams:
        if getattr(result, name + "_injected_requests") != expected or getattr(result, name + "_completed_requests") != expected:
            raise ValueError(f"{name} request counts did not complete")
        if bool(expected) != bool(getattr(result, name + "_completion_cycles")):
            raise ValueError(f"invalid {name} completion milestone")
    if result.total_completion_cycles != max(result.gpu_completion_cycles, result.kv_completion_cycles, result.pim_completion_cycles):
        raise ValueError("invalid total completion milestone")
    if result.controller_read_completed_requests != shape.gpu_read_transactions + shape.kv_read_transactions:
        raise ValueError("controller did not complete all normal READs")
    if result.gpu_accepted_to_column_issue_cycles < 0 or result.kv_accepted_to_column_issue_cycles < 0:
        raise ValueError("invalid request residence statistics")


def _binding_provenance(cycle: SieveCycleV1Config) -> dict[str, str]:
    root = ROOT / "third_party/ramulator2"
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != cycle.ramulator_commit:
        raise ValueError("Ramulator checkout differs from pinned revision")
    installed = {
        "sieve_kv_read_frontend.cpp": ("sieve_kv_read", "frontend/impl/memory_trace"),
        "sieve_hbm_pim_controller_v1.cpp": ("sieve_hbm_pim", "controller/impl"),
    }
    for name, (extension, directory) in installed.items():
        if sha256_file(ROOT / "ramulator/extensions" / extension / "source" / name) != sha256_file(root / "src/ramulator" / directory / name):
            raise ValueError(f"installed Ramulator source differs: {name}")
    bindings = sorted((root / "python/ramulator").glob("_ramulator.cpython-*.so"))
    if len(bindings) != 1:
        raise ValueError("expected exactly one Ramulator Python binding")
    library = root / "libramulator.so"
    if not library.is_file():
        raise ValueError("Ramulator shared library is missing")
    return {
        "ramulator_commit": revision,
        "binding_sha256": sha256_file(bindings[0]),
        "ramulator_library_sha256": sha256_file(library),
    }


def _run_shape(shape: dict[str, int]) -> dict[str, int]:
    return asdict(run_kv_read_workload(ROOT / "third_party/ramulator2", SieveCycleV1Config.load(CYCLE), **shape))


def run_exact(plan_path: Path, output: Path = OUTPUT, workers: int = 1) -> dict[str, Any]:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    with (CACHE_ROOT / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_exact_locked(plan_path, output, workers)


def _run_exact_locked(plan_path: Path, output: Path, workers: int) -> dict[str, Any]:
    freeze_or_verify(verify_only=True)
    payload = read_json(plan_path)
    if payload.get("method") != "kv-read-representative-microbenchmark-v1" or payload.get("planner_sha256") != sha256_file(Path(__file__)):
        raise ValueError("unsupported or stale KV READ plan")
    cycle = SieveCycleV1Config.load(CYCLE)
    if payload.get("frozen_baseline_sha256") != sha256_file(FROZEN):
        raise ValueError("frozen baseline inventory changed after planning")
    if payload.get("simulation_context") != kv_read_cache_context(ROOT, CYCLE) or payload.get("cycle_config_snapshot") != asdict(cycle):
        raise ValueError("simulation inputs changed after planning")
    for case, metadata in payload["cases"].items():
        current, consumers = _case_plan(case)
        if current != metadata or consumers != [row for row in payload["consumers"] if row["case"] == case]:
            raise ValueError(f"{case}: trace or placement inputs changed after planning")
    expected_workloads: dict[str, Any] = {}
    for row in payload["consumers"]:
        key = row["shape_key"]
        expected_workloads.setdefault(key, {"shape_key": key, "shape": row["shape"], "consumers": []})["consumers"].append({"case": row["case"], "placement": row["placement"], "mode": row["mode"]})
    if payload["workloads"] != [expected_workloads[key] for key in sorted(expected_workloads)]:
        raise ValueError("planned workloads differ from verified consumers")
    execution = _binding_provenance(cycle)
    plan_hash = sha256_file(plan_path)
    context = {**payload["simulation_context"], **execution, "plan_sha256": plan_hash}
    evidence_path = output / "kv_read_exact_results.json"
    previous = read_json(evidence_path) if evidence_path.is_file() else {}
    previous_entries = previous.get("entries", {}) if previous.get("cache_context") == context else {}
    entries: dict[str, Any] = {}
    pending: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def cache_path(shape: dict[str, int]) -> Path:
        return CACHE_ROOT / (digest({"context": context, "shape": shape}) + ".json")

    def accept(row: dict[str, Any], result: KVReadWorkloadResult, origin: str) -> None:
        validate_result(KVWorkloadShape(**row["shape"]), result, cycle)
        path = cache_path(row["shape"])
        if origin == "new-exact-run":
            atomic_json(path, {"cache_input": {"context": context, "shape": row["shape"]}, "result": asdict(result)})
        entries[row["shape_key"]] = {**row, "origin": origin, "cache_file_sha256": sha256_file(path), "result": asdict(result)}

    for row in payload["workloads"]:
        path = cache_path(row["shape"])
        if path.is_file():
            cached = read_json(path)
            if cached.get("cache_input") != {"context": context, "shape": row["shape"]}:
                raise ValueError("exact KV cache provenance mismatch")
            old = previous_entries.get(row["shape_key"])
            if old and old.get("cache_file_sha256") != sha256_file(path):
                raise ValueError("exact KV cache differs from saved evidence hash")
            accept(row, KVReadWorkloadResult(**cached["result"]), "exact-cache")
        else:
            pending.append(row)

    def save() -> dict[str, Any]:
        summary = {"required_shapes": len(payload["workloads"]), "completed": len(entries), "remaining": len(payload["workloads"]) - len(entries), "failed": len(failures), "interpolated_workloads": 0}
        atomic_json(output / "kv_read_exact_results.json", {"schema_version": 1, "method": payload["method"], "scope": SCOPE, "classification": "exact Ramulator controlled simultaneous-injection microbenchmark, not A800 timing or end-to-end replay", "plan_sha256": sha256_file(plan_path), "cache_context": context, "program_snapshot": payload["program_snapshot"], "summary": summary, "entries": entries, "failures": failures})
        return summary

    save()
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_shape, row["shape"]): row for row in pending}
            for future in as_completed(futures):
                row = futures[future]
                try:
                    accept(row, KVReadWorkloadResult(**future.result()), "new-exact-run")
                except Exception as exc:
                    failures.append({"shape_key": row["shape_key"], "shape": row["shape"], "error": str(exc)})
                print(json.dumps(save(), sort_keys=True), flush=True)
    summary = save()
    freeze_or_verify(verify_only=True)
    if (_binding_provenance(cycle) != execution
            or kv_read_cache_context(ROOT, CYCLE) != payload["simulation_context"]
            or sha256_file(plan_path) != plan_hash
            or sha256_file(Path(__file__)) != payload["planner_sha256"]):
        raise ValueError("source, binary or plan provenance changed during exact execution")
    if summary["remaining"] == 0 and summary["failed"] == 0:
        report(payload, entries, output)
    return summary


def report(plan_payload: dict[str, Any], entries: dict[str, Any], output: Path) -> None:
    rows: list[dict[str, Any]] = []
    for case in plan_payload["cases"]:
        for placement in PLACEMENTS:
            consumers = {row["mode"]: row for row in plan_payload["consumers"] if row["case"] == case and row["placement"] == placement}
            selected = {mode: KVReadWorkloadResult(**entries[row["shape_key"]]["result"]) for mode, row in consumers.items()}
            expert, kv, combined = (selected[mode] for mode in MODES)
            def mean_us(cycles: int, count: int) -> float:
                return cycles * combined.tick_ps / 1_000_000.0 / count if count else 0.0
            rows.append({
                "case": case, "placement": placement, "scope": SCOPE,
                "gpu_experts": len(consumers["combined"]["gpu_experts"]),
                "pim_experts": len(consumers["combined"]["pim_experts"]),
                "kv_read_bytes_per_layer": consumers["combined"]["kv_read_bytes_per_layer"],
                "expert_only_us": expert.total_completion_us,
                "kv_only_us": kv.total_completion_us,
                "combined_us": combined.total_completion_us,
                "serial_memory_reference_us": expert.total_completion_us + kv.total_completion_us,
                "combined_minus_isolated_max_us": combined.total_completion_us - max(expert.total_completion_us, kv.total_completion_us),
                "gpu_expert_completion_delta_us": combined.gpu_completion_us - expert.gpu_completion_us,
                "pim_expert_completion_delta_us": combined.pim_completion_us - expert.pim_completion_us,
                "kv_completion_delta_us": combined.kv_completion_us - kv.kv_completion_us,
                "gpu_expert_only_completion_us": expert.gpu_completion_us,
                "gpu_combined_completion_us": combined.gpu_completion_us,
                "pim_expert_only_completion_us": expert.pim_completion_us,
                "pim_combined_completion_us": combined.pim_completion_us,
                "kv_isolated_completion_us": kv.kv_completion_us,
                "kv_combined_completion_us": combined.kv_completion_us,
                "gpu_requested": combined.gpu_read_transactions,
                "gpu_injected": combined.gpu_injected_requests,
                "gpu_completed": combined.gpu_completed_requests,
                "kv_requested": combined.kv_read_transactions,
                "kv_injected": combined.kv_injected_requests,
                "kv_completed": combined.kv_completed_requests,
                "pim_injected": combined.pim_injected_requests,
                "pim_completed": combined.pim_completed_requests,
                "controller_read_completed": combined.controller_read_completed_requests,
                "gpu_request_residence_cycles": combined.gpu_request_residence_cycles,
                "kv_request_residence_cycles": combined.kv_request_residence_cycles,
                "gpu_accepted_to_column_issue_cycles": combined.gpu_accepted_to_column_issue_cycles,
                "kv_accepted_to_column_issue_cycles": combined.kv_accepted_to_column_issue_cycles,
                "gpu_request_residence_mean_us": mean_us(combined.gpu_request_residence_cycles, combined.gpu_completed_requests),
                "kv_request_residence_mean_us": mean_us(combined.kv_request_residence_cycles, combined.kv_completed_requests),
                "gpu_accepted_to_column_issue_mean_us": mean_us(combined.gpu_accepted_to_column_issue_cycles, combined.gpu_completed_requests),
                "kv_accepted_to_column_issue_mean_us": mean_us(combined.kv_accepted_to_column_issue_cycles, combined.kv_completed_requests),
                "gpu_request_residence_max_us": combined.gpu_max_request_residence_cycles * combined.tick_ps / 1_000_000.0,
                "kv_request_residence_max_us": combined.kv_max_request_residence_cycles * combined.tick_ps / 1_000_000.0,
                "gpu_isolated_request_residence_mean_us": mean_us(expert.gpu_request_residence_cycles, expert.gpu_completed_requests),
                "kv_isolated_request_residence_mean_us": mean_us(kv.kv_request_residence_cycles, kv.kv_completed_requests),
                "gpu_injection_rejected_attempts": combined.gpu_injection_rejected_attempts,
                "kv_injection_rejected_attempts": combined.kv_injection_rejected_attempts,
                "expert_shape_key": consumers["expert-only"]["shape_key"],
                "kv_shape_key": consumers["kv-only"]["shape_key"],
                "combined_shape_key": consumers["combined"]["shape_key"],
            })
    limitations = [
        "These are memory microbenchmarks under forced simultaneous injection, not end-to-end timings",
        "Frozen Oracle placement is inherited from the Expert-only baseline and is not re-optimized",
        "KV READs are abstract sequential requests, not an A800 memory trace",
        "Accepted-to-column-issue cycles include arbitration and ACT/PRE preparation; they are not pure FIFO wait",
        "No paging, spill, eviction or Shared CXL-PIM",
    ]
    _write_csv(output / "kv_read_comparison.csv", rows)
    atomic_json(output / "kv_read_comparison.json", {
        "scope": SCOPE, "rows": rows, "limitations": limitations,
        "plan_sha256": sha256_file(output / "kv_read_plan.json"),
        "exact_results_sha256": sha256_file(output / "kv_read_exact_results.json"),
        "frozen_baseline_sha256": sha256_file(FROZEN),
        "mode_definitions": {
            "expert-only": "Exact Ramulator Expert requests for the fixed placement; zero KV READ",
            "kv-only": "Exact Ramulator abstract KV READ stream; zero Expert requests",
            "combined": "Exact Ramulator controlled simultaneous Expert+KV injection; not the serial decode graph",
        },
    })
    lines = [
        "# KV READ 代表性请求级竞争实验", "",
        "每组仅使用已独立捕获真实 Router 的 step=0、layer=0。36 个配置消费者共享去重后的精确 Ramulator workload；本报告不覆盖完整 384 个 layer-batch。", "",
        "Expert-only、KV-only 与 combined 使用同一请求模型和固定放置；combined 强制同时注入 KV 与 Expert 请求。当前端到端图仍按 Attention → Router → Expert 串行，因此下面是受控竞争微基准，不能直接作为端到端时延或真实 A800 带宽结论。", "",
        "| 配置 | 放置 | GPU/PIM experts | Expert-only (us) | KV-only (us) | Combined (us) | Combined − max(isolated) (us) |", 
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['case']} | {row['placement']} | {row['gpu_experts']}/{row['pim_experts']} | {row['expert_only_us']:.6f} | {row['kv_only_us']:.6f} | {row['combined_us']:.6f} | {row['combined_minus_isolated_max_us']:.6f} |")
    lines += ["", "请求注入/完成计数、控制器 READ 完成数、请求驻留时间、接收到列命令发出前的周期以及注入被拒次数均保存在 JSON/CSV。接收到列命令前的周期包含仲裁及 ACT/PRE 准备，不能解释为纯 FIFO 排队。", "", "固定半数放置按 (-token_count, expert_id) 排序，前 ceil(active/2) 个专家分给 GPU；本轮代表性层均为 4 GPU + 4 PIM。它是独立的典型混合放置，并非旧 fixed16 消融。Oracle 放置继承旧 Expert-only 回放，没有按新 KV 竞争重新优化。", "", "A800 提供真实路由输入，不提供本报告时延；模拟范围仍为 1 GPU + 8 Local HBM-PIM stacks。没有新增 paging、spill、eviction 或 Shared CXL-PIM。", ""]
    (output / "kv_read_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, action="append")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--plan-only", action="store_true", help="only write the plan (default)")
    action.add_argument("--run", action="store_true", help="execute the complete small plan exactly")
    args = parser.parse_args(argv)
    if args.workers <= 0:
        parser.error("--workers must be positive")
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    with (CACHE_ROOT / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        payload = plan(args.output, args.case or CASES)
        summary = _run_exact_locked(args.output / "kv_read_plan.json", args.output, args.workers) if args.run else payload["summary"]
    print(json.dumps(summary, sort_keys=True))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
