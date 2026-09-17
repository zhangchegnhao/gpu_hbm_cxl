#!/usr/bin/env python3
"""Fill the union of exact KV-stage expert workloads with strict provenance."""
from __future__ import annotations

import argparse
import fcntl
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sieve_replay.config import load_configuration
from sieve_replay.report.writer import sha256_file
from sieve_replay.trace import load_trace_set
from sieve_replay.trace_manifest import TraceManifest
from sieve_replay.ramulator.mixed_workload import (
    MixedWorkloadResult, SieveCycleV1Config, run_mixed_workload,
)
from sieve_replay.ramulator.workload_cache import (
    ContentionWorkloadCache, WorkloadShape, cache_context,
)
from sieve_replay.ramulator.workload_catalog import build_workload_catalog

CASES = ("b8_c4k", "b8_c8k", "b8_c16k", "b16_c4k", "b16_c8k")
CYCLE = ROOT / "configs/ramulator/sieve_hbm3e_cycle_v1.json"
CACHE = ROOT / "ramulator/timing_tables/generated/.cache/kv_long_context_cycle_v1"
EVIDENCE = ROOT / "results/kv_workload_timing_v1/exact_workloads.json"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def portable(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def validate_result(shape: WorkloadShape, result: MixedWorkloadResult,
                    cycle: SieveCycleV1Config) -> None:
    for name, value in asdict(result).items():
        if type(value) is not int or value < 0:
            raise ValueError(f"invalid exact result field: {name}")
    if any(getattr(result, key) != value for key, value in asdict(shape).items()):
        raise ValueError("cached result does not describe the requested shape")
    if result.tick_ps != cycle.expected_tick_ps:
        raise ValueError("cached result has a different clock")
    pim = (shape.pim_gwrite_waves + shape.pim_mac_waves + shape.pim_read_waves) * cycle.total_pseudo_channels
    if (result.gpu_injected_requests != shape.gpu_read_transactions
            or result.gpu_completed_requests != shape.gpu_read_transactions
            or result.pim_injected_requests != pim or result.pim_completed_requests != pim):
        raise ValueError("cached stream did not complete all requests")
    if bool(shape.gpu_read_transactions) != bool(result.gpu_completion_cycles):
        raise ValueError("invalid GPU completion milestone")
    if bool(pim) != bool(result.pim_completion_cycles):
        raise ValueError("invalid PIM completion milestone")
    if result.total_completion_cycles != max(result.gpu_completion_cycles, result.pim_completion_cycles):
        raise ValueError("invalid total completion milestone")
    if not (result.pim_gwrite_completion_cycles <= result.pim_mac_completion_cycles
            <= result.pim_read_completion_cycles == result.pim_completion_cycles):
        raise ValueError("invalid PIM pipeline milestones")


def read_exact(path: Path, shape: WorkloadShape, context: dict,
               cycle: SieveCycleV1Config) -> MixedWorkloadResult | None:
    raw = json.loads(path.read_text())
    if raw.get("cache_input") != {"context": context, "shape": asdict(shape)}:
        return None
    result = MixedWorkloadResult(**raw["result"])
    validate_result(shape, result, cycle)
    return result


def run_shape(shape: WorkloadShape) -> tuple[MixedWorkloadResult, float]:
    start = time.monotonic()
    result = run_mixed_workload(ROOT / "third_party/ramulator2",
                                SieveCycleV1Config.load(CYCLE), **asdict(shape))
    return result, time.monotonic() - start


def verify_binding_sources(cycle: SieveCycleV1Config) -> dict:
    ramulator = ROOT / "third_party/ramulator2"
    revision = subprocess.check_output(["git", "-C", str(ramulator), "rev-parse", "HEAD"], text=True).strip()
    if revision != cycle.ramulator_commit:
        raise ValueError("Ramulator checkout differs from pinned revision")
    installed = {
        "sieve_mixed_frontend.cpp": "frontend/impl/memory_trace",
        "sieve_hbm_pim_controller_v1.cpp": "controller/impl",
    }
    for name, directory in installed.items():
        if sha256_file(ROOT / "ramulator/extensions/sieve_hbm_pim/source" / name) != sha256_file(ramulator / "src/ramulator" / directory / name):
            raise ValueError(f"installed extension differs from source: {name}")
    binding = sorted((ramulator / "python/ramulator").glob("_ramulator.cpython-*.so"))
    if len(binding) != 1:
        raise ValueError("expected exactly one production Ramulator Python binding")
    return {"ramulator_commit": revision, "binding_sha256": sha256_file(binding[0])}


def fill(workers: int, plan_only: bool = False) -> dict:
    cycle = SieveCycleV1Config.load(CYCLE)
    context = cache_context(ROOT, CYCLE)
    cache = ContentionWorkloadCache(CACHE, context)
    binding = verify_binding_sources(cycle)
    if (CACHE / "empty_planning_cache").exists():
        raise ValueError("the planning path must stay empty")
    shapes: dict[str, WorkloadShape] = {}
    case_evidence = {}
    for case in CASES:
        path = ROOT / f"configs/experiments/full_decode_real_kv_{case}.json"
        cfg = load_configuration(path)
        e = cfg.experiment
        traces = load_trace_set(e.trace_path, cfg.model, e.layers, e.steps)
        if e.trace_manifest_path is None:
            raise ValueError("KV traces require a manifest")
        manifest = TraceManifest.load_and_validate(e.trace_manifest_path, e.trace_path, cfg.model, traces)
        if len(traces.batches) != 384:
            raise ValueError("expected 384 KV layer-batches")
        # Plan against an empty path so the legacy cache fallback never accepts
        # an entry missing one of the current provenance fields.
        catalog = build_workload_catalog(path, CYCLE, CACHE / "empty_planning_cache",
                                         CACHE / f"{case}.catalog.json")
        keys = []
        for row in catalog["workloads"]:
            shape = WorkloadShape(**row["shape"])
            key = cache.key(shape)
            shapes[key] = shape
            keys.append(key)
        paths = {"experiment": path, "trace": e.trace_path, "trace_manifest": e.trace_manifest_path,
                 "prompts": e.trace_manifest_path.parent / manifest.raw["prompts"]["file"],
                 "model": e.model_path, "hardware": e.hardware_path, "cycle_config": CYCLE}
        case_evidence[case] = {"input_hashes": {k + "_sha256": sha256_file(v) for k, v in paths.items()},
                               "configuration": e.raw, "layer_batches": len(traces.batches),
                               "workload_keys": sorted(keys), "unique_workload_shapes": len(keys),
                               "placement_consumers": catalog["summary"]["placement_consumers"]}

    previous = json.loads(EVIDENCE.read_text()) if EVIDENCE.is_file() else {}
    if previous and previous["cache_context"] != context:
        raise ValueError("existing stage evidence has different simulation inputs")
    entries = {}
    for key, shape in shapes.items():
        path = cache.path(shape)
        if path.is_file():
            result = read_exact(path, shape, context, cycle)
            if result is None:
                raise ValueError(f"invalid destination provenance: {key}")
            entry = previous.get("entries", {}).get(key)
            if entry and entry["cache_file_sha256"] != sha256_file(path):
                raise ValueError(f"destination cache changed: {key}")
            entries[key] = entry or {"origin": "existing-exact-cache", "shape": asdict(shape),
                                      "cache_file_sha256": sha256_file(path), "source": portable(path)}

    sources = (ROOT / "ramulator/timing_tables/generated/.cache/qwen3_real_pilot_cycle_v1",
               ROOT / "ramulator/timing_tables/generated/.cache/kv_b8_c4k")
    for directory in sources:
        for path in sorted(directory.glob("*.json")):
            raw = json.loads(path.read_text())
            incoming = raw.get("cache_input", {})
            if incoming.get("context") != context:
                continue
            shape = WorkloadShape(**incoming["shape"])
            key = cache.key(shape)
            if key not in shapes:
                continue
            result = read_exact(path, shape, context, cycle)
            if key in entries:
                if result != read_exact(cache.path(shape), shape, context, cycle):
                    raise ValueError(f"conflicting exact caches: {key}")
                continue
            atomic_json(cache.path(shape), raw)
            entries[key] = {"origin": "reused-exact-shape", "shape": asdict(shape),
                            "source": portable(path), "source_sha256": sha256_file(path),
                            "cache_file_sha256": sha256_file(cache.path(shape))}

    source_table = ROOT / "ramulator/timing_tables/generated/qwen3_real_pilot_decode_cycle_v1_contention.json"
    source_evidence = source_table.with_suffix(".evidence.json")
    source_metadata = json.loads(source_table.read_text())["metadata"]
    if (source_metadata["ramulator_commit"] != cycle.ramulator_commit
            or source_metadata["cycle_config_sha256"] != context["cycle_config_sha256"]
            or source_metadata["extension_commit"] != "cycle-v1-sha256:" + context["extension_sha256"]):
        raise ValueError("historical exact-cache attestation differs from current simulation inputs")
    evidence = {"schema_version": 1, "cache_context": context, "execution": binding,
                "cases": case_evidence, "entries": entries, "failures": [],
                "source_attestation": {"table": portable(source_table), "table_sha256": sha256_file(source_table),
                                       "evidence_sha256": sha256_file(source_evidence), "metadata": source_metadata},
                "runner_sha256": sha256_file(Path(__file__)),
                "provenance_note": "Cycle hash includes pinned revision; historical caches retain source hashes, not their original binary hash. execution.binding_sha256 identifies the current binding only.",
                "classification": "exact Ramulator expert request shapes; no KV READ contention",
                "reuse_rule": "identical shape and every cache_context hash; no legacy fallback",
                "interpolated_workloads": 0}

    def save() -> None:
        evidence["summary"] = {"required_unique_shapes": len(shapes), "completed": len(entries),
                               "remaining": len(shapes) - len(entries), "failed": len(evidence["failures"]),
                               "reused_exact_shapes": sum(e["origin"] == "reused-exact-shape" for e in entries.values()),
                               "new_exact_shapes": sum(e["origin"] == "new-exact-run" for e in entries.values())}
        atomic_json(EVIDENCE, evidence)

    save()
    print(json.dumps(evidence["summary"], sort_keys=True), flush=True)
    missing = [shape for key, shape in shapes.items() if key not in entries]
    missing.sort(key=lambda s: s.gpu_read_transactions + cycle.total_pseudo_channels *
                 (s.pim_gwrite_waves + s.pim_mac_waves + s.pim_read_waves), reverse=True)
    if not plan_only and missing:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_shape, s): s for s in missing}
            for future in as_completed(futures):
                shape = futures[future]
                key = cache.key(shape)
                try:
                    result, seconds = future.result()
                    validate_result(shape, result, cycle)
                    atomic_json(cache.path(shape), {"cache_input": {"shape": asdict(shape), "context": context},
                                                     "result": asdict(result)})
                    entries[key] = {"origin": "new-exact-run", "shape": asdict(shape), "wall_time_s": round(seconds, 3),
                                    "cache_file_sha256": sha256_file(cache.path(shape))}
                except Exception as error:
                    evidence["failures"].append({"key": key, "error": repr(error)})
                save()
                print(json.dumps(evidence["summary"], sort_keys=True), flush=True)
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("workers must be positive")
    CACHE.mkdir(parents=True, exist_ok=True)
    with (CACHE / ".stage.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        evidence = fill(args.workers, args.plan_only)
    return int(bool(evidence["summary"]["failed"] or (not args.plan_only and evidence["summary"]["remaining"])))


if __name__ == "__main__":
    raise SystemExit(main())
