#!/usr/bin/env python3
"""Materialize and replay the five KV experiments only after all exact runs pass."""
from __future__ import annotations

import json
import fcntl
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fill_kv_workloads import (CASES, CACHE, CYCLE, EVIDENCE, atomic_json,
                               portable, read_exact)
from build_kv_isolated_tables import expected_attestation, attestation_matches
from sieve_replay.cli import main as replay_main
from sieve_replay.config import load_configuration
from sieve_replay.report.writer import sha256_file
from sieve_replay.ramulator.decode_contention_table import build_decode_contention_table
from sieve_replay.ramulator.mixed_workload import SieveCycleV1Config
from sieve_replay.ramulator.workload_cache import ContentionWorkloadCache, WorkloadShape, cache_context
from sieve_replay.ramulator.workload_catalog import build_workload_catalog
from sieve_replay.timing import RamulatorContentionTable, RamulatorTimingTable
from sieve_replay.trace import load_trace_set
from sieve_replay.trace_manifest import TraceManifest


def validate_inputs() -> tuple[dict, dict]:
    evidence = json.loads(EVIDENCE.read_text())
    context = cache_context(ROOT, CYCLE)
    cycle = SieveCycleV1Config.load(CYCLE)
    cache = ContentionWorkloadCache(CACHE, context)
    if evidence["cache_context"] != context:
        raise ValueError("exact-workload evidence has stale simulation hashes")
    summary = evidence["summary"]
    if (summary["failed"] != 0 or summary["remaining"] != 0
            or summary["completed"] != summary["required_unique_shapes"]
            or evidence["interpolated_workloads"] != 0):
        raise ValueError("all exact workloads must complete without failures or interpolation")
    if len(evidence["entries"]) != summary["required_unique_shapes"]:
        raise ValueError("exact-workload evidence is incomplete")
    for key, entry in evidence["entries"].items():
        shape = WorkloadShape(**entry["shape"])
        path = cache.path(shape)
        if (key != cache.key(shape) or sha256_file(path) != entry["cache_file_sha256"]
                or read_exact(path, shape, context, cycle) is None):
            raise ValueError(f"exact cache entry failed verification: {key}")

    configs = {}
    for case in CASES:
        experiment = ROOT / f"configs/experiments/full_decode_real_kv_{case}.json"
        cfg = load_configuration(experiment)
        e = cfg.experiment
        traces = load_trace_set(e.trace_path, cfg.model, e.layers, e.steps)
        manifest = TraceManifest.load_and_validate(e.trace_manifest_path, e.trace_path, cfg.model, traces)
        saved = evidence["cases"][case]
        paths = {"experiment": experiment, "trace": e.trace_path, "trace_manifest": e.trace_manifest_path,
                 "prompts": e.trace_manifest_path.parent / manifest.raw["prompts"]["file"], "model": e.model_path,
                 "hardware": e.hardware_path, "cycle_config": CYCLE}
        hashes = {k + "_sha256": sha256_file(v) for k, v in paths.items()}
        if saved["input_hashes"] != hashes or saved["configuration"] != e.raw:
            raise ValueError(f"experiment inputs changed after planning: {case}")
        catalog = build_workload_catalog(experiment, CYCLE, CACHE,
                                         CACHE / f"{case}.catalog.json")
        if catalog["summary"]["missing_workload_shapes"] != 0:
            raise ValueError(f"exact cache still missing workloads: {case}")
        keys = sorted(row["cache_key"] for row in catalog["workloads"])
        if saved["workload_keys"] != keys:
            raise ValueError(f"planned workload keys changed: {case}")
        table = RamulatorTimingTable.load(e.pim_timing_table_path)
        for batch in traces.batches:
            if table.schema_version == 1:
                if len(set(batch.context_lengths)) != 1:
                    raise ValueError("uniform-context timing table used for nonuniform inputs")
                table.attention_us(batch.batch_size, batch.context_lengths[0])
            else:
                table.attention_us(batch.context_lengths)
        isolated_evidence = e.pim_timing_table_path.with_suffix(".evidence.json")
        expected = expected_attestation(experiment, ROOT / "configs/ramulator/sieve_hbm3e_cycle_v0.json")
        if not attestation_matches(isolated_evidence, e.pim_timing_table_path, expected):
            raise ValueError(f"isolated table does not attest the current trace and inputs: {case}")
        isolated = json.loads(isolated_evidence.read_text())
        if (len(isolated["runs"]) != 3
                or any(run["injected_requests"] != run["completed_requests"] for run in isolated["runs"])):
            raise ValueError(f"isolated PIM command stream incomplete: {case}")
        configs[case] = cfg
    return evidence, configs


def run_stage() -> int:
    evidence, configs = validate_inputs()
    output = ROOT / "results/full_decode_real_kv_cycle_v1"
    stage = {"schema_version": 1, "exact_workloads_sha256": sha256_file(EVIDENCE), "cases": {},
             "interpolated_workloads": 0, "new_a800_measurements": False,
             "classification": "A800-captured routing with simulated 1 GPU + 8 Local HBM-PIM stacks"}
    for case, cfg in configs.items():
        experiment = ROOT / f"configs/experiments/full_decode_real_kv_{case}.json"
        e = cfg.experiment
        table = build_decode_contention_table(experiment, CYCLE, CACHE, e.contention_timing_table_path)
        RamulatorContentionTable.load(e.contention_timing_table_path).validate_inputs(
            e.pim_timing_table_path, e.trace_path, CYCLE, e.model_path, e.hardware_path)
        proof_path = e.contention_timing_table_path.with_suffix(".evidence.json")
        proof = json.loads(proof_path.read_text())
        proof.update({"input_hashes": evidence["cases"][case]["input_hashes"],
                      "configuration_snapshot": e.raw, "exact_workloads_sha256": sha256_file(EVIDENCE),
                      "table_sha256": sha256_file(e.contention_timing_table_path),
                      "interpolated_workloads": 0, "missing_workload_shapes": 0, "failed": 0})
        atomic_json(proof_path, proof)
        status = replay_main(["run-all", "--experiment", str(experiment), "--output", str(output / case)])
        if status:
            raise RuntimeError(f"replay failed: {case}")
        stage["cases"][case] = {
            "input_hashes": evidence["cases"][case]["input_hashes"],
            "configuration_snapshot": e.raw,
            "contention_entries": len(table["expert_contention"]),
            "exact_workload_shapes": len(evidence["cases"][case]["workload_keys"]),
            "pim_timing_table_sha256": sha256_file(e.pim_timing_table_path),
            "pim_evidence_sha256": sha256_file(e.pim_timing_table_path.with_suffix(".evidence.json")),
            "contention_timing_table_sha256": sha256_file(e.contention_timing_table_path),
            "contention_evidence_sha256": sha256_file(proof_path),
            "replays": {policy: {"summary_sha256": sha256_file(output / case / policy / "summary.json"),
                                  "run_manifest_sha256": sha256_file(output / case / policy / "run_manifest.json"),
                                  "layers_sha256": sha256_file(output / case / policy / "layers.csv"),
                                  "events_sha256": sha256_file(output / case / policy / "events.csv")}
                         for policy in e.policies}}
        atomic_json(output / "stage_manifest.json", stage)
        print(json.dumps({"completed_case": case, "entries": len(table["expert_contention"])}), flush=True)
    return 0


def main() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    with (CACHE / ".stage.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_stage()


if __name__ == "__main__":
    raise SystemExit(main())
