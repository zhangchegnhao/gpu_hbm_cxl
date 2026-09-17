#!/usr/bin/env python3
"""Create once, then verify the completed long-context baseline's file hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sieve_replay.report.writer import sha256_file

CASES = ("b8_c4k", "b8_c8k", "b8_c16k", "b16_c4k", "b16_c8k")
FROZEN = ROOT / "results/kv_read_experiment_v1/frozen_baseline.json"


def baseline_paths() -> list[Path]:
    paths = set()
    for directory in ("results/full_decode_real_kv_cycle_v1", "results/kv_capacity_sweep_v1"):
        paths.update(p for p in (ROOT / directory).rglob("*") if p.is_file())
    for case in CASES:
        cfg = ROOT / f"configs/experiments/full_decode_real_kv_{case}.json"
        raw = json.loads(cfg.read_text())
        paths.add(cfg)
        paths.update(ROOT / raw[k] for k in ("model", "hardware", "ramulator_cycle_config"))
        for key in ("pim_timing_table", "contention_timing_table"):
            table = ROOT / raw[key]
            paths.update((table, table.with_suffix(".evidence.json")))
        trace_dir = (ROOT / raw["trace_manifest"]).parent
        paths.update(trace_dir / name for name in ("prompts.jsonl", "router.jsonl", "manifest.json"))
    paths.add(ROOT / "results/kv_workload_timing_v1/exact_workloads.json")
    paths.update(ROOT / "docs" / name for name in (
        "kv_capacity_sweep_v1.md", "kv_long_context_capture_v1.md", "kv_long_context_cycle_v1_analysis.md"))
    for directory in ("src/sieve_replay/ramulator", "ramulator/extensions/sieve_hbm_pim"):
        paths.update(p for p in (ROOT / directory).rglob("*")
                     if p.suffix in {".py", ".cpp", ".patch"} and not p.name.startswith("kv_read"))
    return sorted(paths)


def freeze_or_verify(path: Path = FROZEN, *, verify_only: bool = False) -> dict:
    if path.exists():
        value = json.loads(path.read_text())
        for relative, digest in value["files"].items():
            source = ROOT / relative
            if not source.is_file() or sha256_file(source) != digest:
                raise ValueError(f"frozen baseline changed: {relative}")
        return value
    if verify_only:
        raise FileNotFoundError(path)
    files = {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in baseline_paths()}
    value = {
        "schema_version": 1,
        "classification": "Frozen real-router baseline; GPU analytic and Ramulator PIM/Expert timing",
        "scope": "5 cases, 7 policies, 384 layer-batches per case and policy",
        "files": files,
        "inventory_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "rule": "Create once; verify before and after KV experiments. Never refresh to accept changed baseline bytes.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    value = freeze_or_verify(verify_only=args.verify)
    print(json.dumps({"verified_files": len(value["files"]), "inventory_sha256": value["inventory_sha256"]}))


if __name__ == "__main__":
    main()
