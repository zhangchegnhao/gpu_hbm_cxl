#!/usr/bin/env python3
"""Build and attest exact cycle-v0 tables for a real KV trace.

The underlying timing table remains schema-v1/v2.  The companion evidence file
is extended with hashes of every input and of the generated artifacts so a
partially completed run can be resumed safely without reusing stale results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sieve_replay.ramulator import build_timing_table  # noqa: E402
from sieve_replay.config import load_configuration  # noqa: E402
from sieve_replay.trace import load_trace_set  # noqa: E402
from sieve_replay.trace_manifest import TraceManifest  # noqa: E402


CASES = {
    "b8_c4k": "configs/experiments/full_decode_real_kv_b8_c4k.json",
    "b8_c8k": "configs/experiments/full_decode_real_kv_b8_c8k.json",
    "b8_c16k": "configs/experiments/full_decode_real_kv_b8_c16k.json",
    "b16_c4k": "configs/experiments/full_decode_real_kv_b16_c4k.json",
    "b16_c8k": "configs/experiments/full_decode_real_kv_b16_c8k.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def combined_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def input_paths(experiment_path: Path, cycle_path: Path) -> dict[str, Path]:
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    manifest_path = PROJECT_ROOT / experiment["trace_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prompts_file = manifest.get("prompts", {}).get("file", "prompts.jsonl")
    paths = {
        "experiment_sha256": experiment_path,
        "model_sha256": PROJECT_ROOT / experiment["model"],
        "hardware_sha256": PROJECT_ROOT / experiment["hardware"],
        "trace_sha256": PROJECT_ROOT / experiment["trace"],
        "trace_manifest_sha256": PROJECT_ROOT / experiment["trace_manifest"],
        "prompts_sha256": manifest_path.parent / prompts_file,
        "cycle_config_sha256": cycle_path,
    }
    return {key: path.resolve() for key, path in paths.items()}


def expected_attestation(experiment_path: Path, cycle_path: Path) -> dict[str, Any]:
    paths = input_paths(experiment_path, cycle_path)
    extension_root = PROJECT_ROOT / "ramulator/extensions/sieve_hbm_pim"
    package_root = PROJECT_ROOT / "src/sieve_replay/ramulator"
    generator_paths = [package_root / "microbenchmark.py", package_root / "table_builder.py"]
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    cycle = json.loads(cycle_path.read_text(encoding="utf-8"))
    model = json.loads((PROJECT_ROOT / experiment["model"]).read_text(encoding="utf-8"))
    hardware = json.loads((PROJECT_ROOT / experiment["hardware"]).read_text(encoding="utf-8"))
    return {
        "input_hashes": {name: sha256(path) for name, path in paths.items()},
        "generator_sha256": combined_sha256(generator_paths),
        "extension_sha256": combined_sha256(
            sorted(extension_root.rglob("*.cpp")) + sorted(extension_root.rglob("*.patch"))
        ),
        "snapshot": {
            "experiment": str(experiment_path.relative_to(PROJECT_ROOT)),
            "cycle_config": str(cycle_path.relative_to(PROJECT_ROOT)),
            "ramulator_root": str((PROJECT_ROOT / "third_party/ramulator2").relative_to(PROJECT_ROOT)),
            "timing_source": "exact Ramulator cycle-v0 PIM milestones; no interpolation; no KV READ requests",
            "experiment_config": experiment,
            "model_config": model,
            "hardware_config": hardware,
            "cycle_config_snapshot": cycle,
        },
    }


def attestation_matches(evidence_path: Path, table_path: Path, expected: dict[str, Any]) -> bool:
    if not evidence_path.is_file() or not table_path.is_file():
        return False
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    runs = evidence.get("runs", [])
    return (
        evidence.get("input_hashes") == expected["input_hashes"]
        and evidence.get("generator_sha256") == expected["generator_sha256"]
        and evidence.get("extension_sha256") == expected["extension_sha256"]
        and evidence.get("table_sha256") == sha256(table_path)
        and evidence.get("snapshot") == expected["snapshot"]
        and evidence.get("interpolation") == "forbidden"
        and evidence.get("exact_workload") is True
        and evidence.get("runs_request_counts_match") is True
        and isinstance(runs, list)
        and bool(runs)
        and all(row.get("injected_requests") == row.get("completed_requests") for row in runs)
        and evidence.get("table_entries", {}).get("attention", 0) > 0
        and evidence.get("table_entries", {}).get("expert_gemv", 0) > 0
    )


def build_case(case: str, *, force: bool = False) -> dict[str, Any]:
    if case not in CASES:
        raise ValueError(f"unknown case {case!r}; choose from {', '.join(CASES)}")
    experiment_path = (PROJECT_ROOT / CASES[case]).resolve()
    cycle_path = (PROJECT_ROOT / "configs/ramulator/sieve_hbm3e_cycle_v0.json").resolve()
    output_path = PROJECT_ROOT / f"ramulator/timing_tables/generated/qwen3_real_kv_{case}_decode_cycle_v0.json"
    evidence_path = output_path.with_name(f"{output_path.stem}.evidence.json")
    configuration = load_configuration(experiment_path)
    trace_set = load_trace_set(
        configuration.experiment.trace_path,
        configuration.model,
        configuration.experiment.layers,
        configuration.experiment.steps,
    )
    if configuration.experiment.trace_manifest_path is None:
        raise ValueError(f"{case}: real trace is missing trace_manifest")
    TraceManifest.load_and_validate(
        configuration.experiment.trace_manifest_path,
        configuration.experiment.trace_path,
        configuration.model,
        trace_set,
    )
    expected = expected_attestation(experiment_path, cycle_path)
    if not force and attestation_matches(evidence_path, output_path, expected):
        print(json.dumps({"case": case, "status": "cache-hit", "table": str(output_path)}), flush=True)
        return {"case": case, "status": "cache-hit"}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = build_timing_table(
        experiment_path,
        cycle_path,
        PROJECT_ROOT / "third_party/ramulator2",
        output_path,
        evidence_path,
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence.update(expected)
    evidence["table_sha256"] = sha256(output_path)
    evidence["interpolation"] = "forbidden"
    evidence["exact_workload"] = True
    evidence["runs_request_counts_match"] = all(
        row.get("injected_requests") == row.get("completed_requests")
        for row in evidence.get("runs", [])
    )
    if not evidence["runs_request_counts_match"]:
        raise RuntimeError(f"Ramulator request counters do not match for {case}")
    evidence["table_entries"] = {
        "attention": len(table["attention"]),
        "expert_gemv": len(table["expert_gemv"]),
    }
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"case": case, "status": "built", "table": str(output_path), "table_sha256": evidence["table_sha256"], **evidence["table_entries"]},
            sort_keys=True,
        ),
        flush=True,
    )
    return {"case": case, "status": "built"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASES), action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cases = sorted(CASES) if args.all or not args.case else args.case
    for case in cases:
        build_case(case, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
