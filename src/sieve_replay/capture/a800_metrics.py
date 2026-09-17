"""Plan-only protocol for independent A800 timing/memory capture.

The repository's existing Qwen3 capture path records Router decisions and has a
synchronizing ``.cpu().tolist()`` hook.  Hardware timing must therefore be a
separate pass with no Router hooks.  This module deliberately does *not* run a
model: it validates the five imported long-context inputs and emits an auditable
plan for a future A800 capture implementation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CASES = (
    ("b8_c4k", 8, 4096),
    ("b8_c8k", 8, 8192),
    ("b8_c16k", 8, 16384),
    ("b16_c4k", 16, 4096),
    ("b16_c8k", 16, 8192),
)
REQUIRED_INPUTS = ("prompts.jsonl", "router.jsonl", "manifest.json")
REQUIRED_OUTPUTS = ("prompts.jsonl", "router.jsonl", "timing.jsonl", "memory.jsonl", "manifest.json")


@dataclass(frozen=True)
class A800CasePlan:
    case: str
    batch_size: int
    context_length: int
    input_dir: str
    output_dir: str
    prompt_sha256: str
    router_sha256: str
    manifest_sha256: str
    prompt_count: int
    router_records: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "case": self.case,
            "batch_size": self.batch_size,
            "context_length": self.context_length,
            "status": "planned",
            "input_dir": self.input_dir,
            "input_artifacts": {
                "prompts": {"file": "prompts.jsonl", "sha256": self.prompt_sha256},
                "router": {"file": "router.jsonl", "sha256": self.router_sha256},
                "manifest": {"file": "manifest.json", "sha256": self.manifest_sha256},
            },
            "output_dir": self.output_dir,
            "output_artifacts": [
                {"file": name, "sha256": "captured-at-run"} for name in REQUIRED_OUTPUTS
            ],
            "prompt_count": self.prompt_count,
            "router_records": self.router_records,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid or missing JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _count_jsonl(path: Path) -> int:
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except FileNotFoundError as exc:
        raise ValueError(f"missing JSONL: {path}") from exc


def validate_case(root: str | Path, case: str, *, output_root: str | Path) -> A800CasePlan:
    """Validate one existing real Router capture without modifying it."""

    root_path = Path(root).resolve()
    match = next((entry for entry in CASES if entry[0] == case), None)
    if match is None:
        raise ValueError(f"unsupported A800 case: {case}")
    _, batch_size, context_length = match
    input_dir = root_path / "traces" / "real" / "qwen3_30b_a3b" / f"kv_{case}"
    paths = {
        "prompts": input_dir / "prompts.jsonl",
        "router": input_dir / "router.jsonl",
        "manifest": input_dir / "manifest.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(f"{case}: missing required input(s): {', '.join(missing)}")
    manifest = _read_json(paths["manifest"])
    workload = manifest.get("workload")
    if not isinstance(workload, dict):
        raise ValueError(f"{case}: manifest workload is missing")
    if workload.get("batch_size") != batch_size or workload.get("prompt_count") != batch_size:
        raise ValueError(f"{case}: manifest batch/prompt count does not match planned batch")
    if workload.get("decode_steps") != 8:
        raise ValueError(f"{case}: expected 8 fixed Decode steps")
    if manifest.get("trace_file") != "router.jsonl":
        raise ValueError(f"{case}: manifest trace_file must be router.jsonl")
    if manifest.get("trace_sha256") != sha256_file(paths["router"]):
        raise ValueError(f"{case}: manifest Router SHA-256 mismatch")
    prompts_meta = manifest.get("prompts")
    if not isinstance(prompts_meta, dict) or prompts_meta.get("file") != "prompts.jsonl":
        raise ValueError(f"{case}: manifest prompt binding is invalid")
    if prompts_meta.get("sha256") != sha256_file(paths["prompts"]):
        raise ValueError(f"{case}: manifest prompt SHA-256 mismatch")
    prompt_count = _count_jsonl(paths["prompts"])
    router_records = _count_jsonl(paths["router"])
    if prompt_count != batch_size or router_records != batch_size * 48 * 8:
        raise ValueError(f"{case}: prompt/router record count is inconsistent")
    output_dir = Path(output_root).resolve() / case
    return A800CasePlan(
        case=case,
        batch_size=batch_size,
        context_length=context_length,
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        prompt_sha256=sha256_file(paths["prompts"]),
        router_sha256=sha256_file(paths["router"]),
        manifest_sha256=sha256_file(paths["manifest"]),
        prompt_count=prompt_count,
        router_records=router_records,
    )


def build_plan(root: str | Path, *, output_root: str | Path) -> dict[str, Any]:
    """Build and return the five-case dry-run plan."""

    cases = [validate_case(root, case, output_root=output_root) for case, _, _ in CASES]
    return {
        "schema_version": 1,
        "capture_kind": "a800-timing-memory-v1",
        "execution_implemented": False,
        "status": "planned",
        "result_classification": "dry-run protocol only; no A800 execution performed",
        "hardware_scope": {
            "device": "NVIDIA A800-SXM4-80GB",
            "single_cuda_device": True,
            "device_map": "explicit-single-device (cuda:0)",
            "cpu_offload": False,
        },
        "model_scope": {
            "name": "Qwen3-30B-A3B",
            "revision": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
            "dtype": "bfloat16",
            "layers": 48,
            "experts": 128,
            "top_k": 8,
        },
        "measurement_protocol": {
            "routing_pass": "fresh Router pass; preserve prompts.jsonl, greedy argmax and fixed 8 Decode steps",
            "timing_pass": "separate model pass with no Router hooks or CPU synchronization",
            "baseline_pass": "no instrumentation; per Decode step CUDA events",
            "attention_pass": "self_attn pre/post CUDA events; includes Python/module launch gaps",
            "memory_pass": "reset_peak_memory_stats per repeat; record allocated and reserved peaks",
            "repeats": 3,
            "warmup": 1,
            "repeat_prefill": "rebuild prefill and KV cache for every measured repeat",
            "artifact_binding": "each case writes independent prompts/router/timing/memory/manifest with SHA-256",
            "token_alignment": "timing pass must assert greedy tokens match fresh Router pass before accepting rows",
        },
        "required_outputs": list(REQUIRED_OUTPUTS),
        "cases": [case.as_dict() for case in cases],
        "limitations": [
            "This command does not load model weights or execute CUDA.",
            "A800 timing applies only to the exact prompt batch, model revision, and fixed Decode scope.",
            "Attention events are module-boundary measurements, not kernel-only timings.",
            "No KV paging, spill, eviction, PIM timing, or Expert READ trace is included.",
            "Existing traces are inputs and are never copied with modified Context values; a future run must capture fresh Router JSONL.",
        ],
    }


def write_plan(root: str | Path, output_path: str | Path) -> Path:
    plan = build_plan(root, output_root=Path(output_path).resolve().parent / "cases")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
