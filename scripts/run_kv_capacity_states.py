#!/usr/bin/env python3
"""Evaluate KV capacity admission independently for A800 and simulated hardware.

This is an analytic byte-allocation model, with no data movement or timing.  The
A800's 80 GB and the simulation's 96 GB belong to separate device domains.  Both
formal domains have zero spill capacity: an overflow is oom/infeasible.  The
reusable classifier supports an explicit spill budget for hypothetical resource
admission, without implementing paging, eviction or transfer performance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from sieve_replay.config import load_configuration
from sieve_replay.model import estimate_memory_footprint
from sieve_replay.model.capacity import classify_capacity
from sieve_replay.trace import RouterTraceRecord, TraceBatch, load_trace_set
from sieve_replay.trace_manifest import TraceManifest


CAPACITY_MATRIX: tuple[tuple[int, int], ...] = (
    (8, 32768), (16, 16384), (16, 32768), (32, 16384),
)
A800_BYTES = 80_000_000_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row(configuration: Any, source: TraceBatch, batch_size: int,
         context_length: int, simulated_bytes: int) -> dict[str, Any]:
    records = tuple(
        RouterTraceRecord(
            step=0, layer=source.layer,
            token_id=f"kv-capacity-state-{batch_size}-{index}",
            context_length=context_length,
            expert_ids=source.records[index % source.batch_size].expert_ids,
            expert_weights=source.records[index % source.batch_size].expert_weights,
        )
        for index in range(batch_size)
    )
    template = TraceBatch(step=0, layer=source.layer, records=records)
    memory = estimate_memory_footprint(configuration.model, template)
    row: dict[str, Any] = {
        "batch_size": batch_size,
        "context_length": context_length,
        "kv_cache_bytes": memory.kv_cache_bytes,
        "model_weights_bytes": memory.model_weights_bytes,
        "activation_bytes": memory.activation_bytes,
        "peak_memory_bytes": memory.total_bytes,
    }
    for domain, capacity in (("a800", A800_BYTES), ("simulated", simulated_bytes)):
        admission = classify_capacity(memory.total_bytes, capacity)
        row.update({f"{domain}_{key}": value for key, value in admission.items()})
    row["timing_modelled"] = False
    return row


def run_capacity_states(experiment_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    configuration = load_configuration(experiment_path)
    simulated_bytes = int(configuration.hardware.hbm_pim_capacity_gb * 1e9)
    trace_set = load_trace_set(
        configuration.experiment.trace_path, configuration.model,
        configuration.experiment.layers, configuration.experiment.steps,
    )
    manifest_path = configuration.experiment.trace_manifest_path
    if manifest_path is None:
        raise ValueError("capacity study requires a manifest-bound routing template")
    manifest = TraceManifest.load_and_validate(
        manifest_path, configuration.experiment.trace_path, configuration.model, trace_set,
    )
    if manifest.raw["schema_version"] != 2:
        raise ValueError("capacity study requires schema-v2 prompt-bound provenance")
    rows = [_row(configuration, trace_set.batches[0], batch, context, simulated_bytes)
            for batch, context in CAPACITY_MATRIX]
    input_paths = {
        "experiment": Path(experiment_path).resolve(),
        "trace": configuration.experiment.trace_path,
        "trace_manifest": manifest_path,
        "prompts": manifest_path.parent / manifest.raw["prompts"]["file"],
        "model": configuration.experiment.model_path,
        "hardware": configuration.experiment.hardware_path,
    }
    metadata = {
        "method": "analytic-kv-capacity-state-v1",
        "result_classification": "解析容量准入分析；不是A800运行结果，也不是Ramulator模拟",
        "scope": "1 GPU + 8 Local HBM-PIM stacks; full-model byte estimate",
        "capacity_domains": {
            "a800": {"resident_capacity_bytes": A800_BYTES, "spill_capacity_bytes": 0,
                     "source": "physical A800 80 GB decimal capacity reference; independent of simulated hardware"},
            "simulated": {"resident_capacity_bytes": simulated_bytes, "spill_capacity_bytes": 0,
                          "source": "hardware configuration hbm_pim_capacity_gb; decimal GB"},
        },
        "matrix": [{"batch_size": b, "context_length": c} for b, c in CAPACITY_MATRIX],
        "state_semantics": {
            "resident": "bytes fit the selected device's resident capacity",
            "spill": "bytes fit an explicitly supplied additional spill budget; none is configured in these formal results",
            "oom": "bytes exceed the selected device's resident plus explicit spill capacity; admission is infeasible",
        },
        "limitations": [
            "The 80 GB A800 and 96 GB simulated device are independent capacity domains; their capacities are never pooled.",
            "This implements resource admission and byte allocation only; no paging, spill transfers, eviction or recovery.",
            "Both formal domains use zero spill capacity; over-capacity points remain infeasible.",
            "No KV READ request-level competition or transfer timing is included.",
            "Memory excludes allocator reserve, fragmentation, implementation workspace and prefill activations; resident is an estimate, not measured hardware feasibility.",
            "Routing is a controlled template from the manifest-validated pilot; no new real trace is created.",
        ],
        "input_hashes": {name: _sha256(path) for name, path in input_paths.items()},
        "config_snapshot": {
            "experiment": configuration.experiment.raw,
            "model": configuration.model_raw,
            "hardware": configuration.hardware_raw,
        },
        "routing_template": {"step": trace_set.batches[0].step,
                             "layer": trace_set.batches[0].layer,
                             "source_batch_size": trace_set.batch_size,
                             "context_assumption": "uniform context; source expert routes tiled for byte estimation only"},
        "manifest_validated": True,
        "timing_modelled": False,
        "rows": len(rows),
    }
    payload = {"metadata": metadata, "rows": rows}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "capacity_states.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "capacity_states.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = run_capacity_states(args.experiment, args.output)
    print(json.dumps({"output": str(Path(args.output).resolve()), "rows": result["metadata"]["rows"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
