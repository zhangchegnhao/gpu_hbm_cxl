#!/usr/bin/env python3
"""Run the first controlled Batch x Context KV-cache sensitivity sweep.

The sweep keeps the expert IDs and weights from one real pilot layer-batch and
reuses that routing template at each requested batch size.  It is therefore an
analytic sensitivity study, not a new router capture or a Ramulator result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from sieve_replay.config import find_project_root, load_configuration
from sieve_replay.model import estimate_memory_footprint
from sieve_replay.policy import create_policy
from sieve_replay.simulation import EventEngine
from sieve_replay.simulation.layer_graph import build_layer_graph
from sieve_replay.timing import AnalyticTimingModel
from sieve_replay.trace import RouterTraceRecord, TraceBatch, load_trace_set
from sieve_replay.trace_manifest import TraceManifest


DEFAULT_MATRIX = ((8, (512, 2048, 4096, 8192, 16384, 32768)),
                  (16, (4096, 8192, 16384, 32768)),
                  (32, (4096, 8192, 16384)))
PHYSICAL_A800_BYTES = 80_000_000_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _round(value: float) -> float:
    return round(float(value), 9)


def _capacity_state(total: int, physical: int, simulated: int) -> str:
    if total <= physical:
        return "feasible"
    if total <= simulated:
        return "simulated_only"
    return "infeasible"


def _route_template(trace: TraceBatch, batch_size: int, context_length: int) -> TraceBatch:
    """Tile a pilot layer-batch while preserving its expert routing pattern."""
    records: list[RouterTraceRecord] = []
    for index in range(batch_size):
        source = trace.records[index % trace.batch_size]
        records.append(
            RouterTraceRecord(
                step=0,
                layer=trace.layer,
                token_id=f"kv-sweep-{batch_size}-{index}",
                context_length=context_length,
                expert_ids=source.expert_ids,
                expert_weights=source.expert_weights,
            )
        )
    return TraceBatch(step=0, layer=trace.layer, records=tuple(records))


def _event_duration(events: tuple[Any, ...], name: str) -> float:
    return next(event.event.duration_us for event in events if event.event.name == name)


def _row(
    model: Any,
    hardware: Any,
    template: TraceBatch,
    policy_name: str,
    physical_bytes: int,
    simulated_bytes: int,
) -> dict[str, Any]:
    timing = AnalyticTimingModel(model, hardware)
    decision = create_policy(policy_name, timing).place(template)
    events = EventEngine().run(build_layer_graph(template, decision, timing))
    memory = estimate_memory_footprint(model, template)
    # Repeat this controlled routing template across all serial Decode layers.
    # Memory already includes all model layers; use the same scope for latency.
    layers = model.num_hidden_layers
    total_latency = layers * max((event.end_us for event in events), default=0.0)
    attention_latency = layers * _event_duration(events, "attention")
    gpu_path = layers * sum(
        _event_duration(events, name)
        for name in ("gpu_weight_load", "gpu_expert_compute")
    )
    pim_path = layers * sum(
        _event_duration(events, name)
        for name in ("pim_gwrite", "pim_expert_compute", "pim_read")
    )
    expert_latency = max(gpu_path, pim_path)
    return {
        "batch_size": template.batch_size,
        "context_length": template.context_lengths[0],
        "policy": policy_name,
        "layers": layers,
        "decode_steps": 1,
        "kv_cache_bytes": memory.kv_cache_bytes,
        "attention_latency_us": _round(attention_latency),
        "expert_latency_us": _round(expert_latency),
        "gpu_expert_latency_us": _round(gpu_path),
        "pim_expert_latency_us": _round(pim_path),
        "total_latency_us": _round(total_latency),
        "throughput_request_tokens_per_s": _round(
            template.batch_size * 1e6 / total_latency if total_latency else 0.0
        ),
        "attention_latency_share": _round(attention_latency / total_latency if total_latency else 0.0),
        "peak_memory_bytes": memory.total_bytes,
        "model_weights_bytes": memory.model_weights_bytes,
        "activation_bytes": memory.activation_bytes,
        "physical_a800_memory_bytes": physical_bytes,
        "simulated_capacity_bytes": simulated_bytes,
        "physical_feasible": memory.total_bytes <= physical_bytes,
        "simulated_feasible": memory.total_bytes <= simulated_bytes,
        "capacity_state": _capacity_state(memory.total_bytes, physical_bytes, simulated_bytes),
        "gpu_expert_count": len(decision.gpu_experts),
        "pim_expert_count": len(decision.pim_experts),
        "routing_template_layer": template.layer,
    }


def run_sweep(
    experiment_path: str | Path,
    output_dir: str | Path,
    policy_name: str = "sieve",
    physical_memory_gb: float = 80.0,
) -> dict[str, Any]:
    configuration = load_configuration(experiment_path)
    if policy_name not in {"gpu-only", "noexp", "allexp", "pimoe", "sieve"}:
        raise ValueError("KV sweep supports analytic policies: gpu-only, noexp, allexp, pimoe, sieve")
    trace_set = load_trace_set(
        configuration.experiment.trace_path,
        configuration.model,
        configuration.experiment.layers,
        configuration.experiment.steps,
    )
    manifest = None
    if configuration.experiment.trace_manifest_path is not None:
        manifest = TraceManifest.load_and_validate(
            configuration.experiment.trace_manifest_path,
            configuration.experiment.trace_path, configuration.model, trace_set,
        )
    template = trace_set.batches[0]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    physical_bytes = int(physical_memory_gb * 1e9)
    simulated_bytes = int(configuration.hardware.hbm_pim_capacity_gb * 1e9)
    rows = [
        _row(configuration.model, configuration.hardware, _route_template(template, batch, context),
             policy_name, physical_bytes, simulated_bytes)
        for batch, contexts in DEFAULT_MATRIX
        for context in contexts
    ]

    input_paths = {
        "experiment": Path(experiment_path).resolve(),
        "model": configuration.experiment.model_path,
        "hardware": configuration.experiment.hardware_path,
        "trace": configuration.experiment.trace_path,
    }
    if configuration.experiment.trace_manifest_path is not None:
        input_paths["trace_manifest"] = configuration.experiment.trace_manifest_path
        if manifest is not None and "prompts" in manifest.raw:
            input_paths["prompts"] = (
                configuration.experiment.trace_manifest_path.parent / manifest.raw["prompts"]["file"]
            )
    metadata = {
        "method": "analytic-kv-capacity-sensitivity-v1",
        "result_classification": "解析敏感性分析；不是A800实测，也不是Ramulator模拟",
        "experiment": configuration.experiment.name,
        "policy": policy_name,
        "latency_scope": f"{configuration.model.num_hidden_layers} serial layers, one Decode step; identical routing template repeated across layers",
        "limitations": [
            "Routing is controlled sensitivity input, not a new real trace.",
            "Capacity is a byte estimate; no KV READ contention, paging, spill or eviction.",
            "Infeasible rows retain unconstrained analytic timing, not realizable throughput.",
        ],
        "routing_template": {
            "source_trace_sha256": _sha256(configuration.experiment.trace_path),
            "source_step": template.step,
            "source_layer": template.layer,
            "source_batch_size": template.batch_size,
            "context_assumption": "uniform context per row; expert IDs/weights tiled from the first pilot layer-batch",
        },
        "physical_a800_memory_bytes": physical_bytes,
        "simulated_capacity_bytes": simulated_bytes,
        "matrix": [{"batch_size": batch, "context_lengths": list(contexts)} for batch, contexts in DEFAULT_MATRIX],
        "input_hashes": {name: _sha256(path) for name, path in input_paths.items()},
        "config_snapshot": configuration.experiment.raw,
        "rows": len(rows),
    }
    (output / "sweep.json").write_text(
        json.dumps({"metadata": metadata, "rows": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output / "sweep.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _write_baseline(output, configuration, policy_name, physical_bytes)
    _plot(rows, output)
    return {"metadata": metadata, "rows": rows}


def _write_baseline(output: Path, configuration: Any, policy_name: str, physical_bytes: int) -> None:
    """Copy the previously completed real-pilot Ramulator baseline when present."""
    root = find_project_root(configuration.experiment.model_path)
    source = root / "results" / "full_decode_real_pilot_cycle_v0" / policy_name / "summary.json"
    destination = output / "baseline_summary.json"
    if source.is_file():
        raw = source.read_text(encoding="utf-8")
        destination.write_text(raw, encoding="utf-8")
        summary = json.loads(raw)
        event = summary.get("event_duration_us_by_category", {})
        critical = summary["critical_path_duration_us_by_category"]
        memory = summary.get("memory", {})
        breakdown = {
            "result_classification": summary.get("result_classification"),
            "timing_backend": summary.get("timing_backend"),
            "policy": summary.get("policy"),
            "layers": summary["layers"],
            "steps": summary["steps"],
            "source_summary_sha256": _sha256(source),
            "input_hashes": summary["input_hashes"],
            "total_latency_us": summary.get("total_latency_us"),
            "throughput_request_tokens_per_s": summary.get("throughput_request_tokens_per_s"),
            "attention_latency_us": event.get("attention", 0.0),
            "expert_latency_us": sum(critical.get(key, 0.0) for key in (
                "gpu_weight_load", "gpu_expert", "pim_write", "pim_expert", "pim_read"
            )),
            "event_duration_us_by_category": event,
            "critical_path_duration_us_by_category": critical,
            "attention_latency_share": event.get("attention", 0.0) / summary["total_latency_us"],
            "kv_cache_bytes": memory.get("peak_kv_cache_bytes"),
            "peak_memory_bytes": memory.get("peak_total_bytes"),
            "simulated_capacity_bytes": memory.get("capacity_bytes"),
            "physical_a800_memory_bytes": physical_bytes,
            "physical_feasible": memory.get("peak_total_bytes", 0) <= physical_bytes,
            "simulated_feasible": memory.get("feasible"),
            "note": "Existing real-pilot cycle-v0 replay; A800 physical feasibility is a byte comparison, not an A800 timing measurement.",
        }
        (output / "baseline_breakdown.json").write_text(
            json.dumps(breakdown, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return
    destination.write_text(
        json.dumps({
            "result_classification": "baseline unavailable in checkout",
            "policy": policy_name,
            "note": "Run the existing full_decode_real_pilot_cycle_v0 replay to produce the Ramulator baseline.",
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _plot(rows: list[dict[str, Any]], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:
        raise RuntimeError("Install requirements/kv_analysis.txt to generate scientific plots") from error
    def save(name: str, title: str) -> None:
        plt.title(title, fontsize=10)
        plt.tight_layout()
        plt.savefig(output / f"{name}.png", dpi=160)
        plt.savefig(output / f"{name}.svg")
        plt.close()
    batches = sorted({int(row["batch_size"]) for row in rows})
    contexts = sorted({int(row["context_length"]) for row in rows})
    for batch in batches:
        subset = sorted((row for row in rows if row["batch_size"] == batch), key=lambda row: row["context_length"])
        x = [row["context_length"] for row in subset]
        plt.plot(x, [row["kv_cache_bytes"] / 1e9 for row in subset], marker="o", label=f"B={batch}")
    plt.xlabel("Context length"); plt.ylabel("KV cache (GB)"); plt.legend(); plt.tight_layout()
    save("context_kv_cache", "Analytic KV capacity estimate | Full model")

    matrix = np.full((len(batches), len(contexts)), np.nan)
    for row in rows:
        matrix[batches.index(int(row["batch_size"])), contexts.index(int(row["context_length"]))] = row["peak_memory_bytes"] / 1e9
    plt.imshow(matrix, aspect="auto", origin="lower")
    plt.xticks(range(len(contexts)), [str(value) for value in contexts]); plt.yticks(range(len(batches)), [str(value) for value in batches])
    plt.xlabel("Context length"); plt.ylabel("Batch size"); plt.colorbar(label="Peak memory (GB)"); plt.tight_layout()
    save("batch_context_memory_heatmap", "Analytic peak-memory estimate | No spill model")

    for field, filename, ylabel in (("attention_latency_us", "context_attention_latency.png", "Attention latency (us)"),
                                    ("attention_latency_share", "context_attention_share.png", "Attention latency share")):
        for batch in batches:
            subset = sorted((row for row in rows if row["batch_size"] == batch), key=lambda row: row["context_length"])
            plt.plot([row["context_length"] for row in subset], [row[field] for row in subset], marker="o", label=f"B={batch}")
        plt.xlabel("Context length"); plt.ylabel(ylabel); plt.legend(); plt.tight_layout()
        save(Path(filename).stem, "Analytic sensitivity | 48 layers, 1 Decode step")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy", default="sieve")
    parser.add_argument("--physical-memory-gb", type=float, default=80.0)
    args = parser.parse_args(argv)
    result = run_sweep(args.experiment, args.output, args.policy, args.physical_memory_gb)
    print(json.dumps({"output": str(Path(args.output).resolve()), "rows": result["metadata"]["rows"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
