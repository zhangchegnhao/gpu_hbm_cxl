#!/usr/bin/env python3
"""Run the stage-2 KV capacity extension sensitivity experiment.

This stage models CXL as a memory-only spill budget.  It does not model a CXL
link, CXL-PIM commands, paging traffic, eviction, recovery, or any timing.
The A800 80 GB and simulated 96 GB local domains remain independent; bytes
that do not fit locally may be admitted to the explicitly supplied CXL budget.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from sieve_replay.config import load_configuration
from sieve_replay.model import estimate_memory_footprint
from sieve_replay.model.capacity import classify_capacity
from sieve_replay.trace import RouterTraceRecord, TraceBatch, load_trace_set
from sieve_replay.trace_manifest import TraceManifest


PRESSURE_MATRIX: tuple[tuple[int, int], ...] = (
    (8, 32768),
    (16, 16384),
    (16, 32768),
    (32, 16384),
)
CXL_BUDGET_GB: tuple[int, ...] = (0, 16, 32, 64)
A800_BYTES = 80_000_000_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _template_batch(source: TraceBatch, batch_size: int, context_length: int) -> TraceBatch:
    records = tuple(
        RouterTraceRecord(
            step=0,
            layer=source.layer,
            token_id=f"kv-cxl-capacity-{batch_size}-{context_length}-{index}",
            context_length=context_length,
            expert_ids=source.records[index % source.batch_size].expert_ids,
            expert_weights=source.records[index % source.batch_size].expert_weights,
        )
        for index in range(batch_size)
    )
    return TraceBatch(step=0, layer=source.layer, records=records)


def _input_paths(configuration: Any, experiment_path: Path, manifest: TraceManifest) -> dict[str, Path]:
    manifest_raw = manifest.raw
    return {
        "experiment": experiment_path.resolve(),
        "trace": configuration.experiment.trace_path,
        "trace_manifest": configuration.experiment.trace_manifest_path,
        "prompts": configuration.experiment.trace_manifest_path.parent / manifest_raw["prompts"]["file"],
        "model": configuration.experiment.model_path,
        "hardware": configuration.experiment.hardware_path,
    }


def _admission_row(
    configuration: Any,
    template: TraceBatch,
    batch_size: int,
    context_length: int,
    domain: str,
    local_capacity_bytes: int,
    cxl_capacity_bytes: int,
) -> dict[str, Any]:
    memory = estimate_memory_footprint(configuration.model, template)
    admission = classify_capacity(
        memory.total_bytes,
        local_capacity_bytes,
        cxl_capacity_bytes,
    )
    row: dict[str, Any] = {
        "batch_size": batch_size,
        "context_length": context_length,
        "domain": domain,
        "local_capacity_bytes": local_capacity_bytes,
        "cxl_capacity_bytes": cxl_capacity_bytes,
        "kv_cache_bytes": memory.kv_cache_bytes,
        "model_weights_bytes": memory.model_weights_bytes,
        "activation_bytes": memory.activation_bytes,
        "peak_memory_bytes": memory.total_bytes,
        "spill_destination": "none" if cxl_capacity_bytes == 0 else "hypothetical-cxl-memory-only",
        "timing_modelled": False,
    }
    row.update(admission)
    return row


def _report(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> str:
    lines = [
        "# KV CXL capacity stage 2",
        "",
        "本阶段只做容量准入敏感性分析：将 CXL 表示为显式的额外 memory-only spill budget。",
        "A800 80 GB 和模拟 96 GB 是两个独立的本地容量域，CXL 字节不会被并入 Local HBM，",
        "也不会把两个本地容量相加。结果不是 A800 实测、不是 Ramulator 时延，也不是 CXL-PIM。",
        "",
        "## Pressure points",
        "",
        "| domain | B | C | peak memory (GB) | CXL budget (GB) | state | feasible | spill (GB) | unallocated (GB) |",
        "|---|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {domain} | {batch_size} | {context_length} | {peak:.6f} | {budget:.0f} | {state} | {feasible} | {spill:.6f} | {unallocated:.6f} |".format(
                domain=row["domain"],
                batch_size=row["batch_size"],
                context_length=row["context_length"],
                peak=row["peak_memory_bytes"] / 1e9,
                budget=row["cxl_capacity_bytes"] / 1e9,
                state=row["state"],
                feasible=str(row["feasible"]).lower(),
                spill=row["spill_bytes"] / 1e9,
                unallocated=row["unallocated_bytes"] / 1e9,
            )
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "`resident` 表示估算峰值全部放入本地容量；`spill` 表示超出本地容量的字节数",
        "可以放入显式 CXL budget；`oom` 表示本地容量和该 budget 仍不足。`spill` 只是",
        "容量准入状态，不代表已经实现传输、分页、驱逐或恢复。",
        "",
        "本阶段使用 manifest 绑定的真实 Router 模板，仅为了保持模型和路由输入身份；",
        "没有复制 Router Trace，也没有重新捕获 A800 timing。所有时延相关字段均不适用。",
        "",
        "## Provenance",
        "",
        f"- method: `{metadata['method']}`",
        f"- input manifest validated: `{metadata['manifest_validated']}`",
        f"- pressure points: `{len(PRESSURE_MATRIX)}`",
        f"- CXL budgets (GB): `{metadata['cxl_budget_gb']}`",
        "- limitations: no CXL link timing, CXL-PIM, paging, eviction, recovery, or A800 memory observation",
        "",
    ]
    return "\n".join(lines)


def run_cxl_capacity_stage2(
    experiment_path: str | Path,
    output_dir: str | Path,
    cxl_budget_gb: Iterable[int] = CXL_BUDGET_GB,
) -> dict[str, Any]:
    experiment = Path(experiment_path)
    configuration = load_configuration(experiment)
    trace_set = load_trace_set(
        configuration.experiment.trace_path,
        configuration.model,
        configuration.experiment.layers,
        configuration.experiment.steps,
    )
    manifest_path = configuration.experiment.trace_manifest_path
    if manifest_path is None:
        raise ValueError("CXL capacity study requires a manifest-bound routing template")
    manifest = TraceManifest.load_and_validate(
        manifest_path,
        configuration.experiment.trace_path,
        configuration.model,
        trace_set,
    )
    if manifest.raw["schema_version"] != 2:
        raise ValueError("CXL capacity study requires schema-v2 prompt-bound provenance")

    budgets = tuple(int(value) for value in cxl_budget_gb)
    if not budgets or any(value < 0 for value in budgets):
        raise ValueError("CXL budgets must contain at least one nonnegative integer GB value")
    if len(set(budgets)) != len(budgets):
        raise ValueError("CXL budgets must be unique")

    simulated_bytes = int(configuration.hardware.hbm_pim_capacity_gb * 1e9)
    domains = (("a800", A800_BYTES), ("simulated", simulated_bytes))
    source = trace_set.batches[0]
    rows: list[dict[str, Any]] = []
    for batch_size, context_length in PRESSURE_MATRIX:
        template = _template_batch(source, batch_size, context_length)
        for domain, local_capacity in domains:
            for budget_gb in budgets:
                rows.append(
                    _admission_row(
                        configuration,
                        template,
                        batch_size,
                        context_length,
                        domain,
                        local_capacity,
                        budget_gb * 1_000_000_000,
                    )
                )

    input_paths = _input_paths(configuration, experiment, manifest)
    metadata: dict[str, Any] = {
        "method": "analytic-kv-cxl-capacity-admission-v1",
        "result_classification": "解析容量准入敏感性分析；CXL为memory-only假设，不是实机或Ramulator结果",
        "scope": "1 GPU + 8 Local HBM-PIM stacks; hypothetical CXL memory-only capacity extension",
        "manifest_validated": True,
        "pressure_matrix": [
            {"batch_size": batch, "context_length": context} for batch, context in PRESSURE_MATRIX
        ],
        "cxl_budget_gb": list(budgets),
        "capacity_domains": {
            "a800": {
                "local_capacity_bytes": A800_BYTES,
                "source": "physical A800 80 GB decimal capacity reference",
            },
            "simulated": {
                "local_capacity_bytes": simulated_bytes,
                "source": "hardware configuration hbm_pim_capacity_gb; decimal GB",
            },
        },
        "semantics": {
            "resident": "full estimated allocation fits the local domain",
            "spill": "overflow fits the explicitly supplied CXL memory-only budget",
            "oom": "local plus explicit CXL budget is insufficient",
            "cxl": "hypothetical additional capacity, not pooled HBM and not a timing resource",
        },
        "limitations": [
            "No CXL link, CXL-PIM command, paging, spill transfer, eviction or recovery timing",
            "No A800 execution, memory sampling or observed OOM",
            "No KV READ request-level competition or Expert timing is included",
            "Peak memory is a model estimate and excludes allocator reserve, fragmentation, workspace and prefill activations",
            "Routing is a manifest-bound controlled template used only for byte estimation",
        ],
        "input_hashes": {name: _sha256(path) for name, path in input_paths.items()},
        "config_snapshot": {
            "experiment": configuration.experiment.raw,
            "model": configuration.model_raw,
            "hardware": configuration.hardware_raw,
        },
        "rows": len(rows),
    }
    payload = {"metadata": metadata, "rows": rows}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "cxl_capacity_states.json"
    csv_path = output / "cxl_capacity_states.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    metadata["output_hashes"] = {"cxl_capacity_states.csv": _sha256(csv_path)}
    report_path = output / "stage2_report.md"
    report_path.write_text(_report(rows, metadata), encoding="utf-8")
    metadata["output_hashes"]["stage2_report.md"] = _sha256(report_path)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "method": metadata["method"],
        "input_hashes": metadata["input_hashes"],
        "output_hashes": {
            "cxl_capacity_states.json": _sha256(json_path),
            "cxl_capacity_states.csv": _sha256(csv_path),
            "stage2_report.md": _sha256(report_path),
        },
        "rows": metadata["rows"],
        "manifest_validated": metadata["manifest_validated"],
    }
    (output / "stage_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--cxl-budget-gb",
        type=int,
        nargs="+",
        default=list(CXL_BUDGET_GB),
        help="explicit hypothetical CXL capacity budgets in decimal GB",
    )
    args = parser.parse_args(argv)
    result = run_cxl_capacity_stage2(args.experiment, args.output, args.cxl_budget_gb)
    print(json.dumps({"output": str(Path(args.output).resolve()), "rows": result["metadata"]["rows"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
