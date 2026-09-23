#!/usr/bin/env python3
"""Run Stage-10 unified CXL-PIM pipeline and partial-state precision experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_kv_cxl_decode_stage5 as stage5
import run_kv_cxl_pim_sensitivity_stage9 as stage9
from sieve_replay.model.partial_attention import run_partial_attention_sweep
from sieve_replay.ramulator.cxl_pim_pipeline import (
    CXLPIMPipelineResult,
    cxl_pim_pipeline_context,
    run_cxl_pim_pipeline,
)
from sieve_replay.ramulator.mixed_workload import SieveCycleV1Config


STAGE9 = ROOT / "results/kv_cxl_pim_sensitivity_stage9_v1/sensitivity.json"
STAGE9_VALIDATION = ROOT / "results/kv_cxl_pim_sensitivity_stage9_v1/validation.json"
DEFAULT_OUTPUT = ROOT / "results/kv_cxl_pim_pipeline_stage10_v1"
DESIGN_IDS = (
    "c2_mac49152_p4",
    "c4_mac24576_p4",
    "c8_mac24576_p4",
)
CONTEXT_LENGTHS = (4_096, 16_384, 32_768)
TOTAL_PSEUDO_CHANNELS = (4, 8, 16)
LOGIT_STDS = (1.0, 4.0)
SEEDS = tuple(range(1, 9))
LAYERS = 48
BATCH_SIZE = 8


def _key(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _manifest_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _load_stage9() -> dict[str, Any]:
    validation = json.loads(STAGE9_VALIDATION.read_text(encoding="utf-8"))
    if validation.get("status") != "passed":
        raise ValueError("Stage 10 requires a passed Stage-9 validation")
    raw = json.loads(STAGE9.read_text(encoding="utf-8"))
    metadata = raw["metadata"]
    if metadata["interpolation"] != 0 or metadata["design_points"] != 18:
        raise ValueError("Stage-9 source is not the exact sensitivity result")
    available = {row["design_id"] for row in raw["design_summary"]}
    if not set(DESIGN_IDS).issubset(available):
        raise ValueError("Stage-9 source is missing a Stage-10 boundary design")
    return raw


def _input_hashes() -> dict[str, str]:
    return {
        "stage9": stage5._sha256(STAGE9),
        "stage9_validation": stage5._sha256(STAGE9_VALIDATION),
        "stage9_runner": stage5._sha256(
            ROOT / "scripts/run_kv_cxl_pim_sensitivity_stage9.py"
        ),
        "model": stage5._sha256(ROOT / "configs/models/qwen3_30b_a3b.json"),
        "cycle": stage5._sha256(stage5.CYCLE),
        "precision_model": stage5._sha256(
            ROOT / "src/sieve_replay/model/partial_attention.py"
        ),
    }


def _cache_input(design: dict[str, Any], execution: dict[str, str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": "kv-cxl-pim-pipeline-stage10-v1",
        "case": "b8_c32k",
        "design_id": design["design_id"],
        "topology": asdict(design["topology"]),
        "mac_interval_ps": design["mac_interval_ps"],
        "shape": asdict(design["shape"]),
        "link_profile": {
            "bandwidth_gb_s": stage5.PROFILE["bandwidth_gb_s"],
            "latency_us": stage5.PROFILE["latency_us"],
            "channel_count": stage5.PROFILE["channel_count"],
        },
        "pipeline_context": cxl_pim_pipeline_context(ROOT, stage5.CYCLE),
        "execution": execution,
        "input_hashes": _input_hashes(),
    }


def _run_or_load(
    output: Path,
    design: dict[str, Any],
    execution: dict[str, str],
    runner: Callable[..., CXLPIMPipelineResult],
) -> tuple[str, Path, CXLPIMPipelineResult, bool, dict[str, Any]]:
    cache_input = _cache_input(design, execution)
    key = _key(cache_input)
    cache_dir = output / "exact_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    if path.is_file():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("cache_input") != cache_input:
            raise ValueError(f"Stage-10 cache provenance mismatch: {path}")
        return key, path, CXLPIMPipelineResult(**cached["result"]), False, cache_input

    shape = design["shape"]
    result = runner(
        stage5.RAMULATOR,
        design["cycle"],
        design["topology"],
        query_link_transactions=shape.query_link_transactions,
        pim_gwrite_waves=shape.pim_gwrite_waves,
        pim_mac_waves=shape.pim_mac_waves,
        pim_read_waves=shape.pim_read_waves,
        result_link_transactions=shape.result_link_transactions,
        cxl_bandwidth_bytes_per_second=float(stage5.PROFILE["bandwidth_gb_s"])
        * 1e9,
        cxl_latency_us=float(stage5.PROFILE["latency_us"]),
        link_channels=int(stage5.PROFILE["channel_count"]),
    )
    path.write_text(
        json.dumps(
            {"cache_input": cache_input, "result": asdict(result)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return key, path, result, True, cache_input


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _precision_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for pseudo_channels in TOTAL_PSEUDO_CHANNELS:
        for scalar_format in ("fp32", "bf16", "fp16"):
            group = [
                row
                for row in rows
                if row["total_pseudo_channels"] == pseudo_channels
                and row["scalar_format"] == scalar_format
            ]
            summary.append(
                {
                    "total_pseudo_channels": pseudo_channels,
                    "scalar_format": scalar_format,
                    "trials": len(group),
                    "max_abs_error": max(row["max_abs_error"] for row in group),
                    "mean_abs_error": statistics.fmean(
                        row["mean_abs_error"] for row in group
                    ),
                    "max_relative_l2_error": max(
                        row["relative_l2_error"] for row in group
                    ),
                    "min_cosine_similarity": min(
                        row["cosine_similarity"] for row in group
                    ),
                    "max_vs_fp32_partial_abs_error": max(
                        row["vs_fp32_partial_max_abs_error"] for row in group
                    ),
                    "max_vs_fp32_partial_relative_l2_error": max(
                        row["vs_fp32_partial_relative_l2_error"] for row in group
                    ),
                }
            )
    return summary


def _write_report(
    output: Path,
    pipeline_rows: list[dict[str, Any]],
    precision_summary: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    lines = [
        "# KV CXL-PIM pipeline Stage 10",
        "",
        "Stage 10 runs three Stage-9 boundary designs through one dependency-gated Ramulator simulation per design. Link controllers and CXL-PIM media controllers remain separate resources in one event timeline.",
        "",
        "| design | isolated us/layer | unified us/layer | delta | serialized Decode ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in pipeline_rows:
        lines.append(
            f"| {row['design_id']} | {row['isolated_branch_us_per_layer']:.3f} | "
            f"{row['unified_branch_us_per_layer']:.3f} | "
            f"{row['unified_minus_isolated_us_per_layer']:.3f} | "
            f"{row['serialized_decode_ms']:.3f} |"
        )
    lines += [
        "",
        f"Exact unified runs: `{metadata['exact_runs']}`; interpolation: `0`.",
        "",
        "The precision sweep uses deterministic synthetic logits and values, not captured A800 activations. It reports FP32, BF16 and FP16 partial-state merge errors without imposing an unsupported accuracy threshold.",
        "",
        "| PCH | format | trials | max abs error | max relative L2 | min cosine |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in precision_summary:
        lines.append(
            f"| {row['total_pseudo_channels']} | {row['scalar_format']} | "
            f"{row['trials']} | {row['max_abs_error']:.6e} | "
            f"{row['max_relative_l2_error']:.6e} | "
            f"{row['min_cosine_similarity']:.9f} |"
        )
    lines += [
        "",
        "This remains a single-GPU analytic/Ramulator study. It does not implement paging, physical CXL protocol, energy, multi-GPU arbitration or Shared CXL-PIM.",
        "",
    ]
    (output / "stage10_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_stage10(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    pipeline_runner: Callable[..., CXLPIMPipelineResult] = run_cxl_pim_pipeline,
    precision_runner: Callable[..., list[dict[str, Any]]] = run_partial_attention_sweep,
) -> dict[str, Any]:
    output = Path(output_dir)
    if output.exists():
        existing = {item.name for item in output.iterdir()}
        if existing and not existing.issubset({"exact_cache"}):
            raise ValueError(f"Stage-10 output already exists and is not resumable: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source = _load_stage9()
    stage8 = stage9._load_stage8()
    configuration, _ = stage5._source_configuration()
    cycle = SieveCycleV1Config.load(stage5.CYCLE)
    designs = stage9._designs(
        configuration.model,
        cycle,
        int(stage8["derived"]["memory_only_cxl_link_bytes_per_layer"]),
    )
    selected = {design["design_id"]: design for design in designs if design["design_id"] in DESIGN_IDS}
    if set(selected) != set(DESIGN_IDS):
        raise ValueError("failed to reconstruct every Stage-10 boundary design")

    summaries = {row["design_id"]: row for row in source["design_summary"]}
    local_attention_ms = float(stage8["derived"]["local_gpu_attention_ms"])
    resident = next(row for row in stage8["rows"] if row["scenario"] == "resident-local-counterfactual")
    non_attention_ms = float(resident["estimated_decode_ms"]) - float(resident["estimated_attention_ms"])
    memory_only_decode_ms = float(source["metadata"]["baseline"]["memory_only_decode_ms"])
    resident_decode_ms = float(source["metadata"]["baseline"]["resident_decode_ms"])
    execution = stage5._execution_provenance()
    cache_records: list[dict[str, Any]] = []
    pipeline_rows: list[dict[str, Any]] = []
    new_runs = 0
    for design_id in DESIGN_IDS:
        design = selected[design_id]
        key, path, result, executed, cache_input = _run_or_load(
            output, design, execution, pipeline_runner
        )
        new_runs += int(executed)
        cache_records.append(
            {
                "design_id": design_id,
                "key": key,
                "cache_path": _manifest_path(path),
                "sha256": stage5._sha256(path),
                "executed_this_invocation": executed,
                "cache_input": cache_input,
            }
        )
        isolated_us = float(summaries[design_id]["branch_us_per_layer"])
        unified_us = result.pipeline_completion_us
        unified_branch_ms = unified_us * LAYERS / 1000.0
        ideal_attention_ms = max(local_attention_ms, unified_branch_ms)
        serialized_attention_ms = local_attention_ms + unified_branch_ms
        ideal_decode_ms = ideal_attention_ms + non_attention_ms
        serialized_decode_ms = serialized_attention_ms + non_attention_ms
        pipeline_rows.append(
            {
                "design_id": design_id,
                "channels": design["channels"],
                "total_pseudo_channels": design["total_pseudo_channels"],
                "mac_interval_ps": design["mac_interval_ps"],
                "partial_scalar_bytes": design["partial_scalar_bytes"],
                "isolated_branch_us_per_layer": isolated_us,
                "unified_branch_us_per_layer": unified_us,
                "unified_minus_isolated_us_per_layer": unified_us - isolated_us,
                "unified_over_isolated": unified_us / isolated_us,
                "query_link_us": result.query_link_us,
                "pim_gwrite_us": result.pim_gwrite_us,
                "pim_mac_us": result.pim_mac_us,
                "pim_read_us": result.pim_read_us,
                "result_link_us": result.result_link_us,
                "ideal_overlap_decode_ms": ideal_decode_ms,
                "ideal_overlap_throughput_tokens_per_s": BATCH_SIZE * 1000.0 / ideal_decode_ms,
                "serialized_decode_ms": serialized_decode_ms,
                "serialized_throughput_tokens_per_s": BATCH_SIZE * 1000.0 / serialized_decode_ms,
                "serialized_beats_memory_only": serialized_decode_ms < memory_only_decode_ms,
                "serialized_beats_resident": serialized_decode_ms < resident_decode_ms,
            }
        )

    precision_rows = precision_runner(
        context_lengths=CONTEXT_LENGTHS,
        total_pseudo_channels=TOTAL_PSEUDO_CHANNELS,
        logit_stds=LOGIT_STDS,
        seeds=SEEDS,
        head_dim=configuration.model.head_dim,
    )
    precision_summary = _precision_summary(precision_rows)
    _write_csv(output / "pipeline_comparison.csv", pipeline_rows)
    _write_csv(output / "partial_precision_trials.csv", precision_rows)
    _write_csv(output / "partial_precision_summary.csv", precision_summary)
    precision_path = output / "partial_precision.json"
    precision_path.write_text(
        json.dumps(
            {
                "method": {
                    "classification": "deterministic synthetic numerical sensitivity; no A800 activation capture",
                    "context_lengths": list(CONTEXT_LENGTHS),
                    "total_pseudo_channels": list(TOTAL_PSEUDO_CHANNELS),
                    "logit_stds": list(LOGIT_STDS),
                    "seeds": list(SEEDS),
                    "head_dim": configuration.model.head_dim,
                    "value_model": "four deterministic basis functions projected to the model head dimension",
                    "partitioning": "context positions striped across pseudo-channels",
                    "partial_formats": ["fp32", "bf16", "fp16"],
                },
                "trials": precision_rows,
                "summary": precision_summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    metadata = {
        "schema_version": 1,
        "stage": "kv-cxl-pim-pipeline-stage10-v1",
        "case": "b8_c32k",
        "design_ids": list(DESIGN_IDS),
        "exact_runs": len(pipeline_rows),
        "new_runs_executed": new_runs,
        "interpolation": 0,
        "pipeline_model": "single dependency-gated event timeline with separate CXL-link and CXL-PIM media controllers",
        "precision_trials": len(precision_rows),
        "execution": execution,
        "input_hashes": _input_hashes(),
        "cache_records": cache_records,
        "baselines": {
            "memory_only_decode_ms": memory_only_decode_ms,
            "resident_decode_ms": resident_decode_ms,
            "local_gpu_attention_ms": local_attention_ms,
            "non_attention_ms": non_attention_ms,
        },
        "limitations": [
            "The link and PIM media use separate controllers; the experiment validates phase dependencies and persistent controller state, not a single shared physical queue.",
            "The CXL link remains a fixed 64 GB/s FIFO READ proxy without protocol direction effects.",
            "The local B8/C32k GPU Attention branch remains an A800 trend extrapolation.",
            "Partial precision uses deterministic synthetic logits and values, not captured model activations.",
            "Paging, eviction, recovery, energy, multi-GPU arbitration and Shared CXL-PIM are absent.",
        ],
    }
    comparison_path = output / "pipeline_comparison.json"
    comparison_path.write_text(
        json.dumps(
            {"metadata": metadata, "rows": pipeline_rows},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_report(output, pipeline_rows, precision_summary, metadata)
    manifest = {
        "schema_version": 1,
        "stage": metadata["stage"],
        "exact_runs": len(pipeline_rows),
        "new_runs_executed": new_runs,
        "interpolation": 0,
        "execution": execution,
        "input_hashes": metadata["input_hashes"],
        "cache_records": cache_records,
        "output_hashes": {
            name: stage5._sha256(output / name)
            for name in (
                "partial_precision.json",
                "partial_precision_summary.csv",
                "partial_precision_trials.csv",
                "pipeline_comparison.csv",
                "pipeline_comparison.json",
                "stage10_report.md",
            )
        },
    }
    (output / "stage_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "metadata": metadata,
        "rows": pipeline_rows,
        "precision_summary": precision_summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    result = run_stage10(args.output)
    print(json.dumps(result["metadata"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
