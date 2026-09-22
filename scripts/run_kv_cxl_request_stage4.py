#!/usr/bin/env python3
"""Run Stage-4 CXL memory-only request-level microbenchmarks.

The experiment uses one manifest-validated real router batch as a controlled
expert-load template. KV traffic is generated from the analytic per-layer KV
size and the A800 spill fractions frozen by Stage 2. Results are exact
Ramulator request-level runs; CXL profiles remain explicit assumptions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sieve_replay.config import load_configuration
from sieve_replay.ramulator.cxl_kv_workload import (
    CXLKVWorkloadResult,
    cxl_kv_cache_context,
    run_cxl_kv_workload,
)
from sieve_replay.ramulator.mixed_workload import SieveCycleV1Config
from sieve_replay.ramulator.workload_cache import shape_for_loads
from sieve_replay.report.writer import sha256_file
from sieve_replay.trace import load_trace
from sieve_replay.types import ExpertLoad


CYCLE = ROOT / "configs/ramulator/sieve_hbm3e_cycle_v1.json"
SOURCE_EXPERIMENT = ROOT / "configs/experiments/full_decode_real_kv_b8_c4k.json"
CAPACITY = ROOT / "results/kv_cxl_capacity_stage2_v1/cxl_capacity_states.json"
RAMULATOR = ROOT / "third_party/ramulator2_cxl"

PRESSURE_POINTS = (
    ("b8_c32k", 8, 32768),
    ("b16_c16k", 16, 16384),
    ("b16_c32k", 16, 32768),
    ("b32_c16k", 32, 16384),
)
MODES = (
    "expert-only",
    "local-kv-only",
    "cxl-kv-only",
    "local-kv-plus-expert",
    "cxl-kv-plus-local-kv",
    "cxl-kv-plus-local-kv-plus-expert",
)
PROFILES: dict[str, dict[str, float | str | int]] = {
    "conservative-assumption": {
        "bandwidth_gb_s": 32.0,
        "latency_us": 0.5,
        "channel_count": 4,
        "capacity_gb": 64,
        "transaction_bytes": 32,
        "queue_depth": 256,
        "address_mapping": "static-spill-offset; pass-through channel mapper",
    },
    "nominal-assumption": {
        "bandwidth_gb_s": 64.0,
        "latency_us": 0.25,
        "channel_count": 4,
        "capacity_gb": 64,
        "transaction_bytes": 32,
        "queue_depth": 256,
        "address_mapping": "static-spill-offset; pass-through channel mapper",
    },
    "optimistic-assumption": {
        "bandwidth_gb_s": 128.0,
        "latency_us": 0.1,
        "channel_count": 4,
        "capacity_gb": 64,
        "transaction_bytes": 32,
        "queue_depth": 256,
        "address_mapping": "static-spill-offset; pass-through channel mapper",
    },
}


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _execution_provenance() -> dict[str, str]:
    revision = subprocess.check_output(
        ["git", "-C", str(RAMULATOR), "rev-parse", "HEAD"], text=True
    ).strip()
    bindings = sorted((RAMULATOR / "python/ramulator").glob("_ramulator.cpython-*.so"))
    if len(bindings) != 1:
        raise ValueError("expected exactly one CXL Ramulator Python binding")
    library = RAMULATOR / "libramulator.so"
    if not library.is_file():
        raise ValueError("CXL Ramulator shared library is missing")
    return {
        "ramulator_commit": revision,
        "binding_sha256": _sha256(bindings[0]),
        "ramulator_library_sha256": _sha256(library),
    }


def _key(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def _capacity_rows() -> dict[tuple[int, int], dict[str, Any]]:
    raw = json.loads(CAPACITY.read_text(encoding="utf-8"))
    rows = {}
    for row in raw["rows"]:
        if row["domain"] == "a800" and row["cxl_capacity_bytes"] == 64_000_000_000:
            rows[(int(row["batch_size"]), int(row["context_length"]))] = row
    return rows


def _kv_transactions(batch: int, context: int, transaction_bytes: int) -> int:
    model = load_configuration(SOURCE_EXPERIMENT).model
    bytes_per_layer = (
        2 * batch * context * model.num_key_value_heads * model.head_dim * model.dtype_bytes
    )
    return math.ceil(bytes_per_layer / transaction_bytes)


def _expert_shape(batch: int) -> dict[str, int]:
    configuration = load_configuration(SOURCE_EXPERIMENT)
    source = load_trace(configuration.experiment.trace_path, configuration.model, 0, 0)
    if source.batch_size != 8:
        raise ValueError("the controlled routing template must be the B8 real pilot")
    factor = batch // source.batch_size
    if factor <= 0 or batch % source.batch_size:
        raise ValueError("pressure batch must be a positive multiple of the B8 template")
    loads = tuple(
        ExpertLoad(load.expert_id, load.token_count * factor)
        for load in source.expert_loads
    )
    hot = tuple(sorted(loads, key=lambda load: (-load.token_count, load.expert_id)))
    split = (len(hot) + 1) // 2
    cycle = SieveCycleV1Config.load(CYCLE)
    return asdict(shape_for_loads(hot[:split], hot[split:], configuration.model, cycle))


def _workload_shape(
    batch: int,
    context: int,
    mode: str,
    capacity: dict[str, Any],
    request_scale: int,
) -> dict[str, int]:
    if request_scale <= 0:
        raise ValueError("request_scale must be positive")
    cycle = SieveCycleV1Config.load(CYCLE)
    expert = _expert_shape(batch)
    expert = {
        key: math.ceil(value / request_scale)
        for key, value in expert.items()
    }
    total_kv = math.ceil(_kv_transactions(batch, context, cycle.transaction_bytes) / request_scale)
    kv_cache = int(capacity["kv_cache_bytes"])
    spill_bytes = int(capacity["spill_bytes"])
    cxl_kv = math.ceil(total_kv * spill_bytes / kv_cache) if kv_cache else 0
    local_kv = total_kv - cxl_kv
    result = {
        "gpu_read_transactions": 0,
        "local_kv_read_transactions": 0,
        "cxl_kv_read_transactions": 0,
        "pim_gwrite_waves": 0,
        "pim_mac_waves": 0,
        "pim_read_waves": 0,
    }
    if mode == "expert-only":
        result.update(expert)
    elif mode == "local-kv-only":
        result["local_kv_read_transactions"] = total_kv
    elif mode == "cxl-kv-only":
        result["cxl_kv_read_transactions"] = cxl_kv
    elif mode == "local-kv-plus-expert":
        result.update(expert)
        result["local_kv_read_transactions"] = local_kv
    elif mode == "cxl-kv-plus-local-kv":
        result["local_kv_read_transactions"] = local_kv
        result["cxl_kv_read_transactions"] = cxl_kv
    elif mode == "cxl-kv-plus-local-kv-plus-expert":
        result.update(expert)
        result["local_kv_read_transactions"] = local_kv
        result["cxl_kv_read_transactions"] = cxl_kv
    else:
        raise ValueError(f"unknown Stage-4 mode: {mode}")
    if not any(result.values()):
        raise ValueError(f"empty workload for {batch=} {context=} {mode=}")
    return result


def _row(
    case: str,
    batch: int,
    context: int,
    mode: str,
    profile_name: str,
    profile: dict[str, Any],
    capacity: dict[str, Any],
    shape: dict[str, int],
    result: CXLKVWorkloadResult,
    cache_key: str,
) -> dict[str, Any]:
    row = {
        "case": case,
        "batch_size": batch,
        "context_length": context,
        "mode": mode,
        "profile": profile_name,
        "classification": "Ramulator request-level memory-only CXL sensitivity; no A800 timing",
        "capacity_domain": "a800-80gb",
        "capacity_state": capacity["state"],
        "spill_bytes": int(capacity["spill_bytes"]),
        "kv_cache_bytes": int(capacity["kv_cache_bytes"]),
        "cache_key": cache_key,
        **shape,
        **asdict(result),
        "total_completion_us": result.total_completion_us,
        "cxl_queue_wait_us": result.cxl_queue_wait_us,
        "cxl_read_latency_us": result.cxl_read_latency_us,
        "cxl_link_busy_ratio": result.cxl_link_busy_ratio,
        "cxl_bandwidth_gb_s": profile["bandwidth_gb_s"],
        "cxl_latency_us": profile["latency_us"],
    }
    return row


def run(
    output_dir: str | Path,
    *,
    profiles: tuple[str, ...] = tuple(PROFILES),
    request_scale: int = 4096,
    cases: tuple[str, ...] = tuple(case for case, _, _ in PRESSURE_POINTS),
    modes: tuple[str, ...] = MODES,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cycle = SieveCycleV1Config.load(CYCLE)
    capacity_rows = _capacity_rows()
    context = cxl_kv_cache_context(ROOT, CYCLE)
    selected_profiles = tuple(profiles)
    if not selected_profiles or any(name not in PROFILES for name in selected_profiles):
        raise ValueError("Stage-4 profiles must be selected from the declared assumptions")
    pressure_by_case = {case: (batch, context_length) for case, batch, context_length in PRESSURE_POINTS}
    selected_cases = tuple(cases)
    if not selected_cases or any(case not in pressure_by_case for case in selected_cases):
        raise ValueError("Stage-4 cases must be selected from the declared pressure points")
    selected_modes = tuple(modes)
    if not selected_modes or any(mode not in MODES for mode in selected_modes):
        raise ValueError("Stage-4 modes must be selected from the declared modes")

    rows: list[dict[str, Any]] = []
    cache_dir = output / "exact_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for case in selected_cases:
        batch, context_length = pressure_by_case[case]
        capacity = capacity_rows[(batch, context_length)]
        for profile_name in selected_profiles:
            profile = PROFILES[profile_name]
            for mode in selected_modes:
                shape = _workload_shape(batch, context_length, mode, capacity, request_scale)
                cache_input = {
                    "context": context,
                    "case": case,
                    "mode": mode,
                    "profile": profile_name,
                    "profile_parameters": profile,
                    "request_scale": request_scale,
                    "shape": shape,
                    "capacity": {
                        "domain": "a800",
                        "local_capacity_bytes": int(capacity["local_capacity_bytes"]),
                        "cxl_capacity_bytes": int(capacity["cxl_capacity_bytes"]),
                        "spill_bytes": int(capacity["spill_bytes"]),
                    },
                }
                cache_key = _key(cache_input)
                cache_path = cache_dir / f"{cache_key}.json"
                if cache_path.is_file():
                    cached = json.loads(cache_path.read_text(encoding="utf-8"))
                    if cached.get("cache_input") != cache_input:
                        raise ValueError(f"cache provenance mismatch: {cache_path}")
                    result = CXLKVWorkloadResult(**cached["result"])
                else:
                    result = run_cxl_kv_workload(
                        RAMULATOR,
                        cycle,
                        **shape,
                        cxl_bandwidth_bytes_per_second=float(profile["bandwidth_gb_s"]) * 1e9,
                        cxl_latency_us=float(profile["latency_us"]),
                        cxl_channels=int(profile["channel_count"]),
                    )
                    cache_path.write_text(
                        json.dumps(
                            {"cache_input": cache_input, "result": asdict(result)},
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                expected = shape["cxl_kv_read_transactions"]
                if result.cxl_kv_completed_requests != expected:
                    raise ValueError("CXL request count did not complete exactly")
                rows.append(_row(case, batch, context_length, mode, profile_name, profile, capacity, shape, result, cache_key))

    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "schema_version": 1,
        "method": "cxl-memory-only-request-level-stage4-v1",
        "classification": "Ramulator request-level sensitivity; CXL profiles are assumptions, not A800 measurements",
        "ramulator_commit": cycle.ramulator_commit,
        "cycle_config": str(CYCLE.relative_to(ROOT)),
        "cycle_config_sha256": _sha256(CYCLE),
        "simulation_context": context,
        "execution": _execution_provenance(),
        "pressure_points": [
            {"case": case, "batch_size": batch, "context_length": context_length}
            for case, batch, context_length in PRESSURE_POINTS
            if case in selected_cases
        ],
        "modes": list(selected_modes),
        "profiles": {name: PROFILES[name] for name in selected_profiles},
        "capacity_source": str(CAPACITY.relative_to(ROOT)),
        "capacity_source_sha256": _sha256(CAPACITY),
        "routing_template": "manifest-validated real B8/C4k step0/layer0, tiled by integer batch factor",
        "request_scale": request_scale,
        "rows": len(rows),
        "exact_runs": len(rows),
        "interpolation": 0,
        "limitations": [
            "CXL controller is a memory-only FIFO service model with explicit assumed bandwidth and latency.",
            "Static spill fraction is derived from the Stage-2 A800 80 GB capacity admission rows.",
            "No paging, eviction, recovery, CXL-PIM computation, or physical CXL device measurement.",
            "Expert routing is a controlled template and is not a new A800 capture.",
            f"KV and Expert/PIM request counts use request_scale={request_scale}; scale=1 represents the full generated request count.",
            "This stage validates request-level counters and pressure trends before full Decode integration.",
        ],
        "input_hashes": {
            "source_experiment": _sha256(SOURCE_EXPERIMENT),
            "source_trace": _sha256(ROOT / "traces/real/qwen3_30b_a3b/kv_b8_c4k/router.jsonl"),
            "source_manifest": _sha256(ROOT / "traces/real/qwen3_30b_a3b/kv_b8_c4k/manifest.json"),
            "model": _sha256(ROOT / "configs/models/qwen3_30b_a3b.json"),
        },
    }
    metadata["output_hashes"] = {
        "comparison.csv": _sha256(output / "comparison.csv"),
    }
    (output / "comparison.json").write_text(
        json.dumps({"metadata": metadata, "rows": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata["output_hashes"]["comparison.json"] = _sha256(output / "comparison.json")
    report_lines = [
        "# KV CXL request-level Stage 4",
        "",
        "本报告是 Ramulator request-level memory-only CXL 敏感性实验。Local HBM controller",
        "继续承载 Expert READ、PIM 命令和 resident KV READ；CXL controller 只承载静态 spilled KV READ。",
        "三个带宽/时延 profile 是明确的模型假设，不是 A800 或真实 CXL 设备测量。",
        f"本轮 request_scale={request_scale}；scale=1 使用完整生成请求量，其他值是统一缩放的微基准。",
        "",
        "| case | profile | mode | CXL READ (M) | total (us) | CXL queue wait (us) | link busy |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        report_lines.append(
            f"| {row['case']} | {row['profile']} | {row['mode']} | "
            f"{row['cxl_kv_read_transactions'] / 1e6:.6f} | {row['total_completion_us']:.6f} | "
            f"{row['cxl_queue_wait_us']:.6f} | {row['cxl_link_busy_ratio']:.6f} |"
        )
    report_lines += [
        "",
        "所有行都由精确 request-level simulation 生成，interpolation=0。这里的 CXL READ",
        "数量是每个压力点一个 Decode layer 的受控请求量；结果不能直接解释为完整 48 层",
        "端到端时延，下一步才接入 Decode event graph。",
    ]
    (output / "stage4_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    metadata["output_hashes"]["stage4_report.md"] = _sha256(output / "stage4_report.md")
    (output / "stage_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "method": metadata["method"],
                "input_hashes": metadata["input_hashes"],
                "execution": metadata["execution"],
                "output_hashes": metadata["output_hashes"],
                "rows": len(rows),
                "exact_runs": len(rows),
                "interpolation": 0,
                "request_scale": request_scale,
                "profiles": list(selected_profiles),
                "cases": list(selected_cases),
                "modes": list(selected_modes),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"metadata": metadata, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "results/kv_cxl_request_stage4_v1"))
    parser.add_argument("--profiles", nargs="+", choices=tuple(PROFILES), default=list(PROFILES))
    parser.add_argument("--request-scale", type=int, default=4096)
    parser.add_argument("--cases", nargs="+", choices=tuple(case for case, _, _ in PRESSURE_POINTS), default=[case for case, _, _ in PRESSURE_POINTS])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    args = parser.parse_args(argv)
    result = run(
        args.output,
        profiles=tuple(args.profiles),
        request_scale=args.request_scale,
        cases=tuple(args.cases),
        modes=tuple(args.modes),
    )
    print(json.dumps({"rows": len(result["rows"]), "output": str(Path(args.output).resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
