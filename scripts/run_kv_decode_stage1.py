#!/usr/bin/env python3
"""Run stage-1 full Decode replays with an explicit local-HBM KV READ event.

The existing five real-router cycle-v1 replays remain immutable. This command
creates deterministic configuration snapshots under the new output directory,
replays every configured policy in serial and overlap modes, and writes a
comparison manifest. KV timing is analytic local-HBM decomposition; Expert
timing remains the exact existing cycle-v1 table.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from freeze_kv_baseline import freeze_or_verify  # noqa: E402
from sieve_replay.cli import main as replay_main  # noqa: E402
from sieve_replay.config import load_configuration  # noqa: E402
from sieve_replay.report.writer import sha256_file  # noqa: E402


CASES = ("b8_c4k", "b8_c8k", "b8_c16k", "b16_c4k", "b16_c8k")
MODES = ("serial-local-hbm-v1", "overlap-local-hbm-v1")
POLICIES = (
    "gpu-only",
    "noexp",
    "allexp",
    "pimoe",
    "sieve",
    "sieve-cycle-v1",
    "sieve-fixed-16-cycle-v1",
)
BASELINE_ROOT = ROOT / "results/full_decode_real_kv_cycle_v1"
DEFAULT_OUTPUT = ROOT / "results/full_decode_real_kv_stage1_v1"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _config_snapshot(output: Path, case: str, mode: str) -> Path:
    source = ROOT / f"configs/experiments/full_decode_real_kv_{case}.json"
    raw = _read_json(source)
    raw["name"] = f"full-decode-real-kv-stage1-{mode}-{case}"
    raw["kv_read_mode"] = mode
    path = output / "config_snapshots" / f"{mode}_{case}.json"
    _write_json(path, raw)
    return path


def _validate_replay(
    output: Path,
    case: str,
    mode: str,
    policy: str,
    baseline_summary: dict[str, Any],
) -> dict[str, Any]:
    directory = output / mode / case / policy
    summary = _read_json(directory / "summary.json")
    if summary.get("kv_read", {}).get("mode") != mode:
        raise ValueError(f"{case}/{mode}/{policy}: KV mode missing or mismatched")
    expected_events = 8 * 48 * 19
    with (directory / "events.csv").open(encoding="utf-8", newline="") as handle:
        event_rows = list(csv.DictReader(handle))
    if len(event_rows) != expected_events:
        raise ValueError(
            f"{case}/{mode}/{policy}: expected {expected_events} events, got {len(event_rows)}"
        )
    kv_rows = [row for row in event_rows if row.get("category") == "kv_read"]
    if len(kv_rows) != 8 * 48:
        raise ValueError(f"{case}/{mode}/{policy}: incomplete KV event coverage")
    baseline_total = float(baseline_summary["total_latency_us"])
    total = float(summary["total_latency_us"])
    return {
        "case": case,
        "mode": mode,
        "policy": policy,
        "total_latency_us": total,
        "baseline_total_latency_us": baseline_total,
        "delta_vs_baseline_us": total - baseline_total,
        "throughput_request_tokens_per_s": float(
            summary["throughput_request_tokens_per_s"]
        ),
        "kv_read_bytes": int(summary["kv_read"]["total_read_bytes"]),
        "kv_read_critical_path_us": float(
            summary["kv_read"]["critical_path_latency_us"]
        ),
        "attention_critical_path_us": float(
            summary["critical_path_duration_us_by_category"].get("attention", 0.0)
        ),
        "memory_peak_bytes": int(summary["memory"]["peak_total_bytes"]),
        "summary_sha256": sha256_file(directory / "summary.json"),
        "events_sha256": sha256_file(directory / "events.csv"),
        "run_manifest_sha256": sha256_file(directory / "run_manifest.json"),
    }


def _write_report(output: Path, rows: list[dict[str, Any]], frozen: dict[str, Any]) -> None:
    fields = list(rows[0])
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _write_json(
        output / "comparison.json",
        {
            "schema_version": 1,
            "classification": (
                "A800-captured routing + exact cycle-v1 Expert timing + "
                "analytic local-HBM KV READ decomposition"
            ),
            "rows": rows,
            "frozen_baseline_inventory_sha256": frozen["inventory_sha256"],
            "limitations": [
                "KV READ timing is analytic local-HBM bandwidth, not a new Ramulator KV run",
                "Expert timing remains the exact existing cycle-v1 table",
                "serial mode waits for KV READ before attention compute",
                "overlap mode joins KV READ and attention compute after RoPE",
                "no paging, spill, eviction, CXL, or physical GPU memory trace",
            ],
        },
    )
    by_case_mode_policy = {
        (row["case"], row["mode"], row["policy"]): row for row in rows
    }
    lines = [
        "# KV Decode stage 1: explicit local-HBM KV READ",
        "",
        "本阶段把 KV READ 接入完整 Attention → Router → Expert → Combine 回放。旧的五组",
        "cycle-v1 基线没有被覆盖；每个新配置都在输出目录保存了完整 JSON snapshot。",
        "",
        "KV READ 使用 `analytic-local-hbm-v1`，Expert 使用已有精确 cycle-v1 contention 表。",
        "因此本阶段验证端到端事件依赖和 accounting，不宣称 KV 请求级 Ramulator 或真实",
        "A800 带宽结果。",
        "",
        "## 代表性 sieve-cycle-v1 结果",
        "",
        "| 配置 | 基线 us | serial us | overlap us | serial delta | overlap delta | KV bytes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in CASES:
        serial = by_case_mode_policy[(case, "serial-local-hbm-v1", "sieve-cycle-v1")]
        overlap = by_case_mode_policy[(case, "overlap-local-hbm-v1", "sieve-cycle-v1")]
        lines.append(
            "| {case} | {base:.3f} | {serial:.3f} | {overlap:.3f} | {sd:+.3f} | {od:+.3f} | {bytes:,} |".format(
                case=case,
                base=serial["baseline_total_latency_us"],
                serial=serial["total_latency_us"],
                overlap=overlap["total_latency_us"],
                sd=serial["delta_vs_baseline_us"],
                od=overlap["delta_vs_baseline_us"],
                bytes=serial["kv_read_bytes"],
            )
        )
    lines += [
        "",
        "串行模式的总时延应与旧基线保持一致，因为它只是把原 Attention 时延拆为 KV READ",
        "和剩余 Attention 计算。Overlap 模式允许两者在 RoPE 后并行，再由 o_proj 等待两者",
        "完成；这是假设性执行语义，用于下一阶段请求级模型的接口验证。",
        "",
        "冻结基线 inventory SHA-256: `" + frozen["inventory_sha256"] + "`",
        "",
    ]
    (output / "stage1_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"stage-1 output already exists and is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    frozen = freeze_or_verify(verify_only=True)
    rows: list[dict[str, Any]] = []
    manifests: dict[str, Any] = {}
    for mode in MODES:
        for case in CASES:
            config_path = _config_snapshot(output, case, mode)
            loaded = load_configuration(config_path)
            if loaded.experiment.kv_read_mode != mode:
                raise ValueError(f"snapshot mode mismatch: {config_path}")
            status = replay_main(
                [
                    "run-all",
                    "--experiment",
                    str(config_path),
                    "--output",
                    str(output / mode / case),
                ]
            )
            if status:
                raise RuntimeError(f"replay failed: {mode}/{case}")
            case_rows: list[dict[str, Any]] = []
            for policy in POLICIES:
                baseline = _read_json(BASELINE_ROOT / case / policy / "summary.json")
                row = _validate_replay(output, case, mode, policy, baseline)
                rows.append(row)
                case_rows.append(row)
            manifests[f"{mode}/{case}"] = {
                "config_sha256": sha256_file(config_path),
                "config_path": str(config_path.relative_to(ROOT)),
                "rows": case_rows,
            }
    _write_report(output, rows, frozen)
    stage = {
        "schema_version": 1,
        "stage": "kv-decode-stage1-v1",
        "classification": (
            "A800-captured routing + exact cycle-v1 Expert timing + "
            "analytic local-HBM KV READ decomposition"
        ),
        "cases": list(CASES),
        "modes": list(MODES),
        "policies": list(POLICIES),
        "frozen_baseline_inventory_sha256": frozen["inventory_sha256"],
        "replays": manifests,
        "comparison_sha256": sha256_file(output / "comparison.json"),
        "report_sha256": sha256_file(output / "stage1_report.md"),
        "limitations": [
            "No KV request-level Ramulator integration in this stage",
            "No A800 timing or memory measurements",
            "No paging, spill, eviction or CXL",
        ],
    }
    _write_json(output / "stage_manifest.json", stage)
    return {
        "cases": len(CASES),
        "modes": len(MODES),
        "policies": len(POLICIES),
        "replays": len(rows),
        "output": str(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--refresh-report",
        action="store_true",
        help="rewrite the report and manifest hashes from an existing completed output",
    )
    args = parser.parse_args()
    if args.refresh_report:
        frozen = freeze_or_verify(verify_only=True)
        payload = json.loads((args.output / "comparison.json").read_text(encoding="utf-8"))
        _write_report(args.output, payload["rows"], frozen)
        stage = _read_json(args.output / "stage_manifest.json")
        stage["report_sha256"] = sha256_file(args.output / "stage1_report.md")
        _write_json(args.output / "stage_manifest.json", stage)
        print(json.dumps({"output": str(args.output), "refreshed": True}, sort_keys=True))
        return 0
    print(json.dumps(run(args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
