#!/usr/bin/env python3
"""Validate and summarize the long-context cycle-v1 replay outputs.

The command is deliberately fail-closed: it requires all five real-router
manifests and all seven policy replays before writing any aggregate output.
It only consumes replay artifacts; it never estimates missing timing or
silently reuses the analytic KV sweep.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from sieve_replay.config import load_configuration
from sieve_replay.trace import load_trace_set
from sieve_replay.trace_manifest import TraceManifest

CASES = ("b8_c4k", "b8_c8k", "b8_c16k", "b16_c4k", "b16_c8k")
POLICIES = (
    "gpu-only", "noexp", "allexp", "pimoe", "sieve", "sieve-cycle-v1",
    "sieve-fixed-16-cycle-v1",
)
LAYERS = 48
STEPS = 8
KV_PER_TOKEN = 98_304
SIM_CAPACITY = 96_000_000_000
A800_CAPACITY = 80_000_000_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def config_case(root: Path, case: str) -> tuple[Path, dict[str, Any]]:
    path = root / "configs" / "experiments" / f"full_decode_real_kv_{case}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, read_json(path)


def validate_trace(root: Path, case: str, cfg: dict[str, Any], config_path: Path) -> dict[str, Any]:
    trace = root / cfg["trace"]
    manifest_path = root / cfg["trace_manifest"]
    if not trace.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"{case}: trace or manifest missing")
    configuration = load_configuration(config_path)
    model = configuration.model
    kv_per_token = (2 * model.num_hidden_layers * model.num_key_value_heads
                    * model.head_dim * model.dtype_bytes)
    if (kv_per_token != KV_PER_TOKEN
            or int(configuration.hardware.hbm_pim_capacity_gb * 1e9) != SIM_CAPACITY
            or configuration.experiment.layers != tuple(range(LAYERS))
            or configuration.experiment.steps != tuple(range(STEPS))):
        raise ValueError(f"{case}: model, capacity or Decode scope differs from this KV protocol")
    trace_set = load_trace_set(trace, configuration.model, configuration.experiment.layers, configuration.experiment.steps)
    TraceManifest.load_and_validate(manifest_path, trace, configuration.model, trace_set)
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != 2:
        raise ValueError(f"{case}: unsupported manifest schema")
    if manifest.get("trace_file") != trace.name:
        raise ValueError(f"{case}: manifest trace_file mismatch")
    expected = manifest.get("trace_sha256")
    if expected != sha256(trace):
        raise ValueError(f"{case}: router trace SHA-256 mismatch")
    prompt_name = manifest.get("prompts", {}).get("file")
    prompts = manifest_path.parent / prompt_name if prompt_name else None
    if prompts is None or not prompts.is_file():
        raise FileNotFoundError(f"{case}: prompt snapshot missing")
    if manifest["prompts"].get("sha256") != sha256(prompts):
        raise ValueError(f"{case}: prompt SHA-256 mismatch")
    workload = manifest.get("workload", {})
    if workload.get("decode_steps") != STEPS:
        raise ValueError(f"{case}: expected {STEPS} decode steps")
    batch = int(workload.get("batch_size", 0))
    expected_records = batch * LAYERS * STEPS
    with trace.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if len(records) != expected_records:
        raise ValueError(f"{case}: expected {expected_records} router records, got {len(records)}")
    contexts = [int(r["context_length"]) for r in records]
    context = min(contexts)
    if context <= 0 or max(contexts) != context + STEPS - 1:
        raise ValueError(f"{case}: context lengths are not contiguous decode steps")
    return {
        "case": case, "batch_size": batch, "context_length": context,
        "trace_sha256": sha256(trace), "manifest_sha256": sha256(manifest_path),
        "prompt_sha256": manifest["prompts"]["sha256"], "records": len(records),
        "trace_relpath": cfg["trace"], "manifest_relpath": cfg["trace_manifest"],
        "config_paths": {k: cfg.get(k) for k in ("model", "hardware", "pim_timing_table", "contention_timing_table", "ramulator_cycle_config")},
        "config_snapshot": cfg,
    }


def _f(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing numeric column {key}") from exc


def summarize_replay(root: Path, case: str, policy: str, trace_meta: dict[str, Any]) -> dict[str, Any]:
    directory = root / "results" / "full_decode_real_kv_cycle_v1" / case / policy
    stage_path = directory.parent.parent / "stage_manifest.json"
    if not stage_path.is_file():
        raise FileNotFoundError(f"{case}/{policy}: stage_manifest.json missing")
    stage = read_json(stage_path)
    stage_case = stage.get("cases", {}).get(case)
    if not isinstance(stage_case, dict) or stage_case.get("configuration_snapshot") != trace_meta.get("config_snapshot"):
        raise ValueError(f"{case}: stage manifest configuration snapshot mismatch")
    replay_stage = stage_case.get("replays", {}).get(policy)
    if not isinstance(replay_stage, dict):
        raise ValueError(f"{case}/{policy}: stage manifest replay entry missing")
    exact_path = root / "results/kv_workload_timing_v1/exact_workloads.json"
    if not exact_path.is_file() or stage.get("exact_workloads_sha256") != sha256(exact_path):
        raise ValueError(f"{case}: exact workload evidence hash mismatch")
    exact = read_json(exact_path)
    exact_summary = exact.get("summary", {})
    if (exact_summary.get("failed") != 0 or exact_summary.get("remaining") != 0
            or exact.get("interpolated_workloads") != 0
            or exact_summary.get("required_unique_shapes", 0) <= 0
            or exact_summary.get("completed") != exact_summary.get("required_unique_shapes")
            or len(exact.get("entries", {})) != exact_summary.get("completed")):
        raise ValueError(f"{case}: exact workload evidence is incomplete")
    required = ["summary.json", "run_manifest.json", "layers.csv", "events.csv"]
    if any(not (directory / name).is_file() for name in required):
        raise FileNotFoundError(f"{case}/{policy}: incomplete replay directory")
    summary = read_json(directory / "summary.json")
    run_manifest = read_json(directory / "run_manifest.json")
    hashes = run_manifest.get("input_hashes", {})
    if summary.get("input_hashes") is not None and summary.get("input_hashes") != hashes:
        raise ValueError(f"{case}/{policy}: summary input hashes differ from run manifest")
    if hashes.get("trace_sha256") != trace_meta["trace_sha256"]:
        raise ValueError(f"{case}/{policy}: run manifest trace hash mismatch")
    if hashes.get("trace_manifest_sha256") != trace_meta["manifest_sha256"]:
        raise ValueError(f"{case}/{policy}: run manifest manifest hash mismatch")
    prompt_hash = hashes.get("prompts_sha256", hashes.get("prompt_sha256"))
    if prompt_hash is not None and prompt_hash != trace_meta["prompt_sha256"]:
        raise ValueError(f"{case}/{policy}: run manifest prompt hash mismatch")
    if run_manifest.get("policy") != policy:
        raise ValueError(f"{case}/{policy}: policy mismatch")
    if summary.get("policy") != policy:
        raise ValueError(f"{case}/{policy}: summary policy mismatch")
    backend = summary.get("timing_backend") or run_manifest.get("hardware", {}).get("timing_backend")
    if backend != "ramulator-contention-v1":
        raise ValueError(f"{case}/{policy}: expected ramulator-contention-v1, got {backend}")
    # Every recorded hash must resolve to the exact file named by the replay manifest.
    experiment = run_manifest.get("experiment", {})
    if experiment.get("trace") != trace_meta.get("trace_relpath", experiment.get("trace")):
        # The hash check below is authoritative; this catches stale replay configs early.
        raise ValueError(f"{case}/{policy}: run manifest trace path differs from current case")
    if experiment.get("trace_manifest") != trace_meta.get("manifest_relpath"):
        raise ValueError(f"{case}/{policy}: run manifest manifest path differs from current case")
    for name, expected_path in trace_meta["config_paths"].items():
        if expected_path is not None and experiment.get(name) != expected_path:
            raise ValueError(f"{case}/{policy}: run manifest {name} path differs from current case")
    hash_paths = {
        "model_sha256": experiment.get("model"),
        "hardware_sha256": experiment.get("hardware"),
        "pim_timing_table_sha256": experiment.get("pim_timing_table"),
        "contention_timing_table_sha256": experiment.get("contention_timing_table"),
        "ramulator_cycle_config_sha256": experiment.get("ramulator_cycle_config"),
    }
    for key, rel in hash_paths.items():
        if not rel or key not in hashes:
            raise ValueError(f"{case}/{policy}: missing {key} input hash/path")
        path = root / rel
        if not path.is_file() or sha256(path) != hashes[key]:
            raise ValueError(f"{case}/{policy}: {key} mismatch")
    contention_path = root / hash_paths["contention_timing_table_sha256"]
    evidence_path = contention_path.with_name(f"{contention_path.stem}.evidence.json")
    if not evidence_path.is_file():
        raise ValueError(f"{case}/{policy}: contention evidence missing: {evidence_path}")
    pim_path = root / hash_paths["pim_timing_table_sha256"]
    case_file_paths = {
        "pim_timing_table_sha256": pim_path,
        "pim_evidence_sha256": pim_path.with_suffix(".evidence.json"),
        "contention_timing_table_sha256": contention_path,
        "contention_evidence_sha256": evidence_path,
    }
    for field, path in case_file_paths.items():
        expected_hash = stage_case.get(field)
        if not path.is_file() or expected_hash != sha256(path):
            raise ValueError(f"{case}: stage manifest {field} mismatch")
    if "trace_manifest_sha256" not in hashes or "trace_sha256" not in hashes:
        raise ValueError(f"{case}/{policy}: missing trace input hashes")
    with (directory / "layers.csv").open(encoding="utf-8", newline="") as f:
        layers = list(csv.DictReader(f))
    for field, filename in (("layers_sha256", "layers.csv"), ("events_sha256", "events.csv"), ("summary_sha256", "summary.json"), ("run_manifest_sha256", "run_manifest.json")):
        expected_hash = replay_stage.get(field)
        if expected_hash is None or expected_hash != sha256(directory / filename):
            raise ValueError(f"{case}/{policy}: stage manifest {field} mismatch")
    if len(layers) != LAYERS * STEPS:
        raise ValueError(f"{case}/{policy}: expected {LAYERS * STEPS} layer rows, got {len(layers)}")
    layer_keys = {(int(r["step"]), int(r["layer"])) for r in layers}
    if len(layer_keys) != len(layers) or layer_keys != {(s, l) for s in range(STEPS) for l in range(LAYERS)}:
        raise ValueError(f"{case}/{policy}: layers.csv does not contain unique complete step/layer keys")
    with (directory / "events.csv").open(encoding="utf-8", newline="") as f:
        events = list(csv.DictReader(f))
    event_keys = {(int(r["step"]), int(r["layer"])) for r in events}
    if event_keys != {(s, l) for s in range(STEPS) for l in range(LAYERS)}:
        raise ValueError(f"{case}/{policy}: events.csv does not cover every step/layer")
    attention = sum(_f(r, "duration_us") for r in events if r.get("category") == "attention")
    expert = 0.0
    for step in range(STEPS):
        for layer in range(LAYERS):
            rows = [r for r in events if int(r.get("step", -1)) == step and int(r.get("layer", -1)) == layer]
            gpu = sum(_f(r, "duration_us") for r in rows if r.get("category") in {"gpu_weight_load", "gpu_expert"})
            pim = sum(_f(r, "duration_us") for r in rows if r.get("category") in {"pim_write", "pim_expert", "pim_read"})
            if not rows:
                raise ValueError(f"{case}/{policy}: missing events for step={step}, layer={layer}")
            expert += max(gpu, pim)
    memory = summary.get("memory", {})
    total = float(summary.get("total_latency_us", 0.0))
    if total <= 0:
        raise ValueError(f"{case}/{policy}: invalid total latency")
    critical = summary.get("critical_path_duration_us_by_category", {})
    if not critical or abs(sum(float(v) for v in critical.values()) - total) > 1e-5:
        raise ValueError(f"{case}/{policy}: critical-path decomposition does not match total latency")
    critical_attention = float(critical.get("attention", 0.0))
    critical_expert = sum(float(critical.get(k, 0.0)) for k in ("gpu_weight_load", "gpu_expert", "pim_write", "pim_expert", "pim_read"))
    if abs(attention - critical_attention) > 1e-5 or abs(expert - critical_expert) > 1e-5:
        raise ValueError(f"{case}/{policy}: event/category decomposition disagrees with critical path")
    if "peak_kv_cache_bytes" not in memory:
        raise ValueError(f"{case}/{policy}: missing peak_kv_cache_bytes")
    kv = int(memory["peak_kv_cache_bytes"])
    peak = int(memory.get("peak_total_bytes", 0))
    if peak <= 0:
        raise ValueError(f"{case}/{policy}: missing peak memory")
    return {
        **trace_meta, "policy": policy, "total_latency_us": total,
        "run_manifest_sha256": sha256(directory / "run_manifest.json"),
        "stage_manifest_sha256": sha256(stage_path),
        "run_input_hashes": hashes,
        "throughput_request_tokens_per_s": float(summary.get("throughput_request_tokens_per_s", 0.0)),
        "attention_latency_us": attention, "expert_latency_us": expert,
        "attention_latency_share": attention / total, "kv_cache_bytes": kv,
        "critical_path_scheduler_us": float(critical.get("scheduler", 0.0)),
        "critical_path_other_us": total - float(critical.get("attention", 0.0)) - float(critical.get("scheduler", 0.0)) - sum(float(critical.get(k, 0.0)) for k in ("gpu_weight_load", "gpu_expert", "pim_write", "pim_expert", "pim_read")),
        "peak_memory_bytes": peak, "simulated_capacity_bytes": SIM_CAPACITY,
        "a800_physical_capacity_bytes": A800_CAPACITY,
        "simulated_capacity_state": "feasible" if peak <= SIM_CAPACITY else "infeasible",
        "a800_capacity_state": "feasible" if peak <= A800_CAPACITY else "infeasible",
        "timing_backend": backend,
        "result_classification": "Ramulator模拟：GPU解析时延 + PIM cycle-v0/cycle-v1请求级时序",
        "contention_evidence_sha256": sha256(evidence_path),
    }


def write_plots(rows: list[dict[str, Any]], out: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to generate scientific plots") from exc
    out.mkdir(parents=True, exist_ok=True)
    oracle = [r for r in rows if r["policy"] == "sieve-cycle-v1"]
    def line(field: str, name: str, ylabel: str, selected: list[dict[str, Any]]) -> None:
        plt.figure(figsize=(7.2, 4.4))
        for batch in sorted({r["batch_size"] for r in selected}):
            x = sorted((r for r in selected if r["batch_size"] == batch), key=lambda r: r["context_length"])
            plt.plot([r["context_length"] for r in x], [r[field] for r in x], marker="o", label=f"B={batch}")
        plt.title("Real-router simulation | 48 layers × 8 decode steps"); plt.xlabel("Context length"); plt.ylabel(ylabel); plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(out / name, dpi=180); plt.close()
    line("kv_cache_bytes", "context_kv_cache.png", "KV cache bytes", oracle)
    plt.figure(figsize=(7.2, 4.4))
    for policy, style in (("gpu-only", "--"), ("sieve-cycle-v1", "-")):
        selected = [r for r in rows if r["policy"] == policy]
        for batch in sorted({r["batch_size"] for r in selected}):
            x = sorted((r for r in selected if r["batch_size"] == batch), key=lambda r: r["context_length"])
            plt.plot([r["context_length"] for r in x], [r["attention_latency_us"] for r in x], marker="o", linestyle=style, label=f"{policy}, B={batch}")
    plt.title("Real-router simulation | Attention: GPU analytic / PIM cycle-v0"); plt.xlabel("Context length"); plt.ylabel("Attention latency (us)"); plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(out / "context_attention_latency.png", dpi=180); plt.close()
    plt.figure(figsize=(7.2, 4.4))
    for policy, style in (("gpu-only", "--"), ("sieve-cycle-v1", "-")):
        selected = [r for r in rows if r["policy"] == policy]
        for batch in sorted({r["batch_size"] for r in selected}):
            x = sorted((r for r in selected if r["batch_size"] == batch), key=lambda r: r["context_length"])
            plt.plot([r["context_length"] for r in x], [r["attention_latency_share"] for r in x], marker="o", linestyle=style, label=f"{policy}, B={batch}")
    plt.title("Real-router simulation | 48 layers × 8 decode steps"); plt.xlabel("Context length"); plt.ylabel("Attention latency share"); plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(out / "context_attention_share.png", dpi=180); plt.close()
    # Required Batch x Context peak-memory heatmap uses the offline oracle.
    plt.figure(figsize=(7.2, 4.4)); batches = sorted({r["batch_size"] for r in oracle}); contexts = sorted({r["context_length"] for r in oracle})
    import math
    matrix = [[next((r["peak_memory_bytes"] / 1e9 for r in oracle if r["batch_size"] == b and r["context_length"] == c), math.nan) for c in contexts] for b in batches]
    image = plt.imshow(matrix, aspect="auto", origin="lower", cmap="viridis"); plt.title("Real-router simulation | Peak memory byte estimate"); plt.xticks(range(len(contexts)), contexts); plt.yticks(range(len(batches)), [f"B={b}" for b in batches]); plt.xlabel("Context length"); plt.ylabel("Batch size"); plt.colorbar(image, label="Peak memory (GB)")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            plt.text(j, i, "--" if math.isnan(value) else f"{value:.1f}", ha="center", va="center", color="white" if not math.isnan(value) and value > 50 else "black")
    plt.tight_layout(); plt.savefig(out / "batch_context_peak_memory_heatmap.png", dpi=180); plt.close()


def summarize(root: Path, output: Path) -> dict[str, Any]:
    trace_meta = {}
    configs = {}
    for case in CASES:
        path, cfg = config_case(root, case); configs[case] = str(path.relative_to(root)); trace_meta[case] = validate_trace(root, case, cfg, path)
    rows = [summarize_replay(root, case, policy, trace_meta[case]) for case in CASES for policy in POLICIES]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "kv_cycle_v1_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    baseline_backend = "real-pilot cycle-v1 contention"
    baseline = []
    baseline_dir = root / "results" / "full_decode_real_pilot_cycle_v1"
    for policy in POLICIES:
        summary_path = baseline_dir / policy / "summary.json"
        if not summary_path.is_file():
            baseline = []
            break
        summary = read_json(summary_path); events = summary.get("event_duration_us_by_category", {}); critical = summary.get("critical_path_duration_us_by_category", {})
        baseline.append({"policy": policy, "total_latency_us": summary.get("total_latency_us"), "throughput_request_tokens_per_s": summary.get("throughput_request_tokens_per_s"), "attention_latency_us": events.get("attention", 0.0), "expert_latency_us": sum(critical.get(k, 0.0) for k in ("gpu_weight_load", "gpu_expert", "pim_write", "pim_expert", "pim_read")), "peak_memory_bytes": summary.get("memory", {}).get("peak_total_bytes")})
    if not baseline:
        baseline_backend = "unavailable"
    payload = {"method": "real-router-cycle-v1-kv-summary", "result_classification": "真实Router捕获 + Ramulator contention-v1模拟；GPU解析、PIM Attention cycle-v0分段命令、Expert cycle-v1混合请求", "limitations": ["prompt为受控长度构造，不代表自然长文数据集", "A800 80 GB与模拟96 GB仅作字节容量比较", "未建模KV READ request-level、paging、spill或eviction", "sieve-cycle-v1是Expert hot-prefix离线Oracle，不代表全系统最优"], "rows": rows, "configs": configs, "baseline_old_pilot": baseline, "baseline_backend": baseline_backend, "validation": {"cases": list(CASES), "policies": list(POLICIES), "interpolation": "forbidden", "kv_per_token_bytes": KV_PER_TOKEN, "a800_capacity_bytes": A800_CAPACITY, "simulated_capacity_bytes": SIM_CAPACITY}}
    (output / "kv_cycle_v1_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_plots(rows, output)
    write_report(root, payload, output)
    return payload


def write_report(root: Path, payload: dict[str, Any], output: Path) -> None:
    by_case: dict[str, list[dict[str, Any]]] = {case: [r for r in payload["rows"] if r["case"] == case] for case in CASES}
    lines = ["# 长上下文 KV Cache cycle-v1 回放分析", "", "本报告由汇总脚本在五组真实 Router Trace 与七策略回放全部通过哈希、层步覆盖和关键路径一致性校验后生成。每个总时延覆盖 8 个 Decode step × 48 层。", "", "## 方法与证据边界", "", "Router 数据来自 A800 上的真实 Qwen3 捕获；prompt 为受控长度构造，不能代表自然长文数据集。Attention 与 Expert 时延来自 Ramulator 模拟：GPU 侧为解析模型，PIM Attention 为 cycle-v0 分段命令时序（表内 milestone 之和），Expert 为 cycle-v1 contention 表。当前没有 KV READ request-level、paging、spill 或 eviction。", "", "容量比较使用 96 GB 模拟容量与 80 GB A800 名义物理显存（80,000,000,000 bytes），均为字节可行性比较，不代表真实驻留或 A800 wall-clock 性能。", "", "## 结果（模拟时延）", "", "表中“其它关键路径”是总关键路径扣除 Attention、Expert 相关命令和 Scheduler 后的剩余类别，保留并发关键路径语义。", "", "| 配置 | 策略 | KV cache (GB) | Attention (us) | Expert critical path (us) | Scheduler (us) | 其它关键路径 (us) | 总时延 (us) | Attention 占比 | 峰值内存 (GB) | 96 GB | 80 GB |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for case in CASES:
        for r in by_case[case]:
            lines.append(f"| {case} | {r['policy']} | {r['kv_cache_bytes']/1e9:.6f} | {r['attention_latency_us']:.6f} | {r['expert_latency_us']:.6f} | {r['critical_path_scheduler_us']:.6f} | {r['critical_path_other_us']:.6f} | {r['total_latency_us']:.6f} | {r['attention_latency_share']:.6f} | {r['peak_memory_bytes']/1e9:.6f} | {r['simulated_capacity_state']} | {r['a800_capacity_state']} |")
    lines += ["", "## 图表", "", f"四幅图位于 `{output.relative_to(root)}`：KV cache 曲线、GPU-only 与 sieve-cycle-v1 Attention 时延对比、对应 Attention 占比对比，以及 Oracle 的 Batch×Context 峰值内存热力图。", "", "## 旧 pilot 参照", "", "旧 pilot 的七策略 cycle-v1 contention summary 仅作为历史参照，未在本汇总器中重跑；它与长上下文配置的 Router、Context 和输入哈希不同，不能直接当作同一 workload 的横向性能结论。"]
    baseline = payload.get("baseline_old_pilot", [])
    if baseline:
        lines += ["", f"基线后端：{payload.get('baseline_backend', 'unknown')}。", "", "| 旧 pilot 策略 | 总时延 (us) |", "|---|---:|"]
        lines.extend(f"| {r.get('policy', '')} | {r.get('total_latency_us', '')} |" for r in baseline)
    oracle = [r for r in payload["rows"] if r["policy"] == "sieve-cycle-v1"]
    if oracle:
        max_share = max(oracle, key=lambda r: r["attention_latency_share"])
        lines += ["", "## 可支持的结论", "", f"在本五组配置和当前模拟容量模型内，Oracle 的最大 Attention 关键路径占比为 {max_share['attention_latency_share']:.4f}（{max_share['case']}）。这些数字只说明当前观察范围内的解析 GPU Attention 与 PIM 分段命令时序如何随 Context 变化。`sieve-cycle-v1` 是 Expert hot-prefix 的离线 Oracle；Attention 固定走 PIM 路径，因此不能据此宣称它是全系统最优策略。容量状态仅是 96 GB/80 GB 字节比较；没有 spill 或 OOM 性能模型。"]
        b8 = sorted((r for r in oracle if r["batch_size"] == 8), key=lambda r: r["context_length"])
        first, last = b8[0], b8[-1]
        lines += ["", f"在 B8 的独立真实路由回放中，Context 从 {first['context_length']} 增至 {last['context_length']}，峰值 KV 从 {first['kv_cache_bytes']/1e9:.4f} GB 增至 {last['kv_cache_bytes']/1e9:.4f} GB；Oracle Attention 占比从 {first['attention_latency_share']:.2%} 变为 {last['attention_latency_share']:.2%}，8-step 模拟总时延从 {first['total_latency_us']/1000:.4f} ms 变为 {last['total_latency_us']/1000:.4f} ms。这里每组使用独立捕获的路由，Expert 负载也会变化，不能将总时延差全部归因于 Context。保持路由模板不变的解析对照单独保存在 `results/kv_capacity_sweep_v1/`。"]
        peak = max(r["peak_memory_bytes"] for r in oracle)
        if peak <= A800_CAPACITY:
            lines += ["", f"五组配置的估算峰值内存最高为 {peak/1e9:.4f} GB，均低于 80 GB 和 96 GB，因此本轮没有触达估算容量上限。解析扫描中的超容量点仍只能标为 infeasible，不能作为真实 OOM、spill 或 paging 性能证据。"]
        lines += ["", "后续若要确认 KV 带宽瓶颈，需要把 KV READ 与 Expert READ 放入同一 HBM 请求队列，保留当前模型作为无 KV 请求竞争对照。若研究容量越界，则需另行实现并验证 resident/spill/oom 状态。本轮的 Attention 占比提供建模动机，尚未证明真实 KV 带宽竞争。", "", "`pimoe` 仍是静态 token 阈值占位策略；本轮没有重跑 runtime-v1/v2，也不构成其跨 workload 泛化验证。"]
    (root / "docs" / "kv_long_context_cycle_v1_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1]); p.add_argument("--output", type=Path, default=None)
    args = p.parse_args(argv); root = args.root.resolve(); output = args.output or root / "results" / "full_decode_real_kv_cycle_v1" / "summary"
    payload = summarize(root, output.resolve()); print(json.dumps({"output": str(output.resolve()), "rows": len(payload["rows"])}, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
