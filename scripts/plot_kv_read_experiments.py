#!/usr/bin/env python3
"""Plot the completed representative KV READ microbenchmark.

The script is fail-closed: it requires the exact 19-shape report, verifies all
plan/result/frozen-baseline hashes, and plots only the three B8 representative
layer-batches.  It never runs Ramulator and never produces end-to-end
throughput figures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from sieve_replay.report.writer import sha256_file
from sieve_replay.ramulator.kv_read_workload import kv_read_cache_context

ALL_CASES = ("b8_c4k", "b8_c8k", "b8_c16k", "b16_c8k")
CASES = ("b8_c4k", "b8_c8k", "b8_c16k")
PLACEMENTS = ("gpu-only", "frozen-oracle", "fixed-half-prefix")
OUTPUT = ROOT / "results/kv_read_experiment_v1"
DEFAULT_PLOTS = OUTPUT / "plots"
FROZEN = OUTPUT / "frozen_baseline.json"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _plot_dependencies() -> Any:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("install matplotlib from requirements/kv_analysis.txt before plotting") from exc
    return plt


def _verify_frozen_baseline(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    frozen = read_json(path)
    for relative, expected in frozen.get("files", {}).items():
        source = ROOT / relative
        if not source.is_file() or sha256_file(source) != expected:
            raise ValueError(f"frozen baseline changed: {relative}")
    return sha256_file(path)


def _load_verified(output: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    plan_path = output / "kv_read_plan.json"
    exact_path = output / "kv_read_exact_results.json"
    comparison_path = output / "kv_read_comparison.json"
    validation_path = output / "validation.json"
    for path in (plan_path, exact_path, comparison_path, validation_path):
        if not path.is_file():
            raise FileNotFoundError(f"complete KV READ result is missing: {path}")
    plan = read_json(plan_path)
    exact = read_json(exact_path)
    comparison = read_json(comparison_path)
    validation = read_json(validation_path)
    if comparison.get("plan_sha256") != sha256_file(plan_path):
        raise ValueError("comparison plan hash does not match kv_read_plan.json")
    if comparison.get("exact_results_sha256") != sha256_file(exact_path):
        raise ValueError("comparison result hash does not match kv_read_exact_results.json")
    frozen_hash = _verify_frozen_baseline(output / "frozen_baseline.json")
    if comparison.get("frozen_baseline_sha256") != frozen_hash:
        raise ValueError("comparison frozen-baseline hash mismatch")
    if (validation.get("exact_results_sha256") != sha256_file(exact_path)
            or validation.get("comparison_sha256") != sha256_file(comparison_path)
            or validation.get("frozen_baseline_sha256") != frozen_hash):
        raise ValueError("independent exact-result validation is stale or mismatched")
    summary = exact.get("summary", {})
    if (summary.get("required_shapes") != 19 or summary.get("completed") != 19
            or summary.get("failed") != 0 or summary.get("remaining") != 0
            or summary.get("interpolated_workloads") != 0):
        raise ValueError("exact KV READ report is incomplete")
    if plan.get("summary", {}).get("required_shapes") != 19:
        raise ValueError("unexpected KV READ plan shape count")
    if plan.get("summary", {}).get("placements") != list(PLACEMENTS):
        raise ValueError("placement set differs from the planned representative matrix")
    if exact.get("cache_context", {}).get("plan_sha256") != sha256_file(plan_path):
        raise ValueError("exact cache context is not bound to the current plan")
    simulation_context = plan.get("simulation_context", {})
    if any(exact.get("cache_context", {}).get(key) != value for key, value in simulation_context.items()):
        raise ValueError("exact cache context differs from the planned simulation context")
    if exact.get("program_snapshot") != plan.get("program_snapshot"):
        raise ValueError("exact result program snapshot differs from the plan")
    program = plan.get("program_snapshot", {})
    for key, relative in (("runner_sha256", "scripts/run_kv_read_experiments.py"),
                          ("request_model_sha256", "src/sieve_replay/ramulator/kv_read_workload.py"),
                          ("frontend_sha256", "ramulator/extensions/sieve_kv_read/source/sieve_kv_read_frontend.cpp")):
        if program.get(key) != sha256_file(ROOT / relative):
            raise ValueError(f"program snapshot differs from current source: {relative}")
    current_context = kv_read_cache_context(ROOT, ROOT / "configs/ramulator/sieve_hbm3e_cycle_v1.json")
    if any(exact.get("cache_context", {}).get(key) != value for key, value in current_context.items()):
        raise ValueError("KV request-model provenance differs from current source files")
    rows = comparison.get("rows", [])
    if len(rows) != len(ALL_CASES) * len(PLACEMENTS):
        raise ValueError("comparison must contain exactly 12 representative rows")
    _validate_matrix(rows)
    entries = exact.get("entries", {})
    for row in rows:
        if row.get("case") not in ALL_CASES or row.get("placement") not in PLACEMENTS:
            raise ValueError("comparison contains an unknown case or placement row")
        for key in ("expert_shape_key", "kv_shape_key", "combined_shape_key"):
            if row.get(key) not in entries:
                raise ValueError(f"comparison row references missing exact shape: {key}")
    return plan, exact, comparison, frozen_hash


def _contexts(plan: dict[str, Any]) -> dict[tuple[str, str], int]:
    contexts: dict[tuple[str, str], int] = {}
    for row in plan.get("consumers", []):
        key = (row.get("case"), row.get("placement"))
        if row.get("mode") != "combined" or key[0] not in CASES:
            continue
        values = row.get("context_lengths")
        if not isinstance(values, list) or not values or len(set(values)) != 1:
            raise ValueError(f"{key}: expected one uniform context signature")
        context = int(values[0])
        if key in contexts and contexts[key] != context:
            raise ValueError(f"{key}: inconsistent context signature")
        contexts[key] = context
    expected = {(case, placement) for case in CASES for placement in PLACEMENTS}
    if set(contexts) != expected:
        raise ValueError("plan does not cover all B8 representative contexts")
    return contexts


def _validate_matrix(rows: list[dict[str, Any]], cases: tuple[str, ...] = ALL_CASES) -> None:
    expected = {(case, placement) for case in cases for placement in PLACEMENTS}
    actual = {(row.get("case"), row.get("placement")) for row in rows}
    if len(rows) != len(expected) or actual != expected:
        raise ValueError("comparison rows do not uniquely cover the representative matrix")


def _save_pair(plt: Any, figure: Any, path: Path) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    svg = path.with_suffix(".svg")
    figure.savefig(svg, bbox_inches="tight")
    plt.close(figure)
    return {"png": sha256_file(path), "svg": sha256_file(svg)}


def _series(rows: list[dict[str, Any]], contexts: dict[tuple[str, str], int], field: str, placement: str) -> tuple[list[int], list[float]]:
    selected = [row for row in rows if row["placement"] == placement]
    selected.sort(key=lambda row: contexts[(row["case"], placement)])
    return ([contexts[(row["case"], placement)] for row in selected], [float(row[field]) for row in selected])


def plot(output: Path = OUTPUT, plot_dir: Path = DEFAULT_PLOTS) -> dict[str, Any]:
    plt = _plot_dependencies()
    plan, exact, comparison, frozen_hash = _load_verified(output)
    rows = [row for row in comparison["rows"] if row["case"] in CASES]
    contexts = _contexts(plan)
    plot_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, dict[str, str]] = {}

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for placement in PLACEMENTS:
        x, isolated = _series(rows, contexts, "kv_only_us", placement)
        _, combined = _series(rows, contexts, "kv_combined_completion_us", placement)
        ax.plot(x, isolated, marker="o", label=f"{placement}: KV-only")
        ax.plot(x, combined, marker="s", linestyle="--", label=f"{placement}: combined")
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("KV completion (us)")
    ax.set_title("B8 representative layer-batch: KV completion")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    figures["b8_context_kv_completion"] = _save_pair(plt, fig, plot_dir / "b8_context_kv_completion.png")

    # Expert completion is measured on the GPU/PIM Expert streams only.  The
    # KV stream's completion milestone is intentionally excluded here.
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for placement in PLACEMENTS:
        selected = [row for row in rows if row["placement"] == placement]
        selected.sort(key=lambda row: contexts[(row["case"], placement)])
        x = [contexts[(row["case"], placement)] for row in selected]
        y = [max(float(row["gpu_combined_completion_us"]), float(row["pim_combined_completion_us"])) - float(row["expert_only_us"]) for row in selected]
        ax.plot(x, y, marker="o", label=placement)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Expert completion slowdown (us)")
    ax.set_title("B8 representative layer-batch: Expert slowdown under KV injection")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    figures["b8_context_expert_slowdown"] = _save_pair(plt, fig, plot_dir / "b8_context_expert_slowdown.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for placement in PLACEMENTS:
        x, wait = _series(rows, contexts, "kv_accepted_to_column_issue_mean_us", placement)
        ax.plot(x, wait, marker="o", label=placement)
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("KV accepted-to-column issue mean (us)")
    ax.set_title("B8 representative layer-batch: KV request residence before column issue")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    figures["b8_context_kv_wait"] = _save_pair(plt, fig, plot_dir / "b8_context_kv_wait.png")

    manifest = {
        "schema_version": 1,
        "method": "kv-read-representative-plots-v1",
        "classification": "Ramulator request-level microbenchmark plots; no end-to-end throughput",
        "scope": "B8 cases only, step 0, layer 0 representative; 4 cases remain in exact report but are not plotted here",
        "input_hashes": {
            "plan_sha256": sha256_file(output / "kv_read_plan.json"),
            "exact_results_sha256": sha256_file(output / "kv_read_exact_results.json"),
            "comparison_sha256": sha256_file(output / "kv_read_comparison.json"),
            "frozen_baseline_sha256": frozen_hash,
            "plotter_sha256": sha256_file(Path(__file__)),
        },
        "figures": figures,
        "limitations": [
            "KV READ and Expert requests are injected simultaneously as a controlled stress microbenchmark",
            "Current decode Attention -> Router -> Expert remains serial; these curves are not E2E latency",
            "Accepted-to-column issue includes arbitration and ACT/PRE preparation, not pure FIFO wait",
            "No paging, spill, eviction, hardware KV trace, or Shared CXL-PIM",
        ],
    }
    (plot_dir / "plots_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOTS)
    args = parser.parse_args(argv)
    manifest = plot(args.output, args.plot_dir)
    print(json.dumps({"figures": sorted(manifest["figures"]), "plot_dir": str(args.plot_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
