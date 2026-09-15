#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sieve_replay.config import load_configuration  # noqa: E402
from sieve_replay.policy import create_policy  # noqa: E402
from sieve_replay.report.writer import sha256_file  # noqa: E402
from sieve_replay.timing import (  # noqa: E402
    RamulatorContentionTable,
    RamulatorContentionTimingModel,
    RamulatorTimingTable,
    RuntimeExpertTimingModel,
)
from sieve_replay.trace import load_trace_set  # noqa: E402
from sieve_replay.trace_manifest import TraceManifest  # noqa: E402


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _error_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    holdout = [row for row in rows if not row["is_anchor"]]
    if not holdout:
        raise ValueError("runtime prediction evaluation requires non-anchor counts")
    absolute = [float(row["absolute_error_us"]) for row in holdout]
    percentages = [
        float(row["percentage_error"])
        for row in holdout
        if float(row["actual_us"]) > 0
    ]
    return {
        "non_anchor_counts": len(holdout),
        "mae_us": sum(absolute) / len(absolute),
        "p95_absolute_error_us": _percentile(absolute, 0.95),
        "max_absolute_error_us": max(absolute),
        "mape_percent": sum(percentages) / len(percentages),
        "p95_percentage_error": _percentile(percentages, 0.95),
        "max_percentage_error": max(percentages),
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _record_exact(
    values: dict[int, float], count: int, duration: float | None, label: str
) -> None:
    if duration is None:
        raise ValueError(f"{label} isolated timing is unavailable for count={count}")
    previous = values.setdefault(count, duration)
    if abs(previous - duration) > 1e-12:
        raise ValueError(f"{label} isolated timing is inconsistent for count={count}")


def _load_result(
    results: Path,
    policy: str,
    expected_hashes: dict[str, str],
    expected_layer_batches: int,
    expected_backend: str,
) -> dict[str, Any]:
    path = results / policy / "summary.json"
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing replay result for {policy}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid replay result JSON for {policy}: {exc}") from exc
    if not isinstance(summary, dict) or summary.get("policy") != policy:
        raise ValueError(f"replay result policy mismatch for {policy}: {path}")
    if summary.get("timing_backend") != expected_backend:
        raise ValueError(f"replay result timing backend mismatch for {policy}: {path}")
    if summary.get("input_hashes") != expected_hashes:
        raise ValueError(f"replay result input hashes mismatch for {policy}: {path}")
    scope = summary.get("scope")
    if not isinstance(scope, dict) or scope.get("layer_executions") != expected_layer_batches:
        raise ValueError(f"replay result layer count mismatch for {policy}: {path}")
    total = summary.get("total_latency_us")
    if isinstance(total, bool) or not isinstance(total, (int, float)) or total <= 0:
        raise ValueError(f"replay result total latency is invalid for {policy}: {path}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate sparse runtime scheduling against the exact cycle-v1 Oracle"
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    configuration = load_configuration(args.experiment)
    experiment = configuration.experiment
    calibration_path = experiment.runtime_calibration_path
    isolated_path = experiment.pim_timing_table_path
    contention_path = experiment.contention_timing_table_path
    cycle_path = experiment.ramulator_cycle_config_path
    if None in (calibration_path, isolated_path, contention_path, cycle_path):
        raise ValueError("runtime evaluation requires all timing and calibration inputs")
    assert calibration_path is not None
    assert isolated_path is not None
    assert contention_path is not None
    assert cycle_path is not None

    trace_set = load_trace_set(
        experiment.trace_path,
        configuration.model,
        experiment.layers,
        experiment.steps,
    )
    if experiment.trace_manifest_path is not None:
        TraceManifest.load_and_validate(
            experiment.trace_manifest_path,
            experiment.trace_path,
            configuration.model,
            trace_set,
        )
    contention = RamulatorContentionTable.load(contention_path)
    contention.validate_inputs(
        isolated_path,
        experiment.trace_path,
        cycle_path,
        experiment.model_path,
        experiment.hardware_path,
    )
    calibration = RuntimeExpertTimingModel.load(calibration_path)
    calibration.validate_inputs(
        experiment.model_path,
        experiment.hardware_path,
        cycle_path,
    )
    timing = RamulatorContentionTimingModel(
        configuration.model,
        configuration.hardware,
        RamulatorTimingTable.load(isolated_path),
        contention,
        calibration,
    )
    runtime_policy = create_policy("sieve-runtime-v1", timing)
    oracle_policy = create_policy("sieve-cycle-v1", timing)

    gpu_exact: dict[int, float] = {}
    pim_exact: dict[int, float] = {}
    layer_rows: list[dict[str, Any]] = []
    for trace in trace_set.batches:
        for row in contention.candidate_timings(trace.expert_loads):
            _record_exact(
                gpu_exact,
                len(row.gpu_experts),
                row.gpu_isolated_us,
                "GPU",
            )
            _record_exact(
                pim_exact,
                row.pim_token_count,
                row.pim_isolated_us,
                "PIM",
            )

        runtime = runtime_policy.place(trace)
        oracle = oracle_policy.place(trace)
        by_id = {load.expert_id: load for load in trace.expert_loads}
        gpu_loads = tuple(by_id[expert] for expert in runtime.gpu_experts)
        pim_loads = tuple(by_id[expert] for expert in runtime.pim_experts)
        actual = contention.expert_timing(gpu_loads, pim_loads)
        gpu_compute_us = timing.gpu_expert_compute(gpu_loads).duration_us
        scheduler_us = timing.scheduler(runtime.policy).duration_us
        selected_actual_us = scheduler_us + max(
            actual.gpu_contended_us + gpu_compute_us,
            actual.pim_contended_us,
        )
        regret_us = max(0.0, selected_actual_us - oracle.estimated_objective_us)
        runtime_prefix = len(runtime.gpu_experts)
        oracle_prefix = len(oracle.gpu_experts)
        prediction_extrapolated = (
            calibration.gpu_prediction_is_extrapolated(runtime_prefix)
            or calibration.pim_prediction_is_extrapolated(
                sum(load.token_count for load in pim_loads)
            )
        )
        layer_rows.append(
            {
                "step": trace.step,
                "layer": trace.layer,
                "active_experts": len(trace.expert_loads),
                "runtime_gpu_prefix": runtime_prefix,
                "oracle_gpu_prefix": oracle_prefix,
                "prefix_match": runtime_prefix == oracle_prefix,
                "prefix_delta": runtime_prefix - oracle_prefix,
                "prediction_extrapolated": prediction_extrapolated,
                "runtime_predicted_objective_us": runtime.estimated_objective_us,
                "runtime_selected_actual_objective_us": selected_actual_us,
                "oracle_objective_us": oracle.estimated_objective_us,
                "regret_us": regret_us,
                "regret_percent": (
                    100.0 * regret_us / oracle.estimated_objective_us
                ),
            }
        )

    gpu_anchor_counts = {anchor.count for anchor in calibration.gpu_anchors}
    pim_anchor_counts = {anchor.count for anchor in calibration.pim_anchors}
    prediction_rows: list[dict[str, Any]] = []
    for count, actual in sorted(gpu_exact.items()):
        predicted = calibration.gpu_expert_read_us(count)
        prediction_rows.append(
            {
                "path": "gpu_expert_read",
                "count": count,
                "is_anchor": count in gpu_anchor_counts,
                "predicted_us": predicted,
                "actual_us": actual,
                "absolute_error_us": abs(predicted - actual),
                "percentage_error": 100.0 * abs(predicted - actual) / actual if actual else 0.0,
            }
        )
    for count, actual in sorted(pim_exact.items()):
        predicted = calibration.pim_expert_pipeline_us(count)
        prediction_rows.append(
            {
                "path": "pim_expert_pipeline",
                "count": count,
                "is_anchor": count in pim_anchor_counts,
                "predicted_us": predicted,
                "actual_us": actual,
                "absolute_error_us": abs(predicted - actual),
                "percentage_error": 100.0 * abs(predicted - actual) / actual if actual else 0.0,
            }
        )

    regrets = [float(row["regret_us"]) for row in layer_rows]
    regret_percentages = [float(row["regret_percent"]) for row in layer_rows]
    matches = sum(bool(row["prefix_match"]) for row in layer_rows)
    extrapolated = sum(bool(row["prediction_extrapolated"]) for row in layer_rows)
    results = Path(args.results)
    expected_hashes = {
        "model_sha256": sha256_file(experiment.model_path),
        "hardware_sha256": sha256_file(experiment.hardware_path),
        "trace_sha256": sha256_file(experiment.trace_path),
        "pim_timing_table_sha256": sha256_file(isolated_path),
        "contention_timing_table_sha256": sha256_file(contention_path),
        "ramulator_cycle_config_sha256": sha256_file(cycle_path),
        "runtime_calibration_sha256": sha256_file(calibration_path),
    }
    if experiment.trace_manifest_path is not None:
        expected_hashes["trace_manifest_sha256"] = sha256_file(
            experiment.trace_manifest_path
        )
    result_summaries = {
        policy: _load_result(
            results,
            policy,
            expected_hashes,
            len(layer_rows),
            configuration.hardware.timing_backend,
        )
        for policy in (
            "sieve-runtime-v1",
            "sieve-cycle-v1",
            "sieve-fixed-16-cycle-v1",
            "sieve",
        )
    }
    runtime_total = float(result_summaries["sieve-runtime-v1"]["total_latency_us"])
    oracle_total = float(result_summaries["sieve-cycle-v1"]["total_latency_us"])
    fixed_total = float(
        result_summaries["sieve-fixed-16-cycle-v1"]["total_latency_us"]
    )
    legacy_total = float(result_summaries["sieve"]["total_latency_us"])
    summary = {
        "schema_version": 1,
        "evaluation_scope": "within-pilot non-anchor count interpolation",
        "runtime_policy": "sieve-runtime-v1",
        "oracle_policy": "sieve-cycle-v1",
        "calibration": calibration.report(),
        "decision_metrics": {
            "layer_batches": len(layer_rows),
            "prefix_matches": matches,
            "prefix_match_rate": matches / len(layer_rows),
            "prediction_extrapolated_layer_batches": extrapolated,
            "mean_regret_us": sum(regrets) / len(regrets),
            "p95_regret_us": _percentile(regrets, 0.95),
            "max_regret_us": max(regrets),
            "mean_regret_percent": sum(regret_percentages) / len(regret_percentages),
            "p95_regret_percent": _percentile(regret_percentages, 0.95),
            "max_regret_percent": max(regret_percentages),
        },
        "prediction_metrics": {
            "gpu_expert_read": _error_metrics(
                [row for row in prediction_rows if row["path"] == "gpu_expert_read"]
            ),
            "pim_expert_pipeline": _error_metrics(
                [
                    row
                    for row in prediction_rows
                    if row["path"] == "pim_expert_pipeline"
                ]
            ),
        },
        "end_to_end": {
            "runtime_total_latency_us": runtime_total,
            "oracle_total_latency_us": oracle_total,
            "legacy_sieve_total_latency_us": legacy_total,
            "fixed_16_total_latency_us": fixed_total,
            "runtime_oracle_gap_us": runtime_total - oracle_total,
            "runtime_oracle_gap_percent": 100.0 * (runtime_total - oracle_total) / oracle_total,
            "runtime_vs_legacy_reduction_percent": 100.0 * (legacy_total - runtime_total) / legacy_total,
            "runtime_vs_fixed_16_reduction_percent": 100.0
            * (fixed_total - runtime_total)
            / fixed_total,
        },
        "input_hashes": expected_hashes,
        "result_validation": {
            "policies": sorted(result_summaries),
            "layer_batches_per_policy": len(layer_rows),
            "input_hashes_match": True,
            "timing_backend_match": True,
        },
        "method_boundary": {
            "runtime_decision_reads_exact_candidates": False,
            "exact_table_role": "post-decision simulator ground truth and Oracle only",
            "new_ramulator_runs": 0,
            "generalization_claim": (
                "none beyond non-anchor counts in the same pilot, model, and hardware"
            ),
        },
    }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_rows(output / "layers.csv", layer_rows)
    _write_rows(output / "prediction_errors.csv", prediction_rows)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
