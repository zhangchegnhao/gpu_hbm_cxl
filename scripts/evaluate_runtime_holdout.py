#!/usr/bin/env python3
"""Evaluate runtime-v1 with signature-level cross-validation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
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
    TimingAnchor,
)
from sieve_replay.trace import TraceBatch, load_trace_set  # noqa: E402
from sieve_replay.trace_manifest import TraceManifest  # noqa: E402


DEFAULT_BUDGETS = (5, 9, 15, 25)
GPU_RATIO = 0.5


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _record_exact(
    values: dict[int, float], count: int, duration: float | None, label: str
) -> None:
    if duration is None:
        raise ValueError(f"{label} isolated timing is unavailable for count={count}")
    previous = values.setdefault(count, duration)
    if abs(previous - duration) > 1e-12:
        raise ValueError(f"{label} isolated timing is inconsistent for count={count}")


def _select_counts(available: set[int], budget: int) -> tuple[int, ...]:
    """Select a nested maximin sequence of nonzero counts."""
    if budget < 2:
        raise ValueError("each timing path requires at least two nonzero calibration points")
    nonzero = sorted(count for count in available if count > 0)
    if not nonzero:
        raise ValueError("training fold has no nonzero timing counts")
    target = min(budget, len(nonzero))
    if target == len(nonzero):
        return tuple(nonzero)
    selected = {nonzero[0], nonzero[-1]}
    while len(selected) < target:
        candidate = max(
            (count for count in nonzero if count not in selected),
            key=lambda count: (
                min(abs(count - anchor) for anchor in selected),
                -count,
            ),
        )
        selected.add(candidate)
    return tuple(sorted(selected))


def _budget_split(budget: int) -> tuple[int, int]:
    if budget < 4:
        raise ValueError("total calibration budget must leave at least two points per path")
    gpu_budget = max(2, int(budget * GPU_RATIO))
    pim_budget = max(2, budget - gpu_budget)
    while gpu_budget + pim_budget > budget:
        if gpu_budget > pim_budget and gpu_budget > 2:
            gpu_budget -= 1
        elif pim_budget > 2:
            pim_budget -= 1
        else:
            raise ValueError("cannot split calibration budget")
    while gpu_budget + pim_budget < budget:
        pim_budget += 1
    return gpu_budget, pim_budget


def _signature(loads: tuple[Any, ...]) -> tuple[int, ...]:
    return tuple(sorted((load.token_count for load in loads), reverse=True))


def _signature_folds(traces: tuple[TraceBatch, ...], fold_count: int) -> tuple[tuple[int, ...], ...]:
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, trace in enumerate(traces):
        groups[_signature(trace.expert_loads)].append(index)
    keys = sorted(groups)
    folds: list[list[int]] = [[] for _ in range(fold_count)]
    for key_index, key in enumerate(keys):
        folds[key_index % fold_count].extend(groups[key])
    return tuple(tuple(sorted(indices)) for indices in folds)


def _prediction_metrics(
    rows: list[dict[str, Any]],
    *,
    exclude_anchors: bool,
) -> dict[str, float | int]:
    evaluated = [row for row in rows if not row["is_anchor"]] if exclude_anchors else rows
    if not evaluated:
        raise ValueError("holdout prediction metrics require observations")
    errors = [float(row["absolute_error_us"]) for row in evaluated]
    percentages = [float(row["percentage_error"]) for row in evaluated]
    return {
        "evaluated_counts": len(evaluated),
        "anchors_excluded": exclude_anchors,
        "extrapolated_counts": sum(bool(row["is_extrapolated"]) for row in evaluated),
        "mae_us": sum(errors) / len(errors),
        "p95_absolute_error_us": _percentile(errors, 0.95),
        "max_absolute_error_us": max(errors),
        "mape_percent": sum(percentages) / len(percentages),
        "p95_percentage_error": _percentile(percentages, 0.95),
        "max_percentage_error": max(percentages),
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _calibration_metadata(
    experiment: Any,
    source_table_path: Path,
    fold: int,
    budget: int,
) -> dict[str, str]:
    return {
        "units": "us",
        "method": "piecewise-linear-isolated-v1",
        "source": "exact isolated timings from training signatures only",
        "scope": f"Qwen3 real-pilot signature holdout; fold={fold}; budget={budget}",
        "model_sha256": sha256_file(experiment.model_path),
        "hardware_sha256": sha256_file(experiment.hardware_path),
        "cycle_config_sha256": sha256_file(experiment.ramulator_cycle_config_path),
        "source_contention_table_sha256": sha256_file(source_table_path),
        "generator_sha256": sha256_file(Path(__file__)),
    }


def _write_calibration(
    path: Path,
    metadata: dict[str, str],
    gpu_anchors: tuple[TimingAnchor, ...],
    pim_anchors: tuple[TimingAnchor, ...],
) -> RuntimeExpertTimingModel:
    raw = {
        "schema_version": 1,
        "metadata": metadata,
        "gpu_expert_count_anchors": [
            {"expert_count": anchor.count, "duration_us": anchor.duration_us}
            for anchor in gpu_anchors
        ],
        "pim_token_count_anchors": [
            {"token_count": anchor.count, "duration_us": anchor.duration_us}
            for anchor in pim_anchors
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return RuntimeExpertTimingModel.load(path)


def _make_model(
    experiment: Any,
    source_table_path: Path,
    fold: int,
    budget: int,
    gpu_exact: dict[int, float],
    pim_exact: dict[int, float],
    output: Path,
) -> RuntimeExpertTimingModel:
    gpu_budget, pim_budget = _budget_split(budget)
    gpu_counts = _select_counts(set(gpu_exact), gpu_budget)
    pim_counts = _select_counts(set(pim_exact), pim_budget)
    gpu_anchors = (TimingAnchor(0, 0.0),) + tuple(
        TimingAnchor(count, gpu_exact[count]) for count in gpu_counts
    )
    pim_anchors = (TimingAnchor(0, 0.0),) + tuple(
        TimingAnchor(count, pim_exact[count]) for count in pim_counts
    )
    metadata = _calibration_metadata(experiment, source_table_path, fold, budget)
    model = _write_calibration(output, metadata, gpu_anchors, pim_anchors)
    model.validate_inputs(
        experiment.model_path,
        experiment.hardware_path,
        experiment.ramulator_cycle_config_path,
        source_table_path,
    )
    return model


def _evaluate_fold(
    traces: tuple[TraceBatch, ...],
    holdout_indices: tuple[int, ...],
    candidate_rows: tuple[tuple[Any, ...], ...],
    configuration: Any,
    contention: RamulatorContentionTable,
    calibration: RuntimeExpertTimingModel,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    timing = RamulatorContentionTimingModel(
        configuration.model,
        configuration.hardware,
        RamulatorTimingTable.load(configuration.experiment.pim_timing_table_path),
        contention,
        calibration,
    )
    runtime_policy = create_policy("sieve-runtime-v1", timing)
    oracle_policy = create_policy("sieve-cycle-v1", timing)
    layer_rows: list[dict[str, Any]] = []
    prediction_values: dict[tuple[str, int], float] = {}
    for index in holdout_indices:
        trace = traces[index]
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
        pim_tokens = sum(load.token_count for load in pim_loads)
        runtime_search = runtime.search_report
        if runtime_search is None:
            raise ValueError("runtime holdout decision is missing its search report")
        any_candidate_extrapolated = any(
            bool(
                candidate["gpu_prediction_extrapolated"]
                or candidate["pim_prediction_extrapolated"]
            )
            for candidate in runtime_search["candidates"]
        )
        layer_rows.append(
            {
                "step": trace.step,
                "layer": trace.layer,
                "signature": ",".join(map(str, _signature(trace.expert_loads))),
                "active_experts": len(trace.expert_loads),
                "runtime_gpu_prefix": len(runtime.gpu_experts),
                "oracle_gpu_prefix": len(oracle.gpu_experts),
                "prefix_match": len(runtime.gpu_experts) == len(oracle.gpu_experts),
                "prefix_delta": len(runtime.gpu_experts) - len(oracle.gpu_experts),
                "gpu_prediction_extrapolated": calibration.gpu_prediction_is_extrapolated(
                    len(runtime.gpu_experts)
                ),
                "pim_prediction_extrapolated": calibration.pim_prediction_is_extrapolated(
                    pim_tokens
                ),
                "any_candidate_prediction_extrapolated": any_candidate_extrapolated,
                "runtime_predicted_objective_us": runtime.estimated_objective_us,
                "runtime_selected_actual_objective_us": selected_actual_us,
                "oracle_objective_us": oracle.estimated_objective_us,
                "regret_us": regret_us,
                "regret_percent": 100.0 * regret_us / oracle.estimated_objective_us,
            }
        )
        for row in candidate_rows[index]:
            if row.gpu_isolated_us is not None:
                prediction_values.setdefault(("gpu_expert_read", len(row.gpu_experts)), row.gpu_isolated_us)
            if row.pim_isolated_us is not None:
                prediction_values.setdefault(("pim_expert_pipeline", row.pim_token_count), row.pim_isolated_us)
    gpu_anchor_counts = {anchor.count for anchor in calibration.gpu_anchors}
    pim_anchor_counts = {anchor.count for anchor in calibration.pim_anchors}
    prediction_rows: list[dict[str, Any]] = []
    for (path, count), actual in sorted(prediction_values.items()):
        predicted = (
            calibration.gpu_expert_read_us(count)
            if path == "gpu_expert_read"
            else calibration.pim_expert_pipeline_us(count)
        )
        anchors = gpu_anchor_counts if path == "gpu_expert_read" else pim_anchor_counts
        extrapolated = (
            calibration.gpu_prediction_is_extrapolated(count)
            if path == "gpu_expert_read"
            else calibration.pim_prediction_is_extrapolated(count)
        )
        prediction_rows.append(
            {
                "path": path,
                "count": count,
                "is_anchor": count in anchors,
                "is_extrapolated": extrapolated,
                "predicted_us": predicted,
                "actual_us": actual,
                "absolute_error_us": abs(predicted - actual),
                "percentage_error": 100.0 * abs(predicted - actual) / actual if actual else 0.0,
            }
        )
    return layer_rows, prediction_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    args = parser.parse_args()
    budgets = tuple(sorted({int(value) for value in args.budgets.split(",")}))
    if args.folds < 2:
        raise ValueError("--folds must be at least two")
    if any(budget < 4 for budget in budgets):
        raise ValueError("each budget must be at least four nonzero points")

    configuration = load_configuration(args.experiment)
    experiment = configuration.experiment
    if None in (
        experiment.pim_timing_table_path,
        experiment.contention_timing_table_path,
        experiment.ramulator_cycle_config_path,
    ):
        raise ValueError("holdout evaluation requires isolated, contention, and cycle inputs")
    assert experiment.pim_timing_table_path is not None
    assert experiment.contention_timing_table_path is not None
    assert experiment.ramulator_cycle_config_path is not None
    traces = load_trace_set(
        experiment.trace_path,
        configuration.model,
        experiment.layers,
        experiment.steps,
    ).batches
    if experiment.trace_manifest_path is not None:
        TraceManifest.load_and_validate(
            experiment.trace_manifest_path,
            experiment.trace_path,
            configuration.model,
            load_trace_set(
                experiment.trace_path,
                configuration.model,
                experiment.layers,
                experiment.steps,
            ),
        )
    contention = RamulatorContentionTable.load(experiment.contention_timing_table_path)
    contention.validate_inputs(
        experiment.pim_timing_table_path,
        experiment.trace_path,
        experiment.ramulator_cycle_config_path,
        experiment.model_path,
        experiment.hardware_path,
    )
    candidate_rows = tuple(
        tuple(contention.candidate_timings(trace.expert_loads)) for trace in traces
    )
    folds = _signature_folds(traces, args.folds)
    signature_by_index = {_signature(trace.expert_loads) for trace in traces}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    all_layer_rows: list[dict[str, Any]] = []
    all_prediction_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for budget in budgets:
        for fold, holdout_indices in enumerate(folds):
            training_indices = tuple(
                index for index in range(len(traces)) if index not in set(holdout_indices)
            )
            gpu_exact: dict[int, float] = {}
            pim_exact: dict[int, float] = {}
            for index in training_indices:
                for row in candidate_rows[index]:
                    _record_exact(gpu_exact, len(row.gpu_experts), row.gpu_isolated_us, "GPU")
                    _record_exact(pim_exact, row.pim_token_count, row.pim_isolated_us, "PIM")
            calibration_path = output / "calibrations" / f"fold_{fold:02d}_budget_{budget:02d}.json"
            calibration = _make_model(
                experiment,
                experiment.contention_timing_table_path,
                fold,
                budget,
                gpu_exact,
                pim_exact,
                calibration_path,
            )
            layer_rows, prediction_rows = _evaluate_fold(
                traces,
                holdout_indices,
                candidate_rows,
                configuration,
                contention,
                calibration,
            )
            for row in layer_rows:
                row.update({"budget": budget, "fold": fold})
            for row in prediction_rows:
                row.update({"budget": budget, "fold": fold})
            all_layer_rows.extend(layer_rows)
            all_prediction_rows.extend(prediction_rows)
            regrets = [float(row["regret_us"]) for row in layer_rows]
            regret_percentages = [float(row["regret_percent"]) for row in layer_rows]
            matches = sum(bool(row["prefix_match"]) for row in layer_rows)
            extrapolated = sum(
                bool(row["gpu_prediction_extrapolated"] or row["pim_prediction_extrapolated"])
                for row in layer_rows
            )
            candidate_extrapolated = sum(
                bool(row["any_candidate_prediction_extrapolated"])
                for row in layer_rows
            )
            fold_rows.append(
                {
                    "budget": budget,
                    "fold": fold,
                    "training_signatures": len({_signature(traces[index].expert_loads) for index in training_indices}),
                    "holdout_signatures": len({_signature(traces[index].expert_loads) for index in holdout_indices}),
                    "holdout_layer_batches": len(layer_rows),
                    "nonzero_calibration_points": len(calibration.gpu_anchors) + len(calibration.pim_anchors) - 2,
                    "prefix_matches": matches,
                    "prefix_match_rate": matches / len(layer_rows),
                    "selected_extrapolated_layer_batches": extrapolated,
                    "candidate_extrapolated_layer_batches": candidate_extrapolated,
                    "mean_regret_us": sum(regrets) / len(regrets),
                    "p95_regret_us": _percentile(regrets, 0.95),
                    "max_regret_us": max(regrets),
                    "mean_regret_percent": sum(regret_percentages) / len(regret_percentages),
                    "p95_regret_percent": _percentile(regret_percentages, 0.95),
                    "max_regret_percent": max(regret_percentages),
                    "calibration_path": str(calibration_path),
                    "calibration_sha256": sha256_file(calibration_path),
                }
            )
        fold_budget_rows = [row for row in fold_rows if row["budget"] == budget]
        budget_layer_rows = [row for row in all_layer_rows if row["budget"] == budget]
        budget_prediction_rows = [row for row in all_prediction_rows if row["budget"] == budget]
        regrets = [float(row["regret_us"]) for row in budget_layer_rows]
        regret_percentages = [float(row["regret_percent"]) for row in budget_layer_rows]
        matches = sum(bool(row["prefix_match"]) for row in budget_layer_rows)
        budget_rows.append(
            {
                "budget": budget,
                "folds": args.folds,
                "unique_signatures": len(signature_by_index),
                "holdout_layer_batches": len(budget_layer_rows),
                "prefix_matches": matches,
                "prefix_match_rate": matches / len(budget_layer_rows),
                "selected_extrapolated_layer_batches": sum(
                    int(row["selected_extrapolated_layer_batches"]) for row in fold_budget_rows
                ),
                "candidate_extrapolated_layer_batches": sum(
                    int(row["candidate_extrapolated_layer_batches"]) for row in fold_budget_rows
                ),
                "mean_regret_us": sum(regrets) / len(regrets),
                "p95_regret_us": _percentile(regrets, 0.95),
                "max_regret_us": max(regrets),
                "mean_regret_percent": sum(regret_percentages) / len(regret_percentages),
                "p95_regret_percent": _percentile(regret_percentages, 0.95),
                "max_regret_percent": max(regret_percentages),
                "gpu_prediction_non_anchor": _prediction_metrics(
                    [row for row in budget_prediction_rows if row["path"] == "gpu_expert_read"],
                    exclude_anchors=True,
                ),
                "gpu_prediction_all_holdout_counts": _prediction_metrics(
                    [row for row in budget_prediction_rows if row["path"] == "gpu_expert_read"],
                    exclude_anchors=False,
                ),
                "pim_prediction_non_anchor": _prediction_metrics(
                    [row for row in budget_prediction_rows if row["path"] == "pim_expert_pipeline"],
                    exclude_anchors=True,
                ),
                "pim_prediction_all_holdout_counts": _prediction_metrics(
                    [row for row in budget_prediction_rows if row["path"] == "pim_expert_pipeline"],
                    exclude_anchors=False,
                ),
            }
        )
    _write_rows(output / "fold_metrics.csv", fold_rows)
    _write_rows(output / "layers.csv", all_layer_rows)
    _write_rows(output / "prediction_errors.csv", all_prediction_rows)
    summary = {
        "schema_version": 1,
        "evaluation_scope": "signature-level holdout; calibration uses training signatures only",
        "fold_count": args.folds,
        "budget_nonzero_points": list(budgets),
        "trace_batches": len(traces),
        "unique_signatures": len(signature_by_index),
        "input_hashes": {
            "model_sha256": sha256_file(experiment.model_path),
            "hardware_sha256": sha256_file(experiment.hardware_path),
            "trace_sha256": sha256_file(experiment.trace_path),
            "trace_manifest_sha256": (
                sha256_file(experiment.trace_manifest_path)
                if experiment.trace_manifest_path is not None
                else None
            ),
            "pim_timing_table_sha256": sha256_file(experiment.pim_timing_table_path),
            "contention_timing_table_sha256": sha256_file(experiment.contention_timing_table_path),
            "ramulator_cycle_config_sha256": sha256_file(experiment.ramulator_cycle_config_path),
            "generator_sha256": sha256_file(Path(__file__)),
        },
        "method_boundary": {
            "runtime_decision_reads_exact_candidates": False,
            "exact_table_role": "post-decision ground truth only",
            "new_ramulator_runs": 0,
            "interpolation": "forbidden for exact evaluation table",
            "generalization_claim": "same pilot signatures only; no cross-prompt claim",
        },
        "budgets": budget_rows,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "budgets": list(budgets), "folds": args.folds}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
