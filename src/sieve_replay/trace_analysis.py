from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Hashable

from .config import ModelConfig
from .trace import RouterTraceRecord, TraceBatch, TraceSet, sha256_file


@dataclass(frozen=True)
class RouterTraceAnalysis:
    summary: dict[str, Any]
    batches: tuple[dict[str, Any], ...]
    expert_loads: tuple[dict[str, Any], ...]
    transitions: tuple[dict[str, Any], ...]


def _round(value: float) -> float:
    return round(value, 9)


def _coefficient_of_variation(values: list[int]) -> float:
    average = mean(values)
    if average == 0:
        return 0.0
    variance = sum((value - average) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / average


def _gini(values: list[int]) -> float:
    ordered = sorted(values)
    total = sum(ordered)
    if total == 0:
        return 0.0
    weighted = sum(index * value for index, value in enumerate(ordered, start=1))
    count = len(ordered)
    return (2 * weighted) / (count * total) - (count + 1) / count


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _request_routes(batch: TraceBatch) -> dict[Hashable, set[int]]:
    return {record.token_id: set(record.expert_ids) for record in batch.records}


def _load_counts(batch: TraceBatch, num_experts: int) -> list[int]:
    counts = [0] * num_experts
    for load in batch.expert_loads:
        counts[load.expert_id] = load.token_count
    return counts


def _hot_order(batch: TraceBatch) -> tuple[int, ...]:
    return tuple(
        load.expert_id
        for load in sorted(
            batch.expert_loads,
            key=lambda load: (-load.token_count, load.expert_id),
        )
    )


def _weight_sums(records: tuple[RouterTraceRecord, ...]) -> Counter[int]:
    result: Counter[int] = Counter()
    for record in records:
        for expert, weight in zip(record.expert_ids, record.expert_weights):
            result[expert] += weight
    return result


def analyze_router_trace(
    trace_set: TraceSet,
    model: ModelConfig,
    trace_path: str | Path,
    trace_kind: str,
    manifest_path: str | Path | None = None,
    hot_expert_count: int = 8,
    trace_label: str | None = None,
) -> RouterTraceAnalysis:
    if hot_expert_count <= 0:
        raise ValueError("hot_expert_count must be positive")
    if not trace_set.batches:
        raise ValueError("router trace analysis requires at least one batch")

    batch_rows: list[dict[str, Any]] = []
    expert_rows: list[dict[str, Any]] = []
    signatures: Counter[tuple[int, ...]] = Counter()
    batches_by_layer: dict[int, list[TraceBatch]] = {}

    for batch in trace_set.batches:
        counts = _load_counts(batch, model.num_experts)
        active_counts = [value for value in counts if value]
        hot_order = _hot_order(batch)
        selected_hot = hot_order[:hot_expert_count]
        total = batch.total_expert_assignments
        signature = tuple(sorted(active_counts, reverse=True))
        signatures[signature] += 1
        weights = _weight_sums(batch.records)
        hot_assignments = sum(counts[expert] for expert in selected_hot)
        row = {
            "step": batch.step,
            "layer": batch.layer,
            "batch_size": batch.batch_size,
            "context_min": min(batch.context_lengths),
            "context_max": max(batch.context_lengths),
            "total_assignments": total,
            "active_experts": len(active_counts),
            "inactive_experts": model.num_experts - len(active_counts),
            "max_expert_load": max(active_counts),
            "mean_active_expert_load": _round(mean(active_counts)),
            "cv_all_experts": _round(_coefficient_of_variation(counts)),
            "cv_active_experts": _round(_coefficient_of_variation(active_counts)),
            "gini_all_experts": _round(_gini(counts)),
            "hot_expert_count": len(selected_hot),
            "hot_experts": ",".join(str(expert) for expert in selected_hot),
            "hot_assignment_share": _round(hot_assignments / total),
            "load_signature": ",".join(str(value) for value in signature),
        }
        batch_rows.append(row)
        hot_rank = {expert: index + 1 for index, expert in enumerate(selected_hot)}
        for expert in hot_order:
            expert_rows.append(
                {
                    "step": batch.step,
                    "layer": batch.layer,
                    "expert": expert,
                    "assignment_count": counts[expert],
                    "normalized_assignment_share": _round(counts[expert] / total),
                    "routing_weight_sum": _round(float(weights[expert])),
                    "hot_rank": hot_rank.get(expert, 0),
                }
            )
        batches_by_layer.setdefault(batch.layer, []).append(batch)

    transition_rows: list[dict[str, Any]] = []
    for layer in sorted(batches_by_layer):
        ordered = sorted(batches_by_layer[layer], key=lambda batch: batch.step)
        for previous, current in zip(ordered, ordered[1:]):
            previous_counts = _load_counts(previous, model.num_experts)
            current_counts = _load_counts(current, model.num_experts)
            previous_active = {index for index, value in enumerate(previous_counts) if value}
            current_active = {index for index, value in enumerate(current_counts) if value}
            previous_hot = set(_hot_order(previous)[:hot_expert_count])
            current_hot = set(_hot_order(current)[:hot_expert_count])
            previous_routes = _request_routes(previous)
            current_routes = _request_routes(current)
            request_ids = tuple(previous_routes)
            route_jaccards = [
                _jaccard(previous_routes[token], current_routes[token])
                for token in request_ids
            ]
            changed = sum(
                previous_routes[token] != current_routes[token] for token in request_ids
            )
            load_denominator = sum(previous_counts) + sum(current_counts)
            load_churn = (
                sum(
                    abs(previous_value - current_value)
                    for previous_value, current_value in zip(
                        previous_counts, current_counts
                    )
                )
                / load_denominator
            )
            transition_rows.append(
                {
                    "layer": layer,
                    "previous_step": previous.step,
                    "current_step": current.step,
                    "step_delta": current.step - previous.step,
                    "active_expert_jaccard": _round(
                        _jaccard(previous_active, current_active)
                    ),
                    "hot_expert_jaccard": _round(_jaccard(previous_hot, current_hot)),
                    "load_churn": _round(load_churn),
                    "mean_request_route_jaccard": _round(mean(route_jaccards)),
                    "changed_request_fraction": _round(changed / len(request_ids)),
                    "same_load_signature": sorted(
                        (value for value in previous_counts if value), reverse=True
                    )
                    == sorted((value for value in current_counts if value), reverse=True),
                }
            )

    dominant_signature_count = max(signatures.values())
    summary = {
        "schema_version": 1,
        "trace_kind": trace_kind,
        "trace_file": trace_label or Path(trace_path).as_posix(),
        "trace_sha256": sha256_file(trace_path),
        "trace_manifest_sha256": (
            sha256_file(manifest_path) if manifest_path is not None else None
        ),
        "model": {
            "name": model.name,
            "num_hidden_layers": model.num_hidden_layers,
            "num_experts": model.num_experts,
            "num_experts_per_tok": model.num_experts_per_tok,
        },
        "selection": {
            "steps": list(trace_set.steps),
            "layers": list(trace_set.layers),
            "batch_size": trace_set.batch_size,
            "layer_batches": len(trace_set.batches),
            "records": sum(batch.batch_size for batch in trace_set.batches),
            "total_assignments": sum(
                batch.total_expert_assignments for batch in trace_set.batches
            ),
            "hot_expert_count": hot_expert_count,
        },
        "load_shape": {
            "unique_signatures": len(signatures),
            "dominant_signature_batches": dominant_signature_count,
            "dominant_signature_fraction": _round(
                dominant_signature_count / len(trace_set.batches)
            ),
        },
        "batch_metrics": {
            "mean_active_experts": _round(
                mean(row["active_experts"] for row in batch_rows)
            ),
            "min_active_experts": min(row["active_experts"] for row in batch_rows),
            "max_active_experts": max(row["active_experts"] for row in batch_rows),
            "mean_max_expert_load": _round(
                mean(row["max_expert_load"] for row in batch_rows)
            ),
            "mean_cv_all_experts": _round(
                mean(row["cv_all_experts"] for row in batch_rows)
            ),
            "mean_gini_all_experts": _round(
                mean(row["gini_all_experts"] for row in batch_rows)
            ),
            "mean_hot_assignment_share": _round(
                mean(row["hot_assignment_share"] for row in batch_rows)
            ),
        },
        "transition_metrics": {
            "transitions": len(transition_rows),
            "mean_active_expert_jaccard": _round(
                mean(row["active_expert_jaccard"] for row in transition_rows)
            )
            if transition_rows
            else None,
            "mean_hot_expert_jaccard": _round(
                mean(row["hot_expert_jaccard"] for row in transition_rows)
            )
            if transition_rows
            else None,
            "mean_load_churn": _round(
                mean(row["load_churn"] for row in transition_rows)
            )
            if transition_rows
            else None,
            "mean_request_route_jaccard": _round(
                mean(row["mean_request_route_jaccard"] for row in transition_rows)
            )
            if transition_rows
            else None,
            "mean_changed_request_fraction": _round(
                mean(row["changed_request_fraction"] for row in transition_rows)
            )
            if transition_rows
            else None,
        },
    }
    return RouterTraceAnalysis(
        summary=summary,
        batches=tuple(batch_rows),
        expert_loads=tuple(expert_rows),
        transitions=tuple(transition_rows),
    )


def _write_csv(path: Path, rows: tuple[dict[str, Any], ...]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_router_trace_analysis(
    output_dir: str | Path,
    analysis: RouterTraceAnalysis,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(analysis.summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output / "batches.csv", analysis.batches)
    _write_csv(output / "expert_loads.csv", analysis.expert_loads)
    _write_csv(output / "transitions.csv", analysis.transitions)
