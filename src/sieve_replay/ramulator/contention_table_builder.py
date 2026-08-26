from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..config import load_configuration
from ..policy import available_policies, create_policy
from ..report.writer import sha256_file
from ..timing import RamulatorTableTimingModel, RamulatorTimingTable
from ..trace import load_trace
from ..types import ExpertLoad
from .mixed_workload import MixedWorkloadResult, SieveCycleV1Config, run_mixed_workload


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _combined_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _extension_hash(project_root: Path) -> str:
    root = project_root / "ramulator/extensions/sieve_hbm_pim"
    return _combined_sha256(
        [
            root / "source/sieve_mixed_frontend.cpp",
            root / "source/sieve_hbm_pim_controller_v1.cpp",
            root / "patches/ramulator2-cycle-v1.patch",
        ]
    )


def _generator_hash(project_root: Path) -> str:
    root = project_root / "src/sieve_replay/ramulator"
    return _combined_sha256(
        [root / "mixed_workload.py", root / "contention_table_builder.py"]
    )


def _result_evidence(label: str, result: MixedWorkloadResult) -> dict[str, Any]:
    return {
        "label": label,
        **asdict(result),
        "gpu_completion_us": result.gpu_completion_us,
        "pim_completion_us": result.pim_completion_us,
        "total_completion_us": result.total_completion_us,
    }


def _shape(
    gpu_loads: tuple[ExpertLoad, ...],
    pim_loads: tuple[ExpertLoad, ...],
    model: Any,
    cycle: SieveCycleV1Config,
) -> dict[str, int]:
    gpu_expert_count = len(gpu_loads)
    pim_tokens = sum(load.token_count for load in pim_loads)
    gpu_bytes = gpu_expert_count * model.expert_weight_elements * model.dtype_bytes
    if gpu_bytes % cycle.transaction_bytes:
        raise ValueError("GPU expert weight bytes must divide into Ramulator transactions")
    operations_per_mac_wave = (
        cycle.total_pseudo_channels
        * cycle.banks_per_pseudo_channel
        * cycle.transaction_bytes
    )
    bytes_per_partitioned_wave = cycle.total_pseudo_channels * cycle.transaction_bytes
    return {
        "gpu_read_transactions": gpu_bytes // cycle.transaction_bytes,
        "pim_gwrite_waves": (
            pim_tokens
            * _ceil_div(model.hidden_size * model.dtype_bytes, cycle.transaction_bytes)
        ),
        "pim_mac_waves": (
            _ceil_div(
                6 * pim_tokens * model.hidden_size * model.moe_intermediate_size,
                operations_per_mac_wave,
            )
            if pim_tokens
            else 0
        ),
        "pim_read_waves": (
            _ceil_div(
                pim_tokens * model.hidden_size * model.dtype_bytes,
                bytes_per_partitioned_wave,
            )
            if pim_tokens
            else 0
        ),
    }


def build_contention_table(
    experiment_path: str | Path,
    cycle_config_path: str | Path,
    ramulator_root: str | Path,
    output_path: str | Path,
    evidence_path: str | Path | None = None,
) -> dict[str, Any]:
    configuration = load_configuration(experiment_path)
    if configuration.hardware.timing_backend not in {
        "ramulator-table-v0",
        "ramulator-contention-v1",
    }:
        raise ValueError(
            "contention table generation requires a Ramulator-backed experiment"
        )
    isolated_path = configuration.experiment.pim_timing_table_path
    if isolated_path is None:
        raise ValueError("contention table generation requires pim_timing_table")
    cycle_path = Path(cycle_config_path).resolve()
    cycle = SieveCycleV1Config.load(cycle_path)
    if not cycle.dual_row_buffer:
        raise ValueError("formal cycle-v1 contention tables require dual_row_buffer=true")
    project_root = Path(__file__).resolve().parents[3]
    output = Path(output_path)
    cache_root = output.parent / ".cache" / output.stem
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_context = {
        "cycle_config_sha256": sha256_file(cycle_path),
        "extension_sha256": _extension_hash(project_root),
        "mixed_workload_sha256": sha256_file(
            project_root / "src/sieve_replay/ramulator/mixed_workload.py"
        ),
    }

    def run_case(label: str, shape: dict[str, int]) -> MixedWorkloadResult:
        cache_input = {"context": cache_context, "shape": shape}
        cache_key = hashlib.sha256(
            json.dumps(cache_input, sort_keys=True).encode("utf-8")
        ).hexdigest()
        cache_path = cache_root / f"{cache_key}.json"
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("cache_input") == cache_input:
                print(f"cache hit: {label}", flush=True)
                return MixedWorkloadResult(**cached["result"])
        for legacy_path in cache_root.glob("*.json"):
            cached = json.loads(legacy_path.read_text(encoding="utf-8"))
            legacy_input = cached.get("cache_input", {})
            legacy_context = legacy_input.get("context", {})
            if (
                legacy_input.get("shape") == shape
                and legacy_context.get("cycle_config_sha256")
                == cache_context["cycle_config_sha256"]
                and legacy_context.get("extension_sha256")
                == cache_context["extension_sha256"]
                and (
                    "mixed_workload_sha256" not in legacy_context
                    or legacy_context.get("mixed_workload_sha256")
                    == cache_context["mixed_workload_sha256"]
                )
            ):
                result = MixedWorkloadResult(**cached["result"])
                cache_path.write_text(
                    json.dumps(
                        {"cache_input": cache_input, "result": asdict(result)},
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                print(f"compatible cache hit: {label}", flush=True)
                return result
        print(f"running: {label} {shape}", flush=True)
        result = run_mixed_workload(ramulator_root, cycle, **shape)
        cache_path.write_text(
            json.dumps(
                {"cache_input": cache_input, "result": asdict(result)},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"completed: {label} total_us={result.total_completion_us:.6f}",
            flush=True,
        )
        return result

    trace = load_trace(
        configuration.experiment.trace_path,
        configuration.model,
        configuration.experiment.layer,
        configuration.experiment.step,
    )
    isolated_timing = RamulatorTableTimingModel(
        configuration.model,
        configuration.hardware,
        RamulatorTimingTable.load(isolated_path),
    )
    load_by_id = {load.expert_id: load for load in trace.expert_loads}
    hot_order = tuple(
        load.expert_id
        for load in sorted(
            trace.expert_loads, key=lambda load: (-load.token_count, load.expert_id)
        )
    )
    placements: dict[
        tuple[tuple[int, ...], tuple[int, ...]],
        tuple[tuple[ExpertLoad, ...], tuple[ExpertLoad, ...]],
    ] = {}
    policies_by_placement: dict[tuple[tuple[int, ...], tuple[int, ...]], list[str]] = {}
    legacy_placements: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()

    def add_placement(
        gpu_experts: tuple[int, ...],
        pim_experts: tuple[int, ...],
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        key = (tuple(sorted(gpu_experts)), tuple(sorted(pim_experts)))
        placements[key] = (
            tuple(load_by_id[expert] for expert in key[0]),
            tuple(load_by_id[expert] for expert in key[1]),
        )
        return key

    for policy_name in configuration.experiment.policies:
        if policy_name == "sieve-cycle-v1":
            continue
        if policy_name not in available_policies():
            raise ValueError(f"unknown policy in contention source experiment: {policy_name}")
        decision = create_policy(policy_name, isolated_timing).place(trace)
        key = add_placement(decision.gpu_experts, decision.pim_experts)
        legacy_placements.add(key)
        policies_by_placement.setdefault(key, []).append(policy_name)

    def prefix_placement(prefix_length: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if not 0 <= prefix_length <= len(hot_order):
            raise ValueError(f"hot-prefix length is out of range: {prefix_length}")
        gpu = tuple(sorted(hot_order[:prefix_length]))
        pim = tuple(sorted(hot_order[prefix_length:]))
        return add_placement(gpu, pim)

    evidence_runs: list[dict[str, Any]] = []
    recorded_evidence: set[str] = set()

    def record_evidence(label: str, result: MixedWorkloadResult) -> None:
        if label not in recorded_evidence:
            evidence_runs.append(_result_evidence(label, result))
            recorded_evidence.add(label)

    mixed_results: dict[
        tuple[tuple[int, ...], tuple[int, ...]], MixedWorkloadResult
    ] = {}
    prefix_evaluations: dict[int, dict[str, Any]] = {}

    def mixed_for_placement(
        key: tuple[tuple[int, ...], tuple[int, ...]],
    ) -> MixedWorkloadResult:
        if key in mixed_results:
            return mixed_results[key]
        gpu_loads, pim_loads = placements[key]
        shape = _shape(gpu_loads, pim_loads, configuration.model, cycle)
        label = (
            f"mixed-gpu{shape['gpu_read_transactions']}-pim"
            f"({shape['pim_gwrite_waves']},{shape['pim_mac_waves']},{shape['pim_read_waves']})"
        )
        result = run_case(label, shape)
        mixed_results[key] = result
        record_evidence(label, result)
        return result

    def evaluate_prefix(prefix_length: int) -> dict[str, Any]:
        if prefix_length in prefix_evaluations:
            return prefix_evaluations[prefix_length]
        key = prefix_placement(prefix_length)
        gpu_loads, pim_loads = placements[key]
        mixed = mixed_for_placement(key)
        gpu_compute_us = isolated_timing.gpu_expert_compute(gpu_loads).duration_us
        gpu_path_us = mixed.gpu_completion_us + gpu_compute_us
        pim_path_us = mixed.pim_completion_us
        scheduler_us = configuration.hardware.sieve_scheduler_overhead_us
        evaluation = {
            "gpu_prefix_length": prefix_length,
            "gpu_experts": list(key[0]),
            "gpu_tokens": sum(load.token_count for load in gpu_loads),
            "pim_tokens": sum(load.token_count for load in pim_loads),
            "gpu_memory_us": mixed.gpu_completion_us,
            "gpu_compute_us": gpu_compute_us,
            "gpu_path_us": gpu_path_us,
            "pim_path_us": pim_path_us,
            "scheduler_us": scheduler_us,
            "objective_us": scheduler_us + max(gpu_path_us, pim_path_us),
        }
        prefix_evaluations[prefix_length] = evaluation
        return evaluation

    low = 0
    high = len(hot_order)
    evaluate_prefix(low)
    evaluate_prefix(high)
    while high - low > 1:
        midpoint = (low + high) // 2
        evaluation = evaluate_prefix(midpoint)
        if float(evaluation["gpu_path_us"]) <= float(evaluation["pim_path_us"]):
            low = midpoint
        else:
            high = midpoint
    for prefix_length in range(max(0, low - 1), min(len(hot_order), high + 1) + 1):
        evaluate_prefix(prefix_length)

    for key in legacy_placements:
        prefix_length = len(key[0])
        if set(key[0]) == set(hot_order[:prefix_length]):
            evaluate_prefix(prefix_length)
        else:
            mixed_for_placement(key)

    ordered_evaluations = [
        prefix_evaluations[index] for index in sorted(prefix_evaluations)
    ]
    for previous, current in zip(ordered_evaluations, ordered_evaluations[1:]):
        if float(current["gpu_path_us"]) + 1e-12 < float(previous["gpu_path_us"]):
            raise ValueError("hot-prefix GPU path is not monotonic; bisection is invalid")
        if float(current["pim_path_us"]) > float(previous["pim_path_us"]) + 1e-12:
            raise ValueError("hot-prefix PIM path is not monotonic; bisection is invalid")

    selected_prefix, selected_evaluation = min(
        prefix_evaluations.items(),
        key=lambda item: (
            float(item[1]["objective_us"]),
            abs(float(item[1]["gpu_path_us"]) - float(item[1]["pim_path_us"])),
            item[0],
        ),
    )
    selected_key = prefix_placement(selected_prefix)
    isolated_required = legacy_placements | {selected_key}
    isolated_gpu: dict[int, MixedWorkloadResult] = {}
    isolated_pim: dict[tuple[int, int, int], MixedWorkloadResult] = {}
    rows: list[dict[str, Any]] = []
    for key in sorted(placements, key=lambda item: (len(item[0]), item)):
        gpu_loads, pim_loads = placements[key]
        shape = _shape(gpu_loads, pim_loads, configuration.model, cycle)
        mixed = mixed_for_placement(key)
        gpu_transactions = shape["gpu_read_transactions"]
        pim_wave_key = (
            shape["pim_gwrite_waves"],
            shape["pim_mac_waves"],
            shape["pim_read_waves"],
        )
        needs_isolated = key in isolated_required
        if needs_isolated and gpu_transactions:
            if not any(pim_wave_key):
                isolated_gpu[gpu_transactions] = mixed
            elif gpu_transactions not in isolated_gpu:
                gpu_shape = {
                    "gpu_read_transactions": gpu_transactions,
                    "pim_gwrite_waves": 0,
                    "pim_mac_waves": 0,
                    "pim_read_waves": 0,
                }
                label = f"gpu-isolated-{gpu_transactions}"
                isolated_gpu[gpu_transactions] = run_case(label, gpu_shape)
                record_evidence(label, isolated_gpu[gpu_transactions])
        if needs_isolated and any(pim_wave_key):
            if not gpu_transactions:
                isolated_pim[pim_wave_key] = mixed
            elif pim_wave_key not in isolated_pim:
                pim_shape = {
                    "gpu_read_transactions": 0,
                    "pim_gwrite_waves": pim_wave_key[0],
                    "pim_mac_waves": pim_wave_key[1],
                    "pim_read_waves": pim_wave_key[2],
                }
                label = f"pim-isolated-{pim_wave_key}"
                isolated_pim[pim_wave_key] = run_case(label, pim_shape)
                record_evidence(label, isolated_pim[pim_wave_key])

        gpu_isolated_us = (
            isolated_gpu[gpu_transactions].gpu_completion_us
            if gpu_transactions in isolated_gpu
            else (0.0 if not gpu_transactions else None)
        )
        pim_isolated_us = (
            isolated_pim[pim_wave_key].pim_completion_us
            if pim_wave_key in isolated_pim
            else (0.0 if not any(pim_wave_key) else None)
        )
        gwrite_end = mixed.pim_gwrite_completion_us
        mac_end = mixed.pim_mac_completion_us
        read_end = mixed.pim_read_completion_us
        rows.append(
            {
                "gpu_experts": list(key[0]),
                "pim_experts": list(key[1]),
                "gpu_token_count": sum(load.token_count for load in gpu_loads),
                "pim_token_count": sum(load.token_count for load in pim_loads),
                **shape,
                "gpu_isolated_us": gpu_isolated_us,
                "pim_isolated_us": pim_isolated_us,
                "gpu_contended_us": mixed.gpu_completion_us,
                "pim_contended_us": mixed.pim_completion_us,
                "pim_gwrite_contended_us": gwrite_end,
                "pim_compute_contended_us": max(0.0, mac_end - gwrite_end),
                "pim_read_contended_us": max(0.0, read_end - max(gwrite_end, mac_end)),
                "total_memory_phase_us": mixed.total_completion_us,
                "gpu_blocked_by_pim_cycles": mixed.gpu_blocked_by_pim_cycles,
                "pim_blocked_by_gpu_cycles": mixed.pim_blocked_by_gpu_cycles,
                "pim_queue_wait_cycles": mixed.pim_queue_wait_cycles,
                "gpu_column_issues": mixed.gpu_column_issues,
                "pim_column_issues": mixed.pim_column_issues,
                "pim_row_activations": mixed.pim_row_activations,
                "pim_row_conflicts": mixed.pim_row_conflicts,
            }
        )

    table = {
        "schema_version": 2,
        "metadata": {
            "units": "us",
            "ramulator_commit": cycle.ramulator_commit,
            "extension_commit": f"cycle-v1-sha256:{_extension_hash(project_root)}",
            "cycle_config_sha256": sha256_file(cycle_path),
            "isolated_timing_table_sha256": sha256_file(isolated_path),
            "trace_sha256": sha256_file(configuration.experiment.trace_path),
            "generator_sha256": _generator_hash(project_root),
            "dual_row_buffer": cycle.dual_row_buffer,
            "model_sha256": sha256_file(configuration.experiment.model_path),
            "hardware_sha256": sha256_file(configuration.experiment.hardware_path),
            "candidate_space": "hot-prefix",
            "search_method": "monotonic-crossing-bisection-v1",
            "search_total_prefixes": len(hot_order) + 1,
            "search_evaluated_prefixes": sorted(prefix_evaluations),
            "search_selected_prefix": selected_prefix,
            "search_measured_monotonicity_verified": True,
        },
        "expert_contention": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def portable_path(path: str | Path) -> str:
        resolved = Path(path).resolve()
        try:
            return resolved.relative_to(project_root).as_posix()
        except ValueError:
            return str(resolved)

    evidence = {
        "schema_version": 2,
        "cycle_config": asdict(cycle),
        "experiment": portable_path(experiment_path),
        "policies_by_placement": {
            f"gpu={list(key[0])};pim={list(key[1])}": value
            for key, value in sorted(policies_by_placement.items())
        },
        "search": {
            "candidate_space": "hot-prefix",
            "method": "monotonic-crossing-bisection-v1",
            "total_prefixes": len(hot_order) + 1,
            "evaluated_prefixes": sorted(prefix_evaluations),
            "selected_prefix": selected_prefix,
            "measured_monotonicity_verified": True,
            "selected_evaluation": selected_evaluation,
            "evaluations": [
                prefix_evaluations[index] for index in sorted(prefix_evaluations)
            ],
        },
        "runs": evidence_runs,
        "table": portable_path(output),
    }
    evidence_output = (
        Path(evidence_path)
        if evidence_path is not None
        else output.with_name(f"{output.stem}.evidence.json")
    )
    evidence_output.parent.mkdir(parents=True, exist_ok=True)
    evidence_output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return table
