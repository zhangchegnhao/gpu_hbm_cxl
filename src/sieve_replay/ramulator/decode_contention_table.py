from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..config import load_configuration
from ..report.writer import sha256_file
from ..trace import load_trace_set
from .mixed_workload import MixedWorkloadResult, SieveCycleV1Config
from .workload_cache import (
    ContentionWorkloadCache,
    WorkloadShape,
    cache_context,
    shape_for_loads,
)


def build_decode_contention_table(
    experiment_path: str | Path,
    cycle_config_path: str | Path,
    cache_dir: str | Path,
    output_path: str | Path,
    evidence_path: str | Path | None = None,
) -> dict[str, Any]:
    configuration = load_configuration(experiment_path)
    isolated_path = configuration.experiment.pim_timing_table_path
    if isolated_path is None:
        raise ValueError("decode contention table requires pim_timing_table")
    cycle_path = Path(cycle_config_path).resolve()
    cycle = SieveCycleV1Config.load(cycle_path)
    if not cycle.dual_row_buffer:
        raise ValueError("formal decode contention tables require dual_row_buffer=true")
    project_root = Path(__file__).resolve().parents[3]
    context = cache_context(project_root, cycle_path)
    cache = ContentionWorkloadCache(cache_dir, context)
    trace_set = load_trace_set(
        configuration.experiment.trace_path,
        configuration.model,
        configuration.experiment.layers,
        configuration.experiment.steps,
    )
    rows_by_key: dict[
        tuple[tuple[int, ...], tuple[int, ...], int, int], dict[str, Any]
    ] = {}
    used_cache_keys: set[str] = set()
    missing_cache_keys: set[str] = set()

    def required(shape: WorkloadShape) -> MixedWorkloadResult | None:
        used_cache_keys.add(cache.key(shape))
        result = cache.load(shape)
        if result is None:
            missing_cache_keys.add(cache.key(shape))
        return result

    for trace in trace_set.batches:
        load_by_id = {load.expert_id: load for load in trace.expert_loads}
        hot_order = tuple(
            load.expert_id
            for load in sorted(
                trace.expert_loads, key=lambda load: (-load.token_count, load.expert_id)
            )
        )
        for prefix in range(len(hot_order) + 1):
            gpu_experts = tuple(sorted(hot_order[:prefix]))
            pim_experts = tuple(sorted(hot_order[prefix:]))
            gpu_loads = tuple(load_by_id[expert] for expert in gpu_experts)
            pim_loads = tuple(load_by_id[expert] for expert in pim_experts)
            shape = shape_for_loads(
                gpu_loads, pim_loads, configuration.model, cycle
            )
            mixed = required(shape)
            gpu_isolated = (
                required(
                    shape_for_loads(gpu_loads, (), configuration.model, cycle)
                )
                if gpu_loads
                else None
            )
            pim_isolated = (
                required(
                    shape_for_loads((), pim_loads, configuration.model, cycle)
                )
                if pim_loads
                else None
            )
            if mixed is None or (gpu_loads and gpu_isolated is None) or (
                pim_loads and pim_isolated is None
            ):
                continue
            gpu_tokens = sum(load.token_count for load in gpu_loads)
            pim_tokens = sum(load.token_count for load in pim_loads)
            row = _row(
                gpu_experts,
                pim_experts,
                gpu_tokens,
                pim_tokens,
                shape,
                mixed,
                gpu_isolated,
                pim_isolated,
            )
            key = (gpu_experts, pim_experts, gpu_tokens, pim_tokens)
            previous = rows_by_key.get(key)
            if previous is not None and previous != row:
                raise ValueError(f"conflicting exact timing rows for placement {key}")
            rows_by_key[key] = row

    if missing_cache_keys:
        sample = ", ".join(sorted(missing_cache_keys)[:8])
        raise ValueError(
            f"decode contention cache is incomplete: {len(missing_cache_keys)} missing "
            f"exact workload keys; first keys: {sample}"
        )
    workload_digest = hashlib.sha256(
        "\n".join(sorted(used_cache_keys)).encode("ascii")
    ).hexdigest()
    generator_digest = hashlib.sha256()
    for path in (
        Path(__file__),
        project_root / "src/sieve_replay/ramulator/workload_cache.py",
        project_root / "src/sieve_replay/ramulator/mixed_workload.py",
    ):
        generator_digest.update(path.name.encode("utf-8"))
        generator_digest.update(path.read_bytes())
    table = {
        "schema_version": 3,
        "metadata": {
            "units": "us",
            "ramulator_commit": cycle.ramulator_commit,
            "extension_commit": f"cycle-v1-sha256:{context['extension_sha256']}",
            "cycle_config_sha256": sha256_file(cycle_path),
            "isolated_timing_table_sha256": sha256_file(isolated_path),
            "trace_sha256": sha256_file(configuration.experiment.trace_path),
            "generator_sha256": generator_digest.hexdigest(),
            "dual_row_buffer": True,
            "model_sha256": sha256_file(configuration.experiment.model_path),
            "hardware_sha256": sha256_file(configuration.experiment.hardware_path),
            "candidate_space": "hot-prefix",
            "search_method": "enumerate-all-hot-prefixes-v1",
            "workload_count": len(trace_set.batches),
            "workload_keys_sha256": workload_digest,
        },
        "expert_contention": [rows_by_key[key] for key in sorted(rows_by_key)],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence = {
        "schema_version": 3,
        "experiment": _portable_path(experiment_path, project_root),
        "cycle_config": asdict(cycle),
        "cache_dir": _portable_path(cache_dir, project_root),
        "trace_batches": len(trace_set.batches),
        "unique_cache_keys": len(used_cache_keys),
        "workload_keys_sha256": workload_digest,
        "contention_entries": len(rows_by_key),
        "table": _portable_path(output, project_root),
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


def _row(
    gpu_experts: tuple[int, ...],
    pim_experts: tuple[int, ...],
    gpu_tokens: int,
    pim_tokens: int,
    shape: WorkloadShape,
    mixed: MixedWorkloadResult,
    gpu_isolated: MixedWorkloadResult | None,
    pim_isolated: MixedWorkloadResult | None,
) -> dict[str, Any]:
    gwrite_end = mixed.pim_gwrite_completion_us
    mac_end = mixed.pim_mac_completion_us
    read_end = mixed.pim_read_completion_us
    return {
        "gpu_experts": list(gpu_experts),
        "pim_experts": list(pim_experts),
        "gpu_token_count": gpu_tokens,
        "pim_token_count": pim_tokens,
        **asdict(shape),
        "gpu_isolated_us": gpu_isolated.gpu_completion_us if gpu_isolated else 0.0,
        "pim_isolated_us": pim_isolated.pim_completion_us if pim_isolated else 0.0,
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


def _portable_path(path: str | Path, project_root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        return str(resolved)
