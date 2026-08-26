from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..config import load_configuration
from ..trace import load_trace_set
from ..types import ExpertLoad
from .mixed_workload import SieveCycleV1Config, run_mixed_workload
from .workload_cache import (
    ContentionWorkloadCache,
    WorkloadShape,
    cache_context,
    shape_for_loads,
)


def build_workload_catalog(
    experiment_path: str | Path,
    cycle_config_path: str | Path,
    cache_dir: str | Path,
    output_path: str | Path,
    ramulator_root: str | Path | None = None,
    max_new_runs: int = 0,
) -> dict[str, Any]:
    if max_new_runs < 0:
        raise ValueError("max_new_runs must be non-negative")
    if max_new_runs and ramulator_root is None:
        raise ValueError("running missing workloads requires ramulator_root")
    configuration = load_configuration(experiment_path)
    trace_set = load_trace_set(
        configuration.experiment.trace_path,
        configuration.model,
        configuration.experiment.layers,
        configuration.experiment.steps,
    )
    cycle_path = Path(cycle_config_path).resolve()
    cycle = SieveCycleV1Config.load(cycle_path)
    project_root = Path(__file__).resolve().parents[3]
    cache = ContentionWorkloadCache(
        cache_dir, cache_context(project_root, cycle_path)
    )
    consumers_by_shape: dict[WorkloadShape, list[dict[str, Any]]] = {}

    def add(
        shape: WorkloadShape,
        trace_step: int,
        trace_layer: int,
        kind: str,
        gpu_experts: tuple[int, ...],
        pim_experts: tuple[int, ...],
        gpu_tokens: int,
        pim_tokens: int,
    ) -> None:
        consumers_by_shape.setdefault(shape, []).append(
            {
                "step": trace_step,
                "layer": trace_layer,
                "kind": kind,
                "gpu_experts": list(gpu_experts),
                "pim_experts": list(pim_experts),
                "gpu_tokens": gpu_tokens,
                "pim_tokens": pim_tokens,
            }
        )

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
            gpu_tokens = sum(load.token_count for load in gpu_loads)
            pim_tokens = sum(load.token_count for load in pim_loads)
            add(
                shape_for_loads(gpu_loads, pim_loads, configuration.model, cycle),
                trace.step,
                trace.layer,
                "mixed-hot-prefix",
                gpu_experts,
                pim_experts,
                gpu_tokens,
                pim_tokens,
            )
            if gpu_loads:
                add(
                    shape_for_loads(gpu_loads, (), configuration.model, cycle),
                    trace.step,
                    trace.layer,
                    "gpu-isolated",
                    gpu_experts,
                    (),
                    gpu_tokens,
                    0,
                )
            if pim_loads:
                add(
                    shape_for_loads((), pim_loads, configuration.model, cycle),
                    trace.step,
                    trace.layer,
                    "pim-isolated",
                    (),
                    pim_experts,
                    0,
                    pim_tokens,
                )

    ran = 0
    rows: list[dict[str, Any]] = []
    for shape in sorted(consumers_by_shape, key=lambda item: tuple(asdict(item).values())):
        result = cache.load(shape)
        if result is None and ran < max_new_runs:
            result = run_mixed_workload(ramulator_root, cycle, **asdict(shape))
            cache.store(shape, result)
            ran += 1
        rows.append(
            {
                "cache_key": cache.key(shape),
                "shape": asdict(shape),
                "cached": result is not None,
                "consumer_count": len(consumers_by_shape[shape]),
                "consumers": consumers_by_shape[shape],
            }
        )
    catalog = {
        "schema_version": 1,
        "experiment": str(Path(experiment_path)),
        "cycle_config": str(cycle_path),
        "cache_context": cache.context,
        "summary": {
            "trace_batches": len(trace_set.batches),
            "placement_consumers": sum(len(value) for value in consumers_by_shape.values()),
            "unique_workload_shapes": len(rows),
            "cached_workload_shapes": sum(bool(row["cached"]) for row in rows),
            "missing_workload_shapes": sum(not bool(row["cached"]) for row in rows),
            "new_runs_completed": ran,
            "interpolation": "forbidden",
        },
        "workloads": rows,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return catalog
