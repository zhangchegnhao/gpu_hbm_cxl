#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sieve_replay.ramulator.mixed_workload import (  # noqa: E402
    MixedWorkloadResult,
    SieveCycleV1Config,
    run_mixed_workload,
)
from sieve_replay.ramulator.workload_cache import (  # noqa: E402
    ContentionWorkloadCache,
    WorkloadShape,
    cache_context,
)
from sieve_replay.ramulator.workload_catalog import build_workload_catalog  # noqa: E402


def _run_shape(
    ramulator_root: str,
    cycle_config: str,
    shape: dict[str, int],
) -> tuple[dict[str, int], dict[str, Any], float]:
    started = time.monotonic()
    cycle = SieveCycleV1Config.load(cycle_config)
    result = run_mixed_workload(ramulator_root, cycle, **shape)
    return shape, asdict(result), time.monotonic() - started


def _request_count(shape: WorkloadShape, total_pseudo_channels: int) -> int:
    pim_waves = shape.pim_gwrite_waves + shape.pim_mac_waves + shape.pim_read_waves
    return shape.gpu_read_transactions + pim_waves * total_pseudo_channels


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fill exact cycle-v1 decode workloads with independent Ramulator workers"
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--cycle-config", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--catalog-output", required=True)
    parser.add_argument(
        "--ramulator-root", default=str(PROJECT_ROOT / "third_party/ramulator2")
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="maximum missing shapes to run; 0 runs every missing shape",
    )
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.limit < 0:
        parser.error("--limit must be non-negative")

    cycle_path = Path(args.cycle_config).resolve()
    cycle = SieveCycleV1Config.load(cycle_path)
    catalog = build_workload_catalog(
        args.experiment,
        cycle_path,
        args.cache_dir,
        args.catalog_output,
        max_new_runs=0,
    )
    missing = [
        WorkloadShape(**row["shape"])
        for row in catalog["workloads"]
        if not row["cached"]
    ]
    missing.sort(
        key=lambda shape: (_request_count(shape, cycle.total_pseudo_channels), asdict(shape)),
        reverse=True,
    )
    if args.limit:
        missing = missing[: args.limit]
    if not missing:
        print(json.dumps({"completed": 0, "failed": 0, "remaining": 0}, sort_keys=True))
        return 0

    cache = ContentionWorkloadCache(
        args.cache_dir, cache_context(PROJECT_ROOT, cycle_path)
    )
    failures: list[dict[str, Any]] = []
    completed = 0
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=min(args.workers, len(missing))) as executor:
        futures = {
            executor.submit(
                _run_shape,
                str(Path(args.ramulator_root).resolve()),
                str(cycle_path),
                asdict(shape),
            ): shape
            for shape in missing
        }
        for future in as_completed(futures):
            shape = futures[future]
            try:
                shape_dict, result_dict, elapsed = future.result()
                completed_shape = WorkloadShape(**shape_dict)
                cache.store(completed_shape, MixedWorkloadResult(**result_dict))
                completed += 1
                print(
                    json.dumps(
                        {
                            "cache_key": cache.key(completed_shape),
                            "completed": completed,
                            "elapsed_s": round(elapsed, 3),
                            "shape": shape_dict,
                            "submitted": len(missing),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - preserve remaining workers
                failure = {"shape": asdict(shape), "error": f"{type(exc).__name__}: {exc}"}
                failures.append(failure)
                print(json.dumps({"failure": failure}, sort_keys=True), file=sys.stderr, flush=True)

    final_catalog = build_workload_catalog(
        args.experiment,
        cycle_path,
        args.cache_dir,
        args.catalog_output,
        max_new_runs=0,
    )
    print(
        json.dumps(
            {
                "completed": completed,
                "failed": len(failures),
                "remaining": final_catalog["summary"]["missing_workload_shapes"],
                "wall_time_s": round(time.monotonic() - started, 3),
            },
            sort_keys=True,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
