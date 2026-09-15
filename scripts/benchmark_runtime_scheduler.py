#!/usr/bin/env python3
"""Measure the CPU cost of the runtime-v1 placement decision."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sieve_replay.config import load_configuration  # noqa: E402
from sieve_replay.policy import create_policy  # noqa: E402
from sieve_replay.timing import AnalyticTimingModel, RuntimeExpertTimingModel  # noqa: E402
from sieve_replay.trace import load_trace_set  # noqa: E402
from sieve_replay.report.writer import sha256_file  # noqa: E402


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    if args.repeats <= 0 or args.warmup < 0:
        raise ValueError("repeats must be positive and warmup cannot be negative")

    configuration = load_configuration(args.experiment)
    calibration_path = configuration.experiment.runtime_calibration_path
    cycle_path = configuration.experiment.ramulator_cycle_config_path
    if calibration_path is None or cycle_path is None:
        raise ValueError("scheduler benchmark requires runtime calibration and cycle config")
    calibration = RuntimeExpertTimingModel.load(calibration_path)
    calibration.validate_inputs(
        configuration.experiment.model_path,
        configuration.experiment.hardware_path,
        cycle_path,
    )
    traces = load_trace_set(
        configuration.experiment.trace_path,
        configuration.model,
        configuration.experiment.layers,
        configuration.experiment.steps,
    ).batches
    timing = AnalyticTimingModel(configuration.model, configuration.hardware, calibration)
    policy = create_policy("sieve-runtime-v1", timing)
    for _ in range(args.warmup):
        for trace in traces:
            policy.place(trace)

    durations_us: list[float] = []
    candidate_counts: set[int] = set()
    for _ in range(args.repeats):
        for trace in traces:
            start = time.perf_counter_ns()
            decision = policy.place(trace)
            durations_us.append((time.perf_counter_ns() - start) / 1000.0)
            if decision.search_report is None:
                raise ValueError("runtime policy did not return a search report")
            candidate_counts.add(int(decision.search_report["candidate_count"]))
    summary = {
        "schema_version": 1,
        "policy": "sieve-runtime-v1",
        "experiment": str(Path(args.experiment)),
        "trace_layer_batches": len(traces),
        "repeats": args.repeats,
        "warmup_rounds": args.warmup,
        "measurements": len(durations_us),
        "candidate_counts": sorted(candidate_counts),
        "decision_cpu_time_us": {
            "mean": statistics.fmean(durations_us),
            "median": percentile(durations_us, 0.50),
            "p95": percentile(durations_us, 0.95),
            "p99": percentile(durations_us, 0.99),
            "min": min(durations_us),
            "max": max(durations_us),
        },
        "configured_scheduler_overhead_us": (
            configuration.hardware.sieve_scheduler_overhead_us
        ),
        "median_overhead_ratio_to_model": (
            percentile(durations_us, 0.50)
            / configuration.hardware.sieve_scheduler_overhead_us
        ),
        "implied_eight_step_total_overhead_us": {
            "configured_model": (
                configuration.hardware.sieve_scheduler_overhead_us * len(traces)
            ),
            "measured_median": percentile(durations_us, 0.50) * len(traces),
            "measured_p95": percentile(durations_us, 0.95) * len(traces),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "input_hashes": {
            "model_sha256": sha256_file(configuration.experiment.model_path),
            "hardware_sha256": sha256_file(configuration.experiment.hardware_path),
            "trace_sha256": sha256_file(configuration.experiment.trace_path),
            "runtime_calibration_sha256": sha256_file(calibration_path),
            "ramulator_cycle_config_sha256": sha256_file(cycle_path),
            "benchmark_script_sha256": sha256_file(Path(__file__)),
        },
        "method_boundary": {
            "exact_table_candidates_read": False,
            "ramulator_runs": 0,
            "measurement_scope": "Python process CPU decision time; excludes event replay",
            "scheduler_model_comparison": "configured 20 us overhead is a separate simulation parameter",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary["decision_cpu_time_us"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
