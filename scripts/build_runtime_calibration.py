#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sieve_replay.config import load_configuration  # noqa: E402
from sieve_replay.report.writer import sha256_file  # noqa: E402
from sieve_replay.timing import RamulatorContentionTable  # noqa: E402
from sieve_replay.trace import load_trace_set  # noqa: E402


def _counts(value: str, label: str) -> tuple[int, ...]:
    try:
        counts = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise ValueError(f"{label} must be a comma-separated integer list") from exc
    if len(counts) < 2 or counts[0] != 0 or counts != tuple(sorted(set(counts))):
        raise ValueError(f"{label} must contain sorted unique counts starting at zero")
    return counts


def _record_exact(
    values: dict[int, float], count: int, duration: float | None, label: str
) -> None:
    if duration is None:
        raise ValueError(f"{label} isolated timing is unavailable for count={count}")
    previous = values.setdefault(count, duration)
    if abs(previous - duration) > 1e-12:
        raise ValueError(f"{label} isolated timing is inconsistent for count={count}")


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a sparse runtime calibration from exact isolated timings"
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evidence-output", required=True)
    parser.add_argument(
        "--gpu-anchors", default="0,1,2,4,8,16,32,49"
    )
    parser.add_argument(
        "--pim-anchors", default="0,1,2,4,8,16,32,48,64"
    )
    args = parser.parse_args()

    configuration = load_configuration(args.experiment)
    experiment = configuration.experiment
    contention_path = experiment.contention_timing_table_path
    isolated_path = experiment.pim_timing_table_path
    cycle_path = experiment.ramulator_cycle_config_path
    if contention_path is None or isolated_path is None or cycle_path is None:
        raise ValueError(
            "calibration source experiment requires contention, isolated, and cycle tables"
        )
    table = RamulatorContentionTable.load(contention_path)
    table.validate_inputs(
        isolated_path,
        experiment.trace_path,
        cycle_path,
        experiment.model_path,
        experiment.hardware_path,
    )
    traces = load_trace_set(
        experiment.trace_path,
        configuration.model,
        experiment.layers,
        experiment.steps,
    ).batches

    gpu_exact: dict[int, float] = {}
    pim_exact: dict[int, float] = {}
    for trace in traces:
        for row in table.candidate_timings(trace.expert_loads):
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

    gpu_counts = _counts(args.gpu_anchors, "gpu anchors")
    pim_counts = _counts(args.pim_anchors, "pim anchors")
    missing_gpu = sorted(set(gpu_counts) - set(gpu_exact))
    missing_pim = sorted(set(pim_counts) - set(pim_exact))
    if missing_gpu or missing_pim:
        raise ValueError(
            f"requested calibration anchors are unavailable; gpu={missing_gpu}, pim={missing_pim}"
        )

    source_table_sha256 = sha256_file(contention_path)
    calibration = {
        "schema_version": 1,
        "metadata": {
            "units": "us",
            "method": "piecewise-linear-isolated-v1",
            "source": "sparse exact isolated anchors extracted from cycle-v1 table",
            "scope": (
                "current Qwen3/B200 HBM-PIM model; not validated across models or hardware"
            ),
            "model_sha256": sha256_file(experiment.model_path),
            "hardware_sha256": sha256_file(experiment.hardware_path),
            "cycle_config_sha256": sha256_file(cycle_path),
            "source_contention_table_sha256": source_table_sha256,
            "generator_sha256": sha256_file(Path(__file__)),
        },
        "gpu_expert_count_anchors": [
            {"expert_count": count, "duration_us": gpu_exact[count]}
            for count in gpu_counts
        ],
        "pim_token_count_anchors": [
            {"token_count": count, "duration_us": pim_exact[count]}
            for count in pim_counts
        ],
    }
    evidence = {
        "schema_version": 1,
        "experiment": str(Path(args.experiment)),
        "source_contention_table": _portable_path(contention_path),
        "source_contention_table_sha256": source_table_sha256,
        "available_gpu_expert_counts": sorted(gpu_exact),
        "available_pim_token_counts": sorted(pim_exact),
        "selected_gpu_expert_counts": list(gpu_counts),
        "selected_pim_token_counts": list(pim_counts),
        "nonzero_calibration_points": len(gpu_counts) + len(pim_counts) - 2,
        "full_cycle_v1_workload_shapes_required_for_runtime_decision": 0,
        "source_table_use": (
            "evaluation proxy for dedicated isolated calibration measurements"
        ),
    }
    output = Path(args.output)
    evidence_output = Path(args.evidence_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    evidence["calibration"] = _portable_path(output)
    evidence["calibration_sha256"] = sha256_file(output)
    evidence_output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "gpu_anchors": len(gpu_counts),
                "pim_anchors": len(pim_counts),
                "nonzero_calibration_points": evidence["nonzero_calibration_points"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
