#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sieve_replay.config import load_configuration  # noqa: E402
from sieve_replay.trace import load_trace_set  # noqa: E402
from sieve_replay.trace_analysis import (  # noqa: E402
    analyze_router_trace,
    write_router_trace_analysis,
)
from sieve_replay.trace_manifest import TraceManifest  # noqa: E402


def _portable(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and characterize a full Decode router trace"
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hot-experts", type=int, default=8)
    args = parser.parse_args()
    try:
        configuration = load_configuration(args.experiment)
        experiment = configuration.experiment
        trace_set = load_trace_set(
            experiment.trace_path,
            configuration.model,
            experiment.layers,
            experiment.steps,
        )
        is_real = "real" in experiment.trace_path.parts
        if is_real and experiment.trace_manifest_path is None:
            raise ValueError("traces under traces/real require an experiment trace_manifest")
        if experiment.trace_manifest_path is not None:
            TraceManifest.load_and_validate(
                experiment.trace_manifest_path,
                experiment.trace_path,
                configuration.model,
                trace_set,
            )
        analysis = analyze_router_trace(
            trace_set,
            configuration.model,
            experiment.trace_path,
            trace_kind="manifest-bound-real" if is_real else "synthetic-validation",
            manifest_path=experiment.trace_manifest_path,
            hot_expert_count=args.hot_experts,
            trace_label=_portable(experiment.trace_path),
        )
        write_router_trace_analysis(args.output, analysis)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": _portable(Path(args.output)),
                "trace_kind": analysis.summary["trace_kind"],
                "layer_batches": analysis.summary["selection"]["layer_batches"],
                "unique_load_signatures": analysis.summary["load_shape"][
                    "unique_signatures"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
