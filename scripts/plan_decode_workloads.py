#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sieve_replay.ramulator.workload_catalog import build_workload_catalog  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan and optionally fill exact cycle-v1 decode workload cache entries"
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--cycle-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--cache-dir",
        default=str(PROJECT_ROOT / "ramulator/timing_tables/generated/.cache/decode_workloads"),
    )
    parser.add_argument(
        "--ramulator-root", default=str(PROJECT_ROOT / "third_party/ramulator2")
    )
    parser.add_argument(
        "--max-new-runs",
        type=int,
        default=0,
        help="0 plans only; a positive value executes at most this many missing exact shapes",
    )
    args = parser.parse_args()
    try:
        catalog = build_workload_catalog(
            args.experiment,
            args.cycle_config,
            args.cache_dir,
            args.output,
            ramulator_root=args.ramulator_root,
            max_new_runs=args.max_new_runs,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(catalog["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
