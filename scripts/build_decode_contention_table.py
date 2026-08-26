#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sieve_replay.ramulator.decode_contention_table import (  # noqa: E402
    build_decode_contention_table,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize a strict multi-workload cycle-v1 contention table"
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--cycle-config", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evidence")
    args = parser.parse_args()
    try:
        table = build_decode_contention_table(
            args.experiment,
            args.cycle_config,
            args.cache_dir,
            args.output,
            args.evidence,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "expert_contention_entries": len(table["expert_contention"]),
                "output": str(Path(args.output).resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
