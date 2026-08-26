#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sieve_replay.ramulator import build_timing_table  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a Sieve PIM timing table with Ramulator")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--cycle-config", required=True)
    parser.add_argument("--ramulator-root", default=str(PROJECT_ROOT / "third_party/ramulator2"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--evidence")
    args = parser.parse_args()
    table = build_timing_table(
        args.experiment,
        args.cycle_config,
        args.ramulator_root,
        args.output,
        args.evidence,
    )
    print(
        json.dumps(
            {
                "attention_entries": len(table["attention"]),
                "expert_entries": len(table["expert_gemv"]),
                "output": str(Path(args.output).resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
