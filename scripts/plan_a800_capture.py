#!/usr/bin/env python3
"""Validate inputs and emit the plan-only A800 capture protocol."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from sieve_replay.capture.a800_metrics import write_plan  # noqa: E402

def main() -> int:
    parser = argparse.ArgumentParser(description="Plan independent A800 timing/memory capture (no execution)")
    parser.add_argument("--output", default="results/a800_capture_plan_v1/plan.json")
    args = parser.parse_args()
    try:
        path = write_plan(PROJECT_ROOT, PROJECT_ROOT / args.output)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    plan = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps({"output": str(path.resolve()), "status": plan["status"], "execution_implemented": plan["execution_implemented"], "cases": len(plan["cases"])}, sort_keys=True))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
