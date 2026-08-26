from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .config import load_configuration
from .policy import available_policies
from .replay import run_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sieve-replay")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="replay one layer and decode step")
    run.add_argument("--experiment", required=True, help="experiment JSON configuration")
    run.add_argument("--policy", required=True, choices=available_policies())
    run.add_argument("--output", required=True, help="output directory")
    run_all = subparsers.add_parser("run-all", help="replay every policy enabled by an experiment")
    run_all.add_argument("--experiment", required=True, help="experiment JSON configuration")
    run_all.add_argument("--output", required=True, help="root output directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run-all":
        return _run_all(args.experiment, args.output)
    try:
        result = run_experiment(args.experiment, args.policy, args.output)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "policy": result.decision.policy,
                "total_latency_us": result.summary["total_latency_us"],
                "gpu_experts": len(result.decision.gpu_experts),
                "pim_experts": len(result.decision.pim_experts),
                "memory_feasible": result.summary["memory"]["feasible"],
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


def _run_all(experiment: str, output_dir: str) -> int:
    output = Path(output_dir)
    rows: list[dict[str, object]] = []
    contention_rows: list[dict[str, object]] = []
    try:
        configured_policies = load_configuration(experiment).experiment.policies
        unknown = sorted(set(configured_policies) - set(available_policies()))
        if unknown:
            raise ValueError(f"experiment enables unknown policies: {', '.join(unknown)}")
        for policy in configured_policies:
            result = run_experiment(experiment, policy, output / policy)
            rows.append(
                {
                    "policy": policy,
                    "total_latency_us": result.summary["total_latency_us"],
                    "attention_target": result.decision.attention_target.value,
                    "gpu_experts": len(result.decision.gpu_experts),
                    "pim_experts": len(result.decision.pim_experts),
                    "gpu_expert_tokens": result.summary["placement"]["gpu_expert_tokens"],
                    "pim_expert_tokens": result.summary["placement"]["pim_expert_tokens"],
                    "memory_feasible": result.summary["memory"]["feasible"],
                }
            )
            if "contention" in result.summary:
                contention_rows.append(
                    {"policy": policy, **result.summary["contention"]}
                )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    output.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output / "comparison.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if contention_rows:
        contention_fields = list(contention_rows[0])
        with (output / "contention.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=contention_fields, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(contention_rows)
    print(json.dumps({"policies": len(rows), "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
