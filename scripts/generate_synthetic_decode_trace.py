#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


WEIGHTS = (0.30, 0.20, 0.14, 0.10, 0.08, 0.07, 0.06, 0.05)


def _experts(layer: int, step: int, request: int, num_experts: int) -> list[int]:
    hot = (7 * layer + 3 * step) % num_experts
    candidates = [hot]
    if request < 5:
        candidates.append((hot + 1) % num_experts)
    rank = 0
    while len(candidates) < len(WEIGHTS):
        candidate = (hot + 2 + request * 11 + rank * 17) % num_experts
        if candidate not in candidates:
            candidates.append(candidate)
        rank += 1
    return candidates


def generate_trace(
    output: str | Path,
    layers: int,
    steps: int,
    batch_size: int,
    initial_context_length: int,
    num_experts: int,
) -> int:
    if min(layers, steps, batch_size, initial_context_length, num_experts) <= 0:
        raise ValueError("all synthetic trace dimensions must be positive")
    if num_experts < len(WEIGHTS):
        raise ValueError("num_experts must be at least the Top-K width")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        for step in range(steps):
            for layer in range(layers):
                for request in range(batch_size):
                    record = {
                        "step": step,
                        "layer": layer,
                        "token_id": f"request-{request}",
                        "context_length": initial_context_length + step,
                        "expert_ids": _experts(layer, step, request, num_experts),
                        "expert_weights": list(WEIGHTS),
                    }
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                    count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic multi-layer, multi-step router trace"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--initial-context-length", type=int, default=1024)
    parser.add_argument("--num-experts", type=int, default=128)
    args = parser.parse_args()
    count = generate_trace(
        args.output,
        args.layers,
        args.steps,
        args.batch_size,
        args.initial_context_length,
        args.num_experts,
    )
    print(json.dumps({"output": str(Path(args.output).resolve()), "records": count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
