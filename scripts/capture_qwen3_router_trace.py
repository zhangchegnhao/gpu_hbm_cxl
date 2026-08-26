#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sieve_replay.capture import capture_qwen3_router_trace  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture Qwen3 MoE Top-K router decisions without hardware timing"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--decode-steps", type=int, required=True)
    parser.add_argument(
        "--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--dataset-split", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        manifest = capture_qwen3_router_trace(
            model_name_or_path=args.model,
            revision=args.revision,
            prompts_path=args.prompts,
            output_dir=args.output,
            decode_steps=args.decode_steps,
            dtype=args.dtype,
            device_map=args.device_map,
            dataset=args.dataset,
            dataset_revision=args.dataset_revision,
            dataset_split=args.dataset_split,
            seed=args.seed,
            max_prompts=args.max_prompts,
            trust_remote_code=args.trust_remote_code,
            overwrite=args.overwrite,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "trace_sha256": manifest["trace_sha256"],
                "records": (
                    manifest["model"]["num_hidden_layers"]
                    * manifest["workload"]["batch_size"]
                    * manifest["workload"]["decode_steps"]
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
