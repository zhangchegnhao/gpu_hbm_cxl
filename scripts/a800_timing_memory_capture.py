#!/usr/bin/env python3
"""Measure independent Qwen3 A800 prefill/decode timing and KV memory."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prompt_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str | int] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) != {"request_id", "prompt"}:
            raise ValueError(f"invalid prompt schema on line {line_number}")
        if row["request_id"] in seen or not isinstance(row["prompt"], str) or not row["prompt"].strip():
            raise ValueError(f"invalid prompt on line {line_number}")
        seen.add(row["request_id"])
        rows.append(row)
    if not rows:
        raise ValueError("prompt file is empty")
    return rows


def cache_bytes(cache: Any) -> int:
    seen: set[int] = set()
    total = 0

    def visit(value: Any) -> None:
        nonlocal total
        if value is None:
            return
        if hasattr(value, "data_ptr") and hasattr(value, "numel"):
            marker = int(value.data_ptr())
            if marker in seen:
                return
            seen.add(marker)
            total += int(value.numel()) * int(value.element_size())
            return
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
            return
        if isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
            return
        for name in ("key_cache", "value_cache"):
            if hasattr(value, name):
                visit(getattr(value, name))

    visit(cache)
    return total


class AttentionEvents:
    def __init__(self, torch: Any, modules: list[tuple[str, Any]]) -> None:
        self.torch = torch
        self.modules = modules
        self.starts: dict[str, Any] = {}
        self.samples: list[tuple[str, Any, Any]] = []
        self.handles: list[Any] = []

    def install(self) -> None:
        for name, module in self.modules:
            def before(_module: Any, _inputs: Any, _name: str = name) -> None:
                event = self.torch.cuda.Event(enable_timing=True)
                event.record()
                self.starts[_name] = event

            def after(_module: Any, _inputs: Any, _output: Any, _name: str = name) -> None:
                start = self.starts.pop(_name, None)
                if start is not None:
                    end = self.torch.cuda.Event(enable_timing=True)
                    end.record()
                    self.samples.append((_name, start, end))

            self.handles.append(module.register_forward_pre_hook(before))
            self.handles.append(module.register_forward_hook(after))

    def clear(self) -> None:
        self.samples.clear()
        self.starts.clear()

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def durations(self) -> list[dict[str, Any]]:
        return [
            {"module": name, "duration_ms": float(start.elapsed_time(end))}
            for name, start, end in self.samples
        ]


def gpu_info(torch: Any) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "total_memory_bytes": int(props.total_memory),
        "capability": [int(props.major), int(props.minor)],
        "device_count": int(torch.cuda.device_count()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    prompts_path = Path(args.prompts).resolve()
    prompts = prompt_records(prompts_path)
    if len(prompts) != args.batch_size:
        raise ValueError(f"prompt count {len(prompts)} != batch size {args.batch_size}")
    model_path = Path(args.model).resolve()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, revision=args.revision,
        torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    encoded = tokenizer([row["prompt"] for row in prompts], return_tensors="pt", padding=True)
    lengths = [int(value) for value in encoded["attention_mask"].sum(dim=1).tolist()]
    expected_prompt_length = args.context_length - 1
    if any(length != expected_prompt_length for length in lengths):
        raise ValueError(f"token lengths {lengths} != expected prompt length {expected_prompt_length}")
    input_device = model.get_input_embeddings().weight.device
    input_ids = encoded["input_ids"].to(input_device)
    attention_mask = encoded["attention_mask"].to(input_device)
    modules = [
        (name, module) for name, module in model.named_modules()
        if name.endswith(".self_attn") or name == "self_attn"
    ]
    if len(modules) != int(model.config.num_hidden_layers):
        raise ValueError(f"attention module count {len(modules)} != 48")
    events = AttentionEvents(torch, modules)
    events.install()
    memory_rows: list[dict[str, Any]] = []
    decode_rows: list[dict[str, Any]] = []

    def prefill_call() -> Any:
        return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, logits_to_keep=1)

    with torch.inference_mode():
        for _ in range(args.warmup_runs):
            warm = prefill_call()
            del warm
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        events.clear()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        prefill = prefill_call()
        end.record()
        torch.cuda.synchronize()
        prefill_ms = float(start.elapsed_time(end))
        prefill_attention = events.durations()
        past = prefill.past_key_values
        next_tokens = prefill.logits[:, -1, :].argmax(dim=-1)
        memory_rows.append({
            "phase": "prefill",
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "kv_cache_bytes": cache_bytes(past),
        })
        for step in range(args.decode_steps):
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((args.batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device),
            ], dim=1)
            events.clear()
            start.record()
            decoded = model(
                input_ids=next_tokens[:, None], attention_mask=attention_mask,
                past_key_values=past, use_cache=True,
            )
            end.record()
            torch.cuda.synchronize()
            duration_ms = float(start.elapsed_time(end))
            step_attention = events.durations()
            past = decoded.past_key_values
            next_tokens = decoded.logits[:, -1, :].argmax(dim=-1)
            memory_rows.append({
                "phase": "decode", "step": step,
                "allocated_bytes": int(torch.cuda.memory_allocated()),
                "reserved_bytes": int(torch.cuda.memory_reserved()),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "kv_cache_bytes": cache_bytes(past),
            })
            decode_rows.append({
                "step": step, "decode_ms": duration_ms,
                "attention_total_ms": sum(row["duration_ms"] for row in step_attention),
                "attention_modules": step_attention,
            })
    events.remove()
    timing = {
        "status": "passed", "case": args.case, "batch_size": args.batch_size,
        "context_length": args.context_length, "decode_steps": args.decode_steps,
        "prefill_ms": prefill_ms,
        "decode_ms": [row["decode_ms"] for row in decode_rows],
        "decode_attention_ms": [row["attention_total_ms"] for row in decode_rows],
        "prefill_attention_ms": sum(row["duration_ms"] for row in prefill_attention),
        "decode_median_ms": sorted(row["decode_ms"] for row in decode_rows)[len(decode_rows) // 2],
        "decode_tokens_per_second": args.batch_size * args.decode_steps * 1000.0 / sum(row["decode_ms"] for row in decode_rows),
        "memory": memory_rows,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "kv_cache_bytes_after_prefill": memory_rows[0]["kv_cache_bytes"],
        "kv_cache_bytes_after_decode": memory_rows[-1]["kv_cache_bytes"],
        "input_token_lengths": lengths,
        "prefill_attention_modules": prefill_attention,
        "software": {
            "python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "transformers": transformers.__version__,
        },
        "device": gpu_info(torch),
    }
    (output / "timing.json").write_text(json.dumps(timing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "memory.json").write_text(json.dumps(memory_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    snapshot = output / "prompts.jsonl"
    with snapshot.open("w", encoding="utf-8", newline="") as handle:
        for row in prompts:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    manifest = {
        "schema_version": 1,
        "status": "passed",
        "experiment": "a800-independent-attention-decode-timing-v1",
        "timing_source": "CUDA events; independent pass without Router hooks",
        "memory_source": "CUDA allocator counters and past_key_values tensors",
        "case": args.case,
        "workload": {"batch_size": args.batch_size, "context_length": args.context_length,
                      "decode_steps": args.decode_steps, "warmup_runs": args.warmup_runs},
        "model": {"path": str(model_path), "revision": args.revision, "dtype": "bfloat16",
                  "num_hidden_layers": int(model.config.num_hidden_layers),
                  "num_key_value_heads": int(model.config.num_key_value_heads),
                  "head_dim": int(model.config.head_dim)},
        "prompts": {"source": str(prompts_path), "sha256": sha256(prompts_path),
                    "snapshot": snapshot.name, "snapshot_sha256": sha256(snapshot),
                    "count": len(prompts)},
        "device": timing["device"], "software": timing["software"],
        "outputs": {"timing.json": sha256(output / "timing.json"),
                     "memory.json": sha256(output / "memory.json"),
                     "prompts.jsonl": sha256(snapshot)},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--warmup-runs", type=int, default=1)
    args = parser.parse_args()
    try:
        manifest = run(args)
    except Exception as exc:
        output = Path(args.output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        status = "oom" if exc.__class__.__name__ == "OutOfMemoryError" else "error"
        (output / "manifest.json").write_text(json.dumps({
            "schema_version": 1, "status": status,
            "experiment": "a800-independent-attention-decode-timing-v1",
            "case": args.case, "batch_size": args.batch_size,
            "context_length": args.context_length, "error": type(exc).__name__,
            "message": str(exc)[:500],
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": status, "case": args.case, "error": type(exc).__name__}, sort_keys=True))
        return 3 if status == "oom" else 2
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
