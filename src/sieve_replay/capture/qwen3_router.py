from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PromptRecord:
    request_id: str | int
    prompt: str


def load_prompt_records(path: str | Path, max_prompts: int | None = None) -> tuple[PromptRecord, ...]:
    if max_prompts is not None and max_prompts <= 0:
        raise ValueError("max_prompts must be positive")
    prompt_path = Path(path)
    try:
        lines = prompt_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ValueError(f"prompt JSONL does not exist: {prompt_path}") from exc
    records: list[PromptRecord] = []
    seen: set[str | int] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid prompt JSON on line {line_number}: {exc}") from exc
        if not isinstance(raw, dict) or set(raw) != {"request_id", "prompt"}:
            raise ValueError(
                f"prompt line {line_number} must contain exactly request_id and prompt"
            )
        request_id = raw["request_id"]
        prompt = raw["prompt"]
        if not isinstance(request_id, (str, int)) or isinstance(request_id, bool):
            raise ValueError(f"prompt line {line_number}: request_id must be a string or integer")
        if request_id in seen:
            raise ValueError(f"prompt line {line_number}: duplicate request_id {request_id!r}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"prompt line {line_number}: prompt must be a non-empty string")
        seen.add(request_id)
        records.append(PromptRecord(request_id, prompt))
        if max_prompts is not None and len(records) >= max_prompts:
            break
    if not records:
        raise ValueError("prompt JSONL contains no records")
    return tuple(records)


class _CaptureState:
    def __init__(self, torch: Any, top_k: int, normalize_topk: bool) -> None:
        self.torch = torch
        self.top_k = top_k
        self.normalize_topk = normalize_topk
        self.active = False
        self.step = -1
        self.context_lengths: tuple[int, ...] = ()
        self.request_ids: tuple[str | int, ...] = ()
        self.records: list[dict[str, Any]] = []

    def begin_step(
        self,
        step: int,
        context_lengths: tuple[int, ...],
        request_ids: tuple[str | int, ...],
    ) -> None:
        self.step = step
        self.context_lengths = context_lengths
        self.request_ids = request_ids
        self.active = True

    def end_step(self) -> None:
        self.active = False

    def hook(self, layer: int):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            if not self.active:
                return
            logits = output[0] if isinstance(output, tuple) else output
            logits = logits.reshape(-1, logits.shape[-1])
            if logits.shape[0] != len(self.request_ids):
                raise ValueError(
                    f"router layer {layer} emitted {logits.shape[0]} tokens; "
                    f"expected decode batch {len(self.request_ids)}"
                )
            probabilities = self.torch.softmax(logits, dim=-1, dtype=self.torch.float32)
            weights, experts = self.torch.topk(
                probabilities, self.top_k, dim=-1
            )
            if self.normalize_topk:
                weights = weights / weights.sum(dim=-1, keepdim=True)
            expert_rows = experts.detach().cpu().tolist()
            weight_rows = weights.detach().cpu().tolist()
            for request_id, context_length, expert_ids, expert_weights in zip(
                self.request_ids,
                self.context_lengths,
                expert_rows,
                weight_rows,
            ):
                normalized = [float(value) for value in expert_weights]
                normalized[-1] += 1.0 - sum(normalized)
                self.records.append(
                    {
                        "step": self.step,
                        "layer": layer,
                        "token_id": request_id,
                        "context_length": context_length,
                        "expert_ids": [int(value) for value in expert_ids],
                        "expert_weights": normalized,
                    }
                )

        return capture


def capture_qwen3_router_trace(
    model_name_or_path: str,
    revision: str,
    prompts_path: str | Path,
    output_dir: str | Path,
    decode_steps: int,
    dtype: str,
    device_map: str,
    dataset: str,
    dataset_revision: str,
    dataset_split: str,
    seed: int,
    max_prompts: int | None = None,
    trust_remote_code: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    if decode_steps <= 0:
        raise ValueError("decode_steps must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ValueError(
            "router capture requires the optional trace-capture dependencies; "
            "install requirements/trace_capture.txt"
        ) from exc

    prompts = load_prompt_records(prompts_path, max_prompts=max_prompts)
    output = Path(output_dir)
    trace_path = output / "router.jsonl"
    prompt_snapshot_path = output / "prompts.jsonl"
    manifest_path = output / "manifest.json"
    if not overwrite and any(
        path.exists() for path in (trace_path, prompt_snapshot_path, manifest_path)
    ):
        raise ValueError(f"capture output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    dtype_map = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype not in dtype_map:
        raise ValueError(f"unsupported dtype {dtype!r}")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        revision=revision,
        torch_dtype=dtype_map[dtype],
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    config = model.config
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or len(layers) != int(config.num_hidden_layers):
        raise ValueError("model does not expose the expected model.layers structure")
    top_k = int(config.num_experts_per_tok)
    capture = _CaptureState(torch, top_k, bool(config.norm_topk_prob))
    hooks = []
    for layer_index, layer in enumerate(layers):
        gate = getattr(getattr(layer, "mlp", None), "gate", None)
        if gate is None:
            raise ValueError(f"layer {layer_index} does not expose mlp.gate")
        hooks.append(gate.register_forward_hook(capture.hook(layer_index)))

    encoded = tokenizer(
        [record.prompt for record in prompts],
        return_tensors="pt",
        padding=True,
    )
    input_device = model.get_input_embeddings().weight.device
    input_ids = encoded["input_ids"].to(input_device)
    attention_mask = encoded["attention_mask"].to(input_device)
    prompt_lengths = tuple(int(value) for value in attention_mask.sum(dim=1).tolist())
    request_ids = tuple(record.request_id for record in prompts)
    try:
        with torch.inference_mode():
            prefill = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                # The prefill only needs the final token logits to seed greedy
                # decode.  Keeping logits for every context token can allocate
                # tens of GiB in the LM head on long-context A800 captures.
                logits_to_keep=1,
            )
            past_key_values = prefill.past_key_values
            next_tokens = prefill.logits[:, -1, :].argmax(dim=-1)
            for step in range(decode_steps):
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            (attention_mask.shape[0], 1),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ],
                    dim=1,
                )
                context_lengths = tuple(length + step + 1 for length in prompt_lengths)
                before = len(capture.records)
                capture.begin_step(step, context_lengths, request_ids)
                decoded = model(
                    input_ids=next_tokens[:, None],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                capture.end_step()
                expected = len(layers) * len(prompts)
                if len(capture.records) - before != expected:
                    raise ValueError(
                        f"decode step {step} captured {len(capture.records) - before} "
                        f"router records; expected {expected}"
                    )
                past_key_values = decoded.past_key_values
                next_tokens = decoded.logits[:, -1, :].argmax(dim=-1)
    finally:
        capture.end_step()
        for hook in hooks:
            hook.remove()

    with trace_path.open("w", encoding="utf-8", newline="") as handle:
        for record in capture.records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    with prompt_snapshot_path.open("w", encoding="utf-8", newline="") as handle:
        for record in prompts:
            handle.write(
                json.dumps(
                    {"request_id": record.request_id, "prompt": record.prompt},
                    separators=(",", ":"),
                )
                + "\n"
            )
    resolved_revision = getattr(config, "_commit_hash", None) or revision
    manifest = {
        "schema_version": 2,
        "trace_file": trace_path.name,
        "trace_sha256": _sha256(trace_path),
        "prompts": {
            "file": prompt_snapshot_path.name,
            "sha256": _sha256(prompt_snapshot_path),
            "count": len(prompts),
        },
        "model": {
            "name": Path(str(model_name_or_path)).name,
            "revision": str(resolved_revision),
            "dtype": _normalize_dtype(config, dtype),
            "num_hidden_layers": int(config.num_hidden_layers),
            "num_experts": int(config.num_experts),
            "num_experts_per_tok": top_k,
        },
        "workload": {
            "dataset": dataset,
            "dataset_revision": dataset_revision,
            "split": dataset_split,
            "prompt_count": len(prompts),
            "batch_size": len(prompts),
            "decode_steps": decode_steps,
            "seed": seed,
        },
        "capture": {
            "framework": "transformers",
            "framework_version": transformers.__version__,
            "torch_version": torch.__version__,
            "device_map": device_map,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "timing_source": "routing-only; no hardware timing captured",
            "generation_strategy": "greedy-argmax-fixed-steps",
            "fixed_batch": True,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _normalize_dtype(config: Any, requested: str) -> str:
    value = getattr(config, "torch_dtype", None)
    if value is None or str(value) == "None":
        return requested
    return str(value).removeprefix("torch.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
