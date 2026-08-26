from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable

from .config import ModelConfig
from .types import ExpertLoad


@dataclass(frozen=True)
class RouterTraceRecord:
    step: int
    layer: int
    token_id: Hashable
    context_length: int
    expert_ids: tuple[int, ...]
    expert_weights: tuple[float, ...]


@dataclass(frozen=True)
class TraceBatch:
    step: int
    layer: int
    records: tuple[RouterTraceRecord, ...]

    @property
    def batch_size(self) -> int:
        return len(self.records)

    @property
    def context_lengths(self) -> tuple[int, ...]:
        return tuple(record.context_length for record in self.records)

    @property
    def expert_loads(self) -> tuple[ExpertLoad, ...]:
        counts = Counter(expert for record in self.records for expert in record.expert_ids)
        return tuple(ExpertLoad(expert, counts[expert]) for expert in sorted(counts))

    @property
    def total_expert_assignments(self) -> int:
        return sum(len(record.expert_ids) for record in self.records)


def _parse_record(raw: object, line_number: int, model: ModelConfig) -> RouterTraceRecord:
    if not isinstance(raw, dict):
        raise ValueError(f"trace line {line_number}: record must be an object")
    required = {"step", "layer", "token_id", "context_length", "expert_ids", "expert_weights"}
    missing = sorted(required - raw.keys())
    extra = sorted(raw.keys() - required)
    if missing:
        raise ValueError(f"trace line {line_number}: missing fields {missing}")
    if extra:
        raise ValueError(f"trace line {line_number}: unknown fields {extra}")

    expert_ids = tuple(int(value) for value in raw["expert_ids"])
    expert_weights = tuple(float(value) for value in raw["expert_weights"])
    if len(expert_ids) != model.num_experts_per_tok:
        raise ValueError(
            f"trace line {line_number}: expected {model.num_experts_per_tok} experts, "
            f"got {len(expert_ids)}"
        )
    if len(expert_weights) != len(expert_ids):
        raise ValueError(f"trace line {line_number}: expert IDs and weights have different lengths")
    if len(set(expert_ids)) != len(expert_ids):
        raise ValueError(f"trace line {line_number}: duplicate expert ID")
    if any(expert < 0 or expert >= model.num_experts for expert in expert_ids):
        raise ValueError(f"trace line {line_number}: expert ID outside model range")
    if any(weight < 0 for weight in expert_weights):
        raise ValueError(f"trace line {line_number}: negative expert weight")
    if abs(sum(expert_weights) - 1.0) > 1e-5:
        raise ValueError(f"trace line {line_number}: normalized expert weights must sum to 1")

    step = int(raw["step"])
    layer = int(raw["layer"])
    context_length = int(raw["context_length"])
    if step < 0 or layer < 0 or context_length <= 0:
        raise ValueError(f"trace line {line_number}: step/layer/context_length is invalid")
    if context_length > model.max_position_embeddings:
        raise ValueError(f"trace line {line_number}: context exceeds model maximum")
    token_id = raw["token_id"]
    if not isinstance(token_id, (str, int)):
        raise ValueError(f"trace line {line_number}: token_id must be a string or integer")
    return RouterTraceRecord(
        step=step,
        layer=layer,
        token_id=token_id,
        context_length=context_length,
        expert_ids=expert_ids,
        expert_weights=expert_weights,
    )


def load_trace(path: str | Path, model: ModelConfig, layer: int, step: int) -> TraceBatch:
    trace_path = Path(path)
    selected: list[RouterTraceRecord] = []
    seen_tokens: set[Hashable] = set()
    try:
        lines = trace_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ValueError(f"trace file does not exist: {trace_path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on trace line {line_number}: {exc}") from exc
        record = _parse_record(raw, line_number, model)
        if record.layer != layer or record.step != step:
            continue
        if record.token_id in seen_tokens:
            raise ValueError(f"duplicate token_id {record.token_id!r} for layer={layer}, step={step}")
        seen_tokens.add(record.token_id)
        selected.append(record)
    if not selected:
        raise ValueError(f"trace contains no records for layer={layer}, step={step}")
    return TraceBatch(step=step, layer=layer, records=tuple(selected))
