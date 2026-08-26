from __future__ import annotations

import json
import hashlib
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


@dataclass(frozen=True)
class TraceSet:
    batches: tuple[TraceBatch, ...]

    @property
    def steps(self) -> tuple[int, ...]:
        return tuple(sorted({batch.step for batch in self.batches}))

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(sorted({batch.layer for batch in self.batches}))

    @property
    def batch_size(self) -> int:
        sizes = {batch.batch_size for batch in self.batches}
        if len(sizes) != 1:
            raise ValueError("trace set has inconsistent batch sizes")
        return next(iter(sizes))


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
    if layer >= model.num_hidden_layers:
        raise ValueError(f"trace line {line_number}: layer exceeds model layer count")
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


def _read_records(path: str | Path, model: ModelConfig) -> tuple[RouterTraceRecord, ...]:
    trace_path = Path(path)
    records: list[RouterTraceRecord] = []
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
        records.append(_parse_record(raw, line_number, model))
    if not records:
        raise ValueError(f"trace contains no records: {trace_path}")
    return tuple(records)


def load_trace_set(
    path: str | Path,
    model: ModelConfig,
    layers: tuple[int, ...],
    steps: tuple[int, ...],
) -> TraceSet:
    selected_layers = set(layers)
    selected_steps = set(steps)
    grouped: dict[tuple[int, int], list[RouterTraceRecord]] = {}
    seen: dict[tuple[int, int], set[Hashable]] = {}
    for record in _read_records(path, model):
        if record.layer not in selected_layers or record.step not in selected_steps:
            continue
        key = (record.step, record.layer)
        seen_tokens = seen.setdefault(key, set())
        if record.token_id in seen_tokens:
            raise ValueError(
                f"duplicate token_id {record.token_id!r} for "
                f"layer={record.layer}, step={record.step}"
            )
        seen_tokens.add(record.token_id)
        grouped.setdefault(key, []).append(record)

    missing = [
        (step, layer)
        for step in steps
        for layer in layers
        if (step, layer) not in grouped
    ]
    if missing:
        raise ValueError(f"trace is missing selected step/layer batches: {missing}")
    batches = tuple(
        TraceBatch(step=step, layer=layer, records=tuple(grouped[(step, layer)]))
        for step in steps
        for layer in layers
    )
    _validate_decode_sequence(batches, layers, steps)
    return TraceSet(batches)


def _validate_decode_sequence(
    batches: tuple[TraceBatch, ...],
    layers: tuple[int, ...],
    steps: tuple[int, ...],
) -> None:
    by_key = {(batch.step, batch.layer): batch for batch in batches}
    reference_tokens: tuple[Hashable, ...] | None = None
    context_by_step: dict[int, tuple[int, ...]] = {}
    for step in steps:
        first = by_key[(step, layers[0])]
        signature = tuple((record.token_id, record.context_length) for record in first.records)
        tokens = tuple(record.token_id for record in first.records)
        if reference_tokens is None:
            reference_tokens = tokens
        elif tokens != reference_tokens:
            raise ValueError("decode trace token ordering must remain stable across steps")
        context_by_step[step] = first.context_lengths
        for layer in layers[1:]:
            current = by_key[(step, layer)]
            current_signature = tuple(
                (record.token_id, record.context_length) for record in current.records
            )
            if current_signature != signature:
                raise ValueError(
                    f"decode trace request/context mismatch at step={step}, layer={layer}"
                )
    for previous, current in zip(steps, steps[1:]):
        expected_delta = current - previous
        if any(
            current_length - previous_length != expected_delta
            for previous_length, current_length in zip(
                context_by_step[previous], context_by_step[current]
            )
        ):
            raise ValueError("decode trace context lengths must advance once per decode step")


def load_trace(path: str | Path, model: ModelConfig, layer: int, step: int) -> TraceBatch:
    return load_trace_set(path, model, (layer,), (step,)).batches[0]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
