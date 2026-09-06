from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PINNED_RAMULATOR_COMMIT = "b30320bc9385b708e86b67ebb9f48858cc66d798"


@dataclass(frozen=True)
class ExpertGemvTiming:
    gwrite_us: float
    compute_us: float
    read_us: float


class RamulatorTimingTable:
    """Strict lookup table; formal runs must never silently interpolate data."""

    def __init__(
        self,
        metadata: dict[str, str],
        attention: dict[tuple[int, ...], float],
        expert_gemv: dict[int, ExpertGemvTiming],
        schema_version: int = 1,
    ) -> None:
        self.metadata = metadata
        self._attention = attention
        self._expert_gemv = expert_gemv
        self.schema_version = schema_version

    @classmethod
    def load(cls, path: str | Path) -> "RamulatorTimingTable":
        table_path = Path(path)
        try:
            raw = json.loads(table_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"Ramulator timing table does not exist: {table_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid Ramulator timing table JSON: {exc}") from exc
        cls._validate_root(raw)
        metadata = raw["metadata"]
        schema_version = raw["schema_version"]
        attention: dict[tuple[int, ...], float] = {}
        for row in raw["attention"]:
            if schema_version == 1:
                cls._require_exact_keys(
                    row,
                    {"batch_size", "context_length", "duration_us"},
                    "attention row",
                )
                key = (
                    cls._positive_int(row["batch_size"], "batch_size"),
                    cls._positive_int(row["context_length"], "context_length"),
                )
            else:
                cls._require_exact_keys(
                    row, {"context_lengths", "duration_us"}, "attention row"
                )
                context_lengths = row["context_lengths"]
                if (
                    not isinstance(context_lengths, list)
                    or not context_lengths
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value <= 0
                        for value in context_lengths
                    )
                ):
                    raise ValueError(
                        "attention row context_lengths must be a non-empty positive integer array"
                    )
                key = tuple(context_lengths)
            if key in attention:
                raise ValueError(f"duplicate attention timing entry: {key}")
            attention[key] = cls._positive_number(row["duration_us"], "duration_us")

        expert_gemv: dict[int, ExpertGemvTiming] = {}
        for row in raw["expert_gemv"]:
            cls._require_exact_keys(
                row,
                {"token_count", "gwrite_us", "compute_us", "read_us"},
                "expert_gemv row",
            )
            token_count = cls._positive_int(row["token_count"], "token_count")
            if token_count in expert_gemv:
                raise ValueError(f"duplicate expert GEMV timing entry: {token_count}")
            expert_gemv[token_count] = ExpertGemvTiming(
                gwrite_us=cls._nonnegative_number(row["gwrite_us"], "gwrite_us"),
                compute_us=cls._positive_number(row["compute_us"], "compute_us"),
                read_us=cls._nonnegative_number(row["read_us"], "read_us"),
            )
        if not attention or not expert_gemv:
            raise ValueError("Ramulator timing table cannot contain empty timing sections")
        return cls(dict(metadata), attention, expert_gemv, schema_version=schema_version)

    @staticmethod
    def _validate_root(raw: Any) -> None:
        if not isinstance(raw, dict):
            raise ValueError("Ramulator timing table root must be an object")
        RamulatorTimingTable._require_exact_keys(
            raw, {"schema_version", "metadata", "attention", "expert_gemv"}, "table"
        )
        if raw["schema_version"] not in {1, 2}:
            raise ValueError(f"unsupported Ramulator timing table schema: {raw['schema_version']}")
        metadata = raw["metadata"]
        if not isinstance(metadata, dict):
            raise ValueError("Ramulator timing metadata must be an object")
        required_metadata = {
            "units",
            "ramulator_commit",
            "extension_commit",
            "hardware_config_sha256",
            "trace_generator_sha256",
        }
        RamulatorTimingTable._require_exact_keys(metadata, required_metadata, "metadata")
        if metadata["units"] != "us":
            raise ValueError("Ramulator timing table units must be microseconds")
        if metadata["ramulator_commit"] != PINNED_RAMULATOR_COMMIT:
            raise ValueError("Ramulator timing table was generated with an unpinned revision")
        if not isinstance(metadata["extension_commit"], str) or not metadata["extension_commit"]:
            raise ValueError("metadata field extension_commit must be a non-empty string")
        for field in ("hardware_config_sha256", "trace_generator_sha256"):
            value = metadata[field]
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"metadata field {field} must be a SHA-256 hex digest")
        if not isinstance(raw["attention"], list) or not isinstance(raw["expert_gemv"], list):
            raise ValueError("Ramulator timing sections must be arrays")

    @staticmethod
    def _require_exact_keys(value: Any, expected: set[str], label: str) -> None:
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be an object")
        actual = set(value)
        if actual != expected:
            raise ValueError(f"{label} fields differ; missing={expected - actual}, extra={actual - expected}")

    @staticmethod
    def _positive_int(value: Any, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} must be a positive integer")
        return value

    @staticmethod
    def _positive_number(value: Any, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{label} must be positive")
        return float(value)

    @staticmethod
    def _nonnegative_number(value: Any, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{label} must be non-negative")
        return float(value)

    def attention_us(
        self,
        batch_size_or_context_lengths: int | tuple[int, ...],
        context_length: int | None = None,
    ) -> float:
        if self.schema_version == 1:
            if (
                isinstance(batch_size_or_context_lengths, bool)
                or not isinstance(batch_size_or_context_lengths, int)
                or context_length is None
            ):
                raise ValueError(
                    "Ramulator timing table schema v1 requires batch_size and context_length"
                )
            key = (batch_size_or_context_lengths, context_length)
        else:
            if context_length is not None or not isinstance(
                batch_size_or_context_lengths, tuple
            ):
                raise ValueError(
                    "Ramulator timing table schema v2 requires the complete context_lengths tuple"
                )
            key = batch_size_or_context_lengths
        try:
            return self._attention[key]
        except KeyError as exc:
            raise ValueError(f"missing Ramulator attention timing for batch/context={key}") from exc

    def expert_us(self, token_count: int) -> ExpertGemvTiming:
        try:
            return self._expert_gemv[token_count]
        except KeyError as exc:
            raise ValueError(f"missing Ramulator expert timing for token_count={token_count}") from exc
