from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..types import ExpertLoad
from .ramulator_table import PINNED_RAMULATOR_COMMIT


@dataclass(frozen=True)
class ExpertContentionTiming:
    gpu_experts: tuple[int, ...]
    pim_experts: tuple[int, ...]
    gpu_token_count: int
    pim_token_count: int
    gpu_read_transactions: int
    pim_gwrite_waves: int
    pim_mac_waves: int
    pim_read_waves: int
    gpu_isolated_us: float | None
    pim_isolated_us: float | None
    gpu_contended_us: float
    pim_contended_us: float
    pim_gwrite_contended_us: float
    pim_compute_contended_us: float
    pim_read_contended_us: float
    total_memory_phase_us: float
    gpu_blocked_by_pim_cycles: int
    pim_blocked_by_gpu_cycles: int
    pim_queue_wait_cycles: int
    gpu_column_issues: int
    pim_column_issues: int
    pim_row_activations: int
    pim_row_conflicts: int

    def as_report(self) -> dict[str, int | float | bool | None]:
        gpu_delta = (
            self.gpu_contended_us - self.gpu_isolated_us
            if self.gpu_isolated_us is not None
            else None
        )
        pim_delta = (
            self.pim_contended_us - self.pim_isolated_us
            if self.pim_isolated_us is not None
            else None
        )
        return {
            "dual_row_buffer": True,
            "gpu_isolated_us": self.gpu_isolated_us,
            "gpu_contended_us": self.gpu_contended_us,
            "gpu_contention_delta_us": gpu_delta,
            "pim_isolated_us": self.pim_isolated_us,
            "pim_contended_us": self.pim_contended_us,
            "pim_contention_delta_us": pim_delta,
            "total_memory_phase_us": self.total_memory_phase_us,
            "gpu_blocked_by_pim_cycles": self.gpu_blocked_by_pim_cycles,
            "pim_blocked_by_gpu_cycles": self.pim_blocked_by_gpu_cycles,
            "pim_queue_wait_cycles": self.pim_queue_wait_cycles,
            "gpu_column_issues": self.gpu_column_issues,
            "pim_column_issues": self.pim_column_issues,
            "pim_row_activations": self.pim_row_activations,
            "pim_row_conflicts": self.pim_row_conflicts,
        }


class RamulatorContentionTable:
    _ENTRY_FIELDS = set(ExpertContentionTiming.__dataclass_fields__)
    _METADATA_FIELDS_V1 = {
        "units",
        "ramulator_commit",
        "extension_commit",
        "cycle_config_sha256",
        "isolated_timing_table_sha256",
        "trace_sha256",
        "generator_sha256",
        "dual_row_buffer",
    }
    _METADATA_FIELDS_V2 = _METADATA_FIELDS_V1 | {
        "model_sha256",
        "hardware_sha256",
        "candidate_space",
        "search_method",
        "search_total_prefixes",
        "search_evaluated_prefixes",
        "search_selected_prefix",
        "search_measured_monotonicity_verified",
    }
    _METADATA_FIELDS_V3 = _METADATA_FIELDS_V1 | {
        "model_sha256",
        "hardware_sha256",
        "candidate_space",
        "search_method",
        "workload_count",
        "workload_keys_sha256",
    }

    def __init__(
        self,
        metadata: dict[str, Any],
        entries: dict[tuple[tuple[int, ...], tuple[int, ...], int, int], ExpertContentionTiming],
        schema_version: int = 1,
    ) -> None:
        self.metadata = metadata
        self._entries = entries
        self.schema_version = schema_version

    @classmethod
    def load(cls, path: str | Path) -> "RamulatorContentionTable":
        table_path = Path(path)
        try:
            raw = json.loads(table_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"Ramulator contention table does not exist: {table_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid Ramulator contention table JSON: {exc}") from exc
        cls._require_exact_keys(raw, {"schema_version", "metadata", "expert_contention"}, "table")
        schema_version = raw["schema_version"]
        if schema_version not in {1, 2, 3}:
            raise ValueError(f"unsupported contention table schema: {raw['schema_version']}")
        metadata = raw["metadata"]
        metadata_fields = {
            1: cls._METADATA_FIELDS_V1,
            2: cls._METADATA_FIELDS_V2,
            3: cls._METADATA_FIELDS_V3,
        }[schema_version]
        cls._require_exact_keys(metadata, metadata_fields, "metadata")
        if metadata["units"] != "us":
            raise ValueError("contention table units must be microseconds")
        if metadata["ramulator_commit"] != PINNED_RAMULATOR_COMMIT:
            raise ValueError("contention table was generated with an unpinned revision")
        if metadata["dual_row_buffer"] is not True:
            raise ValueError("formal contention tables require dual_row_buffer=true")
        if schema_version == 2:
            cls._validate_search_metadata(metadata)
        if schema_version == 3:
            cls._validate_workload_metadata(metadata)
        if not isinstance(metadata["extension_commit"], str) or not metadata["extension_commit"]:
            raise ValueError("extension_commit must be a non-empty string")
        for field in (
            "cycle_config_sha256",
            "isolated_timing_table_sha256",
            "trace_sha256",
            "generator_sha256",
        ):
            value = metadata[field]
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"metadata field {field} must be a SHA-256 digest")
        if schema_version in {2, 3}:
            for field in ("model_sha256", "hardware_sha256"):
                value = metadata[field]
                if not isinstance(value, str) or len(value) != 64:
                    raise ValueError(f"metadata field {field} must be a SHA-256 digest")
        if not isinstance(raw["expert_contention"], list) or not raw["expert_contention"]:
            raise ValueError("expert_contention must be a non-empty array")

        entries: dict[
            tuple[tuple[int, ...], tuple[int, ...], int, int], ExpertContentionTiming
        ] = {}
        for row in raw["expert_contention"]:
            cls._require_exact_keys(row, cls._ENTRY_FIELDS, "expert_contention row")
            gpu_experts = cls._expert_ids(row["gpu_experts"], "gpu_experts")
            pim_experts = cls._expert_ids(row["pim_experts"], "pim_experts")
            if set(gpu_experts) & set(pim_experts):
                raise ValueError("contention table GPU and PIM expert sets overlap")
            integer_fields = {
                "gpu_token_count",
                "pim_token_count",
                "gpu_read_transactions",
                "pim_gwrite_waves",
                "pim_mac_waves",
                "pim_read_waves",
                "gpu_blocked_by_pim_cycles",
                "pim_blocked_by_gpu_cycles",
                "pim_queue_wait_cycles",
                "gpu_column_issues",
                "pim_column_issues",
                "pim_row_activations",
                "pim_row_conflicts",
            }
            optional_float_fields = {
                "gpu_isolated_us",
                "pim_isolated_us",
            }
            float_fields = {
                "gpu_contended_us",
                "pim_contended_us",
                "pim_gwrite_contended_us",
                "pim_compute_contended_us",
                "pim_read_contended_us",
                "total_memory_phase_us",
            }
            values: dict[str, Any] = {
                "gpu_experts": gpu_experts,
                "pim_experts": pim_experts,
            }
            values.update(
                {name: cls._nonnegative_int(row[name], name) for name in integer_fields}
            )
            values.update(
                {name: cls._nonnegative_number(row[name], name) for name in float_fields}
            )
            for name in optional_float_fields:
                value = row[name]
                if value is None:
                    if schema_version == 1:
                        raise ValueError(f"schema v1 field {name} cannot be null")
                    values[name] = None
                else:
                    values[name] = cls._nonnegative_number(value, name)
            entry = ExpertContentionTiming(**values)
            if entry.gpu_experts and entry.gpu_read_transactions == 0:
                raise ValueError("GPU expert placement requires GPU read transactions")
            if entry.pim_experts and entry.pim_mac_waves == 0:
                raise ValueError("PIM expert placement requires PIM MAC waves")
            key = cls._key(entry.gpu_experts, entry.pim_experts, entry.gpu_token_count, entry.pim_token_count)
            if key in entries:
                raise ValueError(f"duplicate contention timing entry: {key}")
            entries[key] = entry
        return cls(dict(metadata), entries, schema_version=schema_version)

    @classmethod
    def _validate_workload_metadata(cls, metadata: dict[str, Any]) -> None:
        if metadata["candidate_space"] != "hot-prefix":
            raise ValueError("schema v3 candidate_space must be hot-prefix")
        if metadata["search_method"] != "enumerate-all-hot-prefixes-v1":
            raise ValueError("unsupported schema v3 contention-table search method")
        if cls._nonnegative_int(metadata["workload_count"], "workload_count") <= 0:
            raise ValueError("schema v3 workload_count must be positive")
        digest = metadata["workload_keys_sha256"]
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("workload_keys_sha256 must be a SHA-256 digest")

    @classmethod
    def _validate_search_metadata(cls, metadata: dict[str, Any]) -> None:
        if metadata["candidate_space"] != "hot-prefix":
            raise ValueError("schema v2 candidate_space must be hot-prefix")
        if metadata["search_method"] != "monotonic-crossing-bisection-v1":
            raise ValueError("unsupported contention-table search method")
        total = cls._nonnegative_int(
            metadata["search_total_prefixes"], "search_total_prefixes"
        )
        if total < 2:
            raise ValueError("search_total_prefixes must include both endpoints")
        prefixes_value = metadata["search_evaluated_prefixes"]
        if not isinstance(prefixes_value, list):
            raise ValueError("search_evaluated_prefixes must be an array")
        prefixes = tuple(
            cls._nonnegative_int(value, "search_evaluated_prefixes")
            for value in prefixes_value
        )
        if prefixes != tuple(sorted(set(prefixes))):
            raise ValueError("search_evaluated_prefixes must be sorted and unique")
        if not prefixes or prefixes[0] != 0 or prefixes[-1] != total - 1:
            raise ValueError("search_evaluated_prefixes must contain both endpoints")
        selected = cls._nonnegative_int(
            metadata["search_selected_prefix"], "search_selected_prefix"
        )
        if selected not in prefixes:
            raise ValueError("search_selected_prefix must be an evaluated prefix")
        if metadata["search_measured_monotonicity_verified"] is not True:
            raise ValueError("schema v2 search must verify measured opposing path monotonicity")

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def validate_inputs(
        self,
        isolated_timing_table_path: str | Path,
        trace_path: str | Path,
        cycle_config_path: str | Path,
        model_path: str | Path | None = None,
        hardware_path: str | Path | None = None,
    ) -> None:
        checks = {
            "isolated_timing_table_sha256": Path(isolated_timing_table_path),
            "trace_sha256": Path(trace_path),
            "cycle_config_sha256": Path(cycle_config_path),
        }
        for metadata_key, path in checks.items():
            actual = self._sha256(path)
            if actual != self.metadata[metadata_key]:
                raise ValueError(
                    f"contention table {metadata_key} does not match current input: {path}"
                )
        if self.schema_version in {2, 3}:
            if model_path is None or hardware_path is None:
                raise ValueError("schema v2 contention validation requires model and hardware paths")
            for metadata_key, path in {
                "model_sha256": Path(model_path),
                "hardware_sha256": Path(hardware_path),
            }.items():
                actual = self._sha256(path)
                if actual != self.metadata[metadata_key]:
                    raise ValueError(
                        f"contention table {metadata_key} does not match current input: {path}"
                    )

    @staticmethod
    def _require_exact_keys(value: Any, expected: set[str], label: str) -> None:
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be an object")
        actual = set(value)
        if actual != expected:
            raise ValueError(
                f"{label} fields differ; missing={expected - actual}, extra={actual - expected}"
            )

    @staticmethod
    def _nonnegative_int(value: Any, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} must be a non-negative integer")
        return value

    @staticmethod
    def _nonnegative_number(value: Any, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{label} must be a non-negative number")
        return float(value)

    @classmethod
    def _expert_ids(cls, value: Any, label: str) -> tuple[int, ...]:
        if not isinstance(value, list):
            raise ValueError(f"{label} must be an array")
        result = tuple(cls._nonnegative_int(item, label) for item in value)
        if result != tuple(sorted(set(result))):
            raise ValueError(f"{label} must be sorted and unique")
        return result

    @staticmethod
    def _key(
        gpu_experts: tuple[int, ...],
        pim_experts: tuple[int, ...],
        gpu_tokens: int,
        pim_tokens: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...], int, int]:
        return gpu_experts, pim_experts, gpu_tokens, pim_tokens

    def expert_timing(
        self,
        gpu_loads: tuple[ExpertLoad, ...],
        pim_loads: tuple[ExpertLoad, ...],
    ) -> ExpertContentionTiming:
        gpu_experts = tuple(sorted(load.expert_id for load in gpu_loads))
        pim_experts = tuple(sorted(load.expert_id for load in pim_loads))
        gpu_tokens = sum(load.token_count for load in gpu_loads)
        pim_tokens = sum(load.token_count for load in pim_loads)
        key = self._key(gpu_experts, pim_experts, gpu_tokens, pim_tokens)
        try:
            return self._entries[key]
        except KeyError as exc:
            raise ValueError(
                "missing exact Ramulator contention timing for "
                f"gpu_experts={gpu_experts}, pim_experts={pim_experts}, "
                f"tokens=({gpu_tokens}, {pim_tokens})"
            ) from exc

    def candidate_timings(
        self,
        loads: tuple[ExpertLoad, ...],
    ) -> tuple[ExpertContentionTiming, ...]:
        token_counts = {load.expert_id: load.token_count for load in loads}
        active = set(token_counts)
        candidates: list[ExpertContentionTiming] = []
        for entry in self._entries.values():
            gpu = set(entry.gpu_experts)
            pim = set(entry.pim_experts)
            if gpu & pim or gpu | pim != active:
                continue
            if entry.gpu_token_count != sum(token_counts[expert] for expert in gpu):
                continue
            if entry.pim_token_count != sum(token_counts[expert] for expert in pim):
                continue
            candidates.append(entry)
        return tuple(
            sorted(candidates, key=lambda entry: (len(entry.gpu_experts), entry.gpu_experts))
        )
