from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ModelConfig
from .trace import TraceSet, sha256_file


@dataclass(frozen=True)
class TraceManifest:
    path: Path
    raw: dict[str, Any]

    @classmethod
    def load_and_validate(
        cls,
        manifest_path: str | Path,
        trace_path: str | Path,
        model: ModelConfig,
        trace_set: TraceSet,
    ) -> "TraceManifest":
        path = Path(manifest_path).resolve()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"trace manifest does not exist: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid trace manifest JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError("trace manifest root must be an object")
        _require_exact_keys(
            raw,
            {"schema_version", "trace_file", "trace_sha256", "model", "workload", "capture"},
            "trace manifest",
        )
        if raw["schema_version"] != 1:
            raise ValueError(f"unsupported trace manifest schema: {raw['schema_version']}")

        manifest_trace = (path.parent / _string(raw["trace_file"], "trace_file")).resolve()
        selected_trace = Path(trace_path).resolve()
        if manifest_trace != selected_trace:
            raise ValueError(
                f"trace manifest points to {manifest_trace}, but experiment uses {selected_trace}"
            )
        expected_hash = _sha256(raw["trace_sha256"], "trace_sha256")
        if sha256_file(selected_trace) != expected_hash:
            raise ValueError("trace manifest SHA-256 does not match the trace file")

        model_raw = _object(raw["model"], "model")
        _require_exact_keys(
            model_raw,
            {
                "name",
                "revision",
                "dtype",
                "num_hidden_layers",
                "num_experts",
                "num_experts_per_tok",
            },
            "trace manifest model",
        )
        expected_model = {
            "name": model.name,
            "dtype": model.dtype,
            "num_hidden_layers": model.num_hidden_layers,
            "num_experts": model.num_experts,
            "num_experts_per_tok": model.num_experts_per_tok,
        }
        for key, expected in expected_model.items():
            if model_raw[key] != expected:
                raise ValueError(
                    f"trace manifest model field {key}={model_raw[key]!r} "
                    f"does not match configuration value {expected!r}"
                )
        _string(model_raw["revision"], "model.revision")

        workload = _object(raw["workload"], "workload")
        _require_exact_keys(
            workload,
            {
                "dataset",
                "dataset_revision",
                "split",
                "prompt_count",
                "batch_size",
                "decode_steps",
                "seed",
            },
            "trace manifest workload",
        )
        for field in ("dataset", "dataset_revision", "split"):
            _string(workload[field], f"workload.{field}")
        prompt_count = _positive_int(workload["prompt_count"], "workload.prompt_count")
        batch_size = _positive_int(workload["batch_size"], "workload.batch_size")
        decode_steps = _positive_int(workload["decode_steps"], "workload.decode_steps")
        _nonnegative_int(workload["seed"], "workload.seed")
        if prompt_count != batch_size or batch_size != trace_set.batch_size:
            raise ValueError("trace manifest prompt_count/batch_size does not match trace records")
        if max(trace_set.steps) >= decode_steps:
            raise ValueError("selected trace step exceeds manifest decode_steps")

        capture = _object(raw["capture"], "capture")
        _require_exact_keys(
            capture,
            {
                "framework",
                "framework_version",
                "torch_version",
                "device_map",
                "created_at_utc",
                "timing_source",
            },
            "trace manifest capture",
        )
        for field in capture:
            _string(capture[field], f"capture.{field}")
        if capture["timing_source"] != "routing-only; no hardware timing captured":
            raise ValueError("trace capture must explicitly exclude hardware timing")
        return cls(path=path, raw=raw)


def _require_exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} fields differ; missing={expected - actual}, extra={actual - expected}"
        )


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _sha256(value: Any, label: str) -> str:
    result = _string(value, label)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return result
