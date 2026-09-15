from __future__ import annotations

import hashlib
import json
import math
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TimingAnchor:
    count: int
    duration_us: float


class RuntimeExpertTimingModel:
    """Small piecewise-linear timing model used by the online scheduler."""

    _METADATA_FIELDS = {
        "units",
        "method",
        "source",
        "scope",
        "model_sha256",
        "hardware_sha256",
        "cycle_config_sha256",
        "source_contention_table_sha256",
        "generator_sha256",
    }

    def __init__(
        self,
        metadata: dict[str, str],
        gpu_anchors: tuple[TimingAnchor, ...],
        pim_anchors: tuple[TimingAnchor, ...],
    ) -> None:
        self.metadata = dict(metadata)
        self.gpu_anchors = gpu_anchors
        self.pim_anchors = pim_anchors

    @classmethod
    def load(cls, path: str | Path) -> "RuntimeExpertTimingModel":
        calibration_path = Path(path)
        try:
            raw = json.loads(calibration_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(
                f"runtime timing calibration does not exist: {calibration_path}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid runtime timing calibration JSON: {exc}") from exc
        cls._require_exact_keys(
            raw,
            {
                "schema_version",
                "metadata",
                "gpu_expert_count_anchors",
                "pim_token_count_anchors",
            },
            "runtime timing calibration",
        )
        if isinstance(raw["schema_version"], bool) or raw["schema_version"] != 1:
            raise ValueError(
                f"unsupported runtime timing calibration schema: {raw['schema_version']}"
            )
        metadata = raw["metadata"]
        cls._require_exact_keys(metadata, cls._METADATA_FIELDS, "calibration metadata")
        if metadata["units"] != "us":
            raise ValueError("runtime timing calibration units must be microseconds")
        if metadata["method"] != "piecewise-linear-isolated-v1":
            raise ValueError("unsupported runtime timing calibration method")
        for field in ("source", "scope"):
            if not isinstance(metadata[field], str) or not metadata[field].strip():
                raise ValueError(f"calibration metadata field {field} must be non-empty")
        for field in (
            "model_sha256",
            "hardware_sha256",
            "cycle_config_sha256",
            "source_contention_table_sha256",
            "generator_sha256",
        ):
            value = metadata[field]
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in string.hexdigits for character in value)
            ):
                raise ValueError(f"calibration metadata field {field} must be SHA-256")
        return cls(
            dict(metadata),
            cls._anchors(raw["gpu_expert_count_anchors"], "expert_count"),
            cls._anchors(raw["pim_token_count_anchors"], "token_count"),
        )

    @staticmethod
    def _anchors(value: Any, count_field: str) -> tuple[TimingAnchor, ...]:
        if not isinstance(value, list):
            raise ValueError(f"{count_field} anchors must be an array")
        anchors: list[TimingAnchor] = []
        for row in value:
            RuntimeExpertTimingModel._require_exact_keys(
                row, {count_field, "duration_us"}, f"{count_field} anchor"
            )
            count = row[count_field]
            duration = row["duration_us"]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(f"{count_field} must be a non-negative integer")
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration)
                or duration < 0
            ):
                raise ValueError("anchor duration_us must be non-negative")
            anchors.append(TimingAnchor(count, float(duration)))
        result = tuple(anchors)
        if len(result) < 2:
            raise ValueError(f"{count_field} calibration requires at least two anchors")
        if result[0] != TimingAnchor(0, 0.0):
            raise ValueError(f"{count_field} calibration must start at (0, 0 us)")
        if tuple(anchor.count for anchor in result) != tuple(
            sorted({anchor.count for anchor in result})
        ):
            raise ValueError(f"{count_field} anchors must be sorted and unique")
        if any(
            current.duration_us < previous.duration_us
            for previous, current in zip(result, result[1:])
        ):
            raise ValueError(f"{count_field} anchor durations must be monotonic")
        return result

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
    def _sha256(path: str | Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def validate_inputs(
        self,
        model_path: str | Path,
        hardware_path: str | Path,
        cycle_config_path: str | Path,
        source_contention_table_path: str | Path | None = None,
    ) -> None:
        checks: dict[str, str | Path] = {
            "model_sha256": model_path,
            "hardware_sha256": hardware_path,
            "cycle_config_sha256": cycle_config_path,
        }
        if source_contention_table_path is not None:
            checks["source_contention_table_sha256"] = source_contention_table_path
        for field, path in checks.items():
            if self._sha256(path) != self.metadata[field]:
                raise ValueError(
                    f"runtime timing calibration {field} does not match current input: {path}"
                )

    @staticmethod
    def _interpolate(anchors: tuple[TimingAnchor, ...], count: int) -> float:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("runtime timing prediction count must be non-negative")
        if count <= anchors[0].count:
            lower, upper = anchors[0], anchors[1]
        elif count >= anchors[-1].count:
            lower, upper = anchors[-2], anchors[-1]
        else:
            lower, upper = next(
                (left, right)
                for left, right in zip(anchors, anchors[1:])
                if left.count <= count <= right.count
            )
        fraction = (count - lower.count) / (upper.count - lower.count)
        return lower.duration_us + fraction * (upper.duration_us - lower.duration_us)

    def gpu_expert_read_us(self, expert_count: int) -> float:
        return self._interpolate(self.gpu_anchors, expert_count)

    def pim_expert_pipeline_us(self, token_count: int) -> float:
        return self._interpolate(self.pim_anchors, token_count)

    def gpu_prediction_is_extrapolated(self, expert_count: int) -> bool:
        return expert_count > self.gpu_anchors[-1].count

    def pim_prediction_is_extrapolated(self, token_count: int) -> bool:
        return token_count > self.pim_anchors[-1].count

    def report(self) -> dict[str, object]:
        return {
            "method": self.metadata["method"],
            "source": self.metadata["source"],
            "scope": self.metadata["scope"],
            "gpu_anchor_count": len(self.gpu_anchors),
            "pim_anchor_count": len(self.pim_anchors),
            "nonzero_calibration_points": (
                len(self.gpu_anchors) + len(self.pim_anchors) - 2
            ),
            "gpu_calibrated_range": [
                self.gpu_anchors[0].count,
                self.gpu_anchors[-1].count,
            ],
            "pim_calibrated_range": [
                self.pim_anchors[0].count,
                self.pim_anchors[-1].count,
            ],
        }
