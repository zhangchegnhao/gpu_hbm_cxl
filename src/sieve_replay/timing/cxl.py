from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import HardwareConfig


@dataclass(frozen=True)
class CXLReadConfig:
    """Configuration for the stage-3 memory-only CXL KV READ model."""

    mode: str = "disabled"
    local_capacity_bytes: int = 0
    capacity_bytes: int = 0
    bandwidth_bytes_per_second: float = 0.0
    latency_us: float = 0.0

    @classmethod
    def from_raw(cls, raw: dict[str, Any], hardware: HardwareConfig) -> "CXLReadConfig":
        mode = raw.get("cxl_mode", "disabled")
        if mode not in {"disabled", "memory-only-v1"}:
            raise ValueError("cxl_mode must be disabled or memory-only-v1")
        local_gb = raw.get("cxl_local_capacity_gb", hardware.hbm_pim_capacity_gb)
        capacity_gb = raw.get("cxl_capacity_gb", 0.0)
        bandwidth_gb_s = raw.get("cxl_bandwidth_gb_s", 0.0)
        latency_us = raw.get("cxl_latency_us", 0.0)
        values = (local_gb, capacity_gb, bandwidth_gb_s, latency_us)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise ValueError("CXL capacity, bandwidth and latency fields must be numeric")
        if local_gb <= 0 or capacity_gb < 0 or bandwidth_gb_s < 0 or latency_us < 0:
            raise ValueError("CXL capacity, bandwidth and latency fields must be nonnegative")
        if mode == "memory-only-v1" and capacity_gb > 0 and bandwidth_gb_s <= 0:
            raise ValueError("memory-only-v1 requires positive cxl_bandwidth_gb_s")
        return cls(
            mode=str(mode),
            local_capacity_bytes=int(local_gb * 1_000_000_000),
            capacity_bytes=int(capacity_gb * 1_000_000_000),
            bandwidth_bytes_per_second=float(bandwidth_gb_s) * 1_000_000_000,
            latency_us=float(latency_us),
        )

    @property
    def enabled(self) -> bool:
        return self.mode == "memory-only-v1"
