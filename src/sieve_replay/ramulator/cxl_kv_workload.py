from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..report.writer import sha256_file
from .microbenchmark import _load_ramulator
from .mixed_workload import SieveCycleV1Config, _controller_config


@dataclass(frozen=True)
class CXLKVWorkloadResult:
    tick_ps: int
    local_channels: int
    cxl_channels: int
    gpu_read_transactions: int
    local_kv_read_transactions: int
    cxl_kv_read_transactions: int
    pim_gwrite_waves: int
    pim_mac_waves: int
    pim_read_waves: int
    gpu_injected_requests: int
    gpu_completed_requests: int
    local_kv_injected_requests: int
    local_kv_completed_requests: int
    cxl_kv_injected_requests: int
    cxl_kv_completed_requests: int
    pim_injected_requests: int
    pim_completed_requests: int
    gpu_completion_cycles: int
    local_kv_completion_cycles: int
    cxl_kv_completion_cycles: int
    pim_completion_cycles: int
    total_completion_cycles: int
    local_read_latency_cycles: int
    cxl_read_latency_cycles: int
    cxl_queue_wait_cycles: int
    cxl_max_queue_wait_cycles: int
    cxl_request_residence_cycles: int
    cxl_max_request_residence_cycles: int
    cxl_link_busy_cycles: int
    cxl_controller_completion_cycles: int
    gpu_injection_rejected_attempts: int
    local_kv_injection_rejected_attempts: int
    cxl_kv_injection_rejected_attempts: int

    @property
    def total_completion_us(self) -> float:
        return self.total_completion_cycles * self.tick_ps / 1_000_000.0

    @property
    def cxl_queue_wait_us(self) -> float:
        return self.cxl_queue_wait_cycles * self.tick_ps / 1_000_000.0

    @property
    def cxl_read_latency_us(self) -> float:
        return self.cxl_read_latency_cycles * self.tick_ps / 1_000_000.0

    @property
    def cxl_link_busy_ratio(self) -> float:
        denominator = self.total_completion_cycles * self.cxl_channels
        return self.cxl_link_busy_cycles / denominator if denominator else 0.0


def cxl_kv_cache_context(project_root: str | Path, cycle_config_path: str | Path) -> dict[str, str]:
    root = Path(project_root)
    sources = [
        root / "ramulator/extensions/sieve_cxl/source/sieve_cxl_kv_frontend.cpp",
        root / "ramulator/extensions/sieve_cxl/source/sieve_cxl_memory_controller.cpp",
        root / "ramulator/extensions/sieve_hbm_pim/source/sieve_hbm_pim_controller_v1.cpp",
        root / "ramulator/extensions/sieve_hbm_pim/patches/ramulator2-cycle-v1.patch",
    ]
    digest = hashlib.sha256()
    for path in sorted(sources):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return {
        "model": "cxl-kv-memory-only-request-v1",
        "cycle_config_sha256": sha256_file(Path(cycle_config_path)),
        "extension_sha256": digest.hexdigest(),
        "cxl_kv_workload_sha256": sha256_file(Path(__file__)),
        "mixed_workload_sha256": sha256_file(root / "src/sieve_replay/ramulator/mixed_workload.py"),
    }


def _cxl_controller_config(
    ramulator: Any,
    cycle: SieveCycleV1Config,
    *,
    bandwidth_bytes_per_second: float,
    access_latency_us: float,
    cxl_channels: int,
) -> dict[str, Any]:
    dram = ramulator.dram.HBM3(
        org_preset=cycle.dram_org_preset,
        timing_preset=cycle.dram_timing_preset,
    ).to_config()
    dram["org"]["count"] = cycle.organization_count
    if bandwidth_bytes_per_second <= 0 or access_latency_us < 0 or cxl_channels <= 0:
        raise ValueError("CXL bandwidth, channel count and latency are invalid")
    return {
        "impl": "SieveCXLMemory",
        "read_buffer_size": cycle.read_buffer_size,
        "cxl_bandwidth_bytes_per_second": bandwidth_bytes_per_second / cxl_channels,
        "cxl_access_latency_ps": int(access_latency_us * 1_000_000),
        "scheduler": {"impl": "FRFCFS"},
        "refresh_manager": {"impl": "NoRefresh"},
        "row_policy": {"impl": "Open"},
        "addr_mapper": {"impl": "PassThroughAddrMapper"},
        "dram": dram,
    }


def run_cxl_kv_workload(
    ramulator_root: str | Path,
    cycle: SieveCycleV1Config,
    *,
    gpu_read_transactions: int,
    local_kv_read_transactions: int,
    cxl_kv_read_transactions: int,
    pim_gwrite_waves: int,
    pim_mac_waves: int,
    pim_read_waves: int,
    cxl_bandwidth_bytes_per_second: float,
    cxl_latency_us: float,
    cxl_channels: int = 1,
) -> CXLKVWorkloadResult:
    values = {
        "gpu_read_transactions": gpu_read_transactions,
        "local_kv_read_transactions": local_kv_read_transactions,
        "cxl_kv_read_transactions": cxl_kv_read_transactions,
        "pim_gwrite_waves": pim_gwrite_waves,
        "pim_mac_waves": pim_mac_waves,
        "pim_read_waves": pim_read_waves,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if cxl_bandwidth_bytes_per_second <= 0 or cxl_latency_us < 0:
        raise ValueError("CXL bandwidth must be positive and latency nonnegative")
    if cxl_channels <= 0:
        raise ValueError("cxl_channels must be positive")
    if not any(values.values()):
        raise ValueError("CXL KV workload must contain at least one request")

    root = Path(ramulator_root).resolve()
    ramulator = _load_ramulator(root)
    local_controller = _controller_config(ramulator, cycle)
    cxl_controller = _cxl_controller_config(
        ramulator,
        cycle,
        bandwidth_bytes_per_second=cxl_bandwidth_bytes_per_second,
        access_latency_us=cxl_latency_us,
        cxl_channels=cxl_channels,
    )
    tick_ps = int(local_controller["dram"]["timing"][-1])
    if tick_ps != cycle.expected_tick_ps:
        raise ValueError(f"Ramulator tick is {tick_ps} ps, expected {cycle.expected_tick_ps} ps")

    local_channels = cycle.total_channels
    frontend = {
        "impl": "SieveCXLKVRead",
        "clock_ratio": 1,
        "local_channels": local_channels,
        "cxl_channels": cxl_channels,
        "pseudo_channels": cycle.pseudo_channels_per_channel,
        "banks_per_pseudo_channel": cycle.banks_per_pseudo_channel,
        "banks_per_bank_group": cycle.banks_per_bank_group,
        "rows": cycle.rows,
        "gpu_cachelines_per_row": cycle.gpu_cachelines_per_row,
        "internal_prefetch_size": cycle.internal_prefetch_size,
        "row_span_waves": cycle.row_span_waves,
        "tick_ps": tick_ps,
        "pim_mac_interval_ps": cycle.pim_mac_interval_ps,
        "pim_io_interval_ps": cycle.pim_io_interval_ps,
        **values,
    }
    memory_system = {
        "impl": "GenericDRAM",
        "clock_ratio": 1,
        "controllers": [local_controller for _ in range(local_channels)]
        + [cxl_controller for _ in range(cxl_channels)],
        "channel_mapper": {"impl": "PassThroughChannelMapper"},
    }
    simulation = ramulator.Simulation(frontend, memory_system)
    simulation.run()
    stats = simulation.stats
    simulation.finalize()

    frontend_stats = stats["frontend"]
    controller_stats = stats["memory_system"]["controller"]
    if not isinstance(controller_stats, list):
        controller_stats = [controller_stats]
    local_stats = controller_stats[:local_channels]
    cxl_stats = controller_stats[local_channels:]
    expected_pim = (pim_gwrite_waves + pim_mac_waves + pim_read_waves) * cycle.total_pseudo_channels
    expected_local = gpu_read_transactions + local_kv_read_transactions
    if int(frontend_stats["gpu_injected_requests"]) != gpu_read_transactions or int(frontend_stats["gpu_completed_requests"]) != gpu_read_transactions:
        raise RuntimeError("incomplete GPU stream")
    if int(frontend_stats["local_kv_injected_requests"]) != local_kv_read_transactions or int(frontend_stats["local_kv_completed_requests"]) != local_kv_read_transactions:
        raise RuntimeError("incomplete local KV stream")
    if int(frontend_stats["cxl_kv_injected_requests"]) != cxl_kv_read_transactions or int(frontend_stats["cxl_kv_completed_requests"]) != cxl_kv_read_transactions:
        raise RuntimeError("incomplete CXL KV stream")
    if sum(int(row["num_read_reqs_served"]) for row in local_stats) != expected_local:
        raise RuntimeError("local controller read counts do not match frontend counts")
    if sum(int(row.get("cxl_requests_served", 0)) for row in cxl_stats) != cxl_kv_read_transactions:
        raise RuntimeError("CXL controller read counts do not match frontend counts")
    if int(frontend_stats["pim_injected_requests"]) != expected_pim or int(frontend_stats["pim_completed_requests"]) != expected_pim:
        raise RuntimeError("incomplete PIM stream")

    def total(key: str, rows: list[dict[str, Any]]) -> int:
        return sum(int(row.get(key, 0)) for row in rows)

    gpu_completion = int(frontend_stats["gpu_completion_cycles"])
    local_kv_completion = int(frontend_stats["local_kv_completion_cycles"])
    cxl_completion = int(frontend_stats["cxl_kv_completion_cycles"])
    pim_completion = int(frontend_stats["pim_completion_cycles"])
    total_completion = max(gpu_completion, local_kv_completion, cxl_completion, pim_completion)
    return CXLKVWorkloadResult(
        tick_ps=tick_ps,
        local_channels=local_channels,
        cxl_channels=cxl_channels,
        **values,
        gpu_injected_requests=int(frontend_stats["gpu_injected_requests"]),
        gpu_completed_requests=int(frontend_stats["gpu_completed_requests"]),
        local_kv_injected_requests=int(frontend_stats["local_kv_injected_requests"]),
        local_kv_completed_requests=int(frontend_stats["local_kv_completed_requests"]),
        cxl_kv_injected_requests=int(frontend_stats["cxl_kv_injected_requests"]),
        cxl_kv_completed_requests=int(frontend_stats["cxl_kv_completed_requests"]),
        pim_injected_requests=int(frontend_stats["pim_injected_requests"]),
        pim_completed_requests=int(frontend_stats["pim_completed_requests"]),
        gpu_completion_cycles=gpu_completion,
        local_kv_completion_cycles=local_kv_completion,
        cxl_kv_completion_cycles=cxl_completion,
        pim_completion_cycles=pim_completion,
        total_completion_cycles=total_completion,
        local_read_latency_cycles=total("read_latency", local_stats),
        cxl_read_latency_cycles=total("read_latency", cxl_stats),
        cxl_queue_wait_cycles=total("cxl_queue_wait_cycles", cxl_stats),
        cxl_max_queue_wait_cycles=max((int(row.get("cxl_max_queue_wait_cycles", 0)) for row in cxl_stats), default=0),
        cxl_request_residence_cycles=total("cxl_request_residence_cycles", cxl_stats),
        cxl_max_request_residence_cycles=max((int(row.get("cxl_max_request_residence_cycles", 0)) for row in cxl_stats), default=0),
        cxl_link_busy_cycles=total("cxl_link_busy_cycles", cxl_stats),
        cxl_controller_completion_cycles=max((int(row.get("cxl_completion_cycles", 0)) for row in cxl_stats), default=0),
        gpu_injection_rejected_attempts=int(frontend_stats["gpu_injection_rejected_attempts"]),
        local_kv_injection_rejected_attempts=int(frontend_stats["local_kv_injection_rejected_attempts"]),
        cxl_kv_injection_rejected_attempts=int(frontend_stats["cxl_kv_injection_rejected_attempts"]),
    )
