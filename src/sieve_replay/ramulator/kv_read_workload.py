from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

from .microbenchmark import _load_ramulator
from .mixed_workload import MixedWorkloadResult, SieveCycleV1Config, _controller_config
from ..report.writer import sha256_file


@dataclass(frozen=True)
class KVWorkloadShape:
    gpu_read_transactions: int
    pim_gwrite_waves: int
    pim_mac_waves: int
    pim_read_waves: int
    kv_read_transactions: int = 0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


def kv_read_cache_context(
    project_root: str | Path, cycle_config_path: str | Path
) -> dict[str, str]:
    root = Path(project_root)
    sources = [
        root / "ramulator/extensions/sieve_kv_read/source/sieve_kv_read_frontend.cpp",
        root / "ramulator/extensions/sieve_hbm_pim/source/sieve_hbm_pim_controller_v1.cpp",
        root / "ramulator/extensions/sieve_hbm_pim/patches/ramulator2-cycle-v1.patch",
    ]
    digest = hashlib.sha256()
    for path in sorted(sources):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return {
        "model": "kv-read-concurrent-stress-v1",
        "cycle_config_sha256": sha256_file(Path(cycle_config_path)),
        "extension_sha256": digest.hexdigest(),
        "kv_read_workload_sha256": sha256_file(Path(__file__)),
        "mixed_workload_sha256": sha256_file(root / "src/sieve_replay/ramulator/mixed_workload.py"),
        "microbenchmark_sha256": sha256_file(root / "src/sieve_replay/ramulator/microbenchmark.py"),
    }


@dataclass(frozen=True)
class KVReadWorkloadResult(MixedWorkloadResult):
    kv_read_transactions: int = 0
    kv_injected_requests: int = 0
    kv_completed_requests: int = 0
    kv_completion_cycles: int = 0
    controller_read_completed_requests: int = 0
    read_latency_cycles: int = 0
    gpu_request_residence_cycles: int = 0
    kv_request_residence_cycles: int = 0
    gpu_max_request_residence_cycles: int = 0
    kv_max_request_residence_cycles: int = 0
    gpu_injection_rejected_attempts: int = 0
    kv_injection_rejected_attempts: int = 0

    @property
    def gpu_accepted_to_column_issue_cycles(self) -> int:
        return self.gpu_request_residence_cycles - self.gpu_completed_requests * self.read_latency_cycles

    @property
    def kv_accepted_to_column_issue_cycles(self) -> int:
        return self.kv_request_residence_cycles - self.kv_completed_requests * self.read_latency_cycles

    @property
    def kv_completion_us(self) -> float:
        return self.kv_completion_cycles * self.tick_ps / 1_000_000.0


def run_kv_read_workload(
    ramulator_root: str | Path,
    cycle: SieveCycleV1Config,
    *,
    gpu_read_transactions: int,
    pim_gwrite_waves: int,
    pim_mac_waves: int,
    pim_read_waves: int,
    kv_read_transactions: int = 0,
) -> KVReadWorkloadResult:
    values = {
        "gpu_read_transactions": gpu_read_transactions,
        "pim_gwrite_waves": pim_gwrite_waves,
        "pim_mac_waves": pim_mac_waves,
        "pim_read_waves": pim_read_waves,
        "kv_read_transactions": kv_read_transactions,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if not any(values.values()):
        raise ValueError("KV workload must contain Expert/PIM work or KV reads")

    ramulator = _load_ramulator(Path(ramulator_root).resolve())
    controller = _controller_config(ramulator, cycle)
    tick_ps = int(controller["dram"]["timing"][-1])
    read_latency_cycles = int(controller["dram"]["read_latency"])
    if tick_ps != cycle.expected_tick_ps:
        raise ValueError(f"Ramulator tick is {tick_ps} ps, expected {cycle.expected_tick_ps} ps")

    frontend = {
        "impl": "SieveKVRead",
        "clock_ratio": 1,
        "channels": cycle.total_channels,
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
        "controllers": [controller for _ in range(cycle.total_channels)],
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
    if "kv_injected_requests" not in frontend_stats:
        raise RuntimeError("Ramulator binding lacks KV READ counters; rebuild SieveKVRead")

    expected_pim = (
        pim_gwrite_waves + pim_mac_waves + pim_read_waves
    ) * cycle.total_pseudo_channels
    expected_normal = gpu_read_transactions + kv_read_transactions
    gpu_injected = int(frontend_stats["gpu_injected_requests"])
    gpu_completed = int(frontend_stats["gpu_completed_requests"])
    pim_injected = int(frontend_stats["pim_injected_requests"])
    pim_completed = int(frontend_stats["pim_completed_requests"])
    controller_gpu_completed = sum(int(row["num_read_reqs_served"]) for row in controller_stats)
    controller_pim_completed = sum(
        int(row["num_pim_gwrite_reqs"])
        + int(row["num_pim_mac_reqs"])
        + int(row["num_pim_read_reqs"])
        for row in controller_stats
    )
    if (
        gpu_injected != gpu_read_transactions
        or gpu_completed != gpu_read_transactions
        or controller_gpu_completed != expected_normal
    ):
        raise RuntimeError(
            "incomplete mixed GPU stream: "
            f"injected={gpu_injected}, frontend_completed={gpu_completed}, "
            f"controller_completed={controller_gpu_completed}, expected={expected_normal}"
        )
    if pim_injected != expected_pim or pim_completed != expected_pim or controller_pim_completed != expected_pim:
        raise RuntimeError(
            "incomplete mixed PIM stream: "
            f"injected={pim_injected}, frontend_completed={pim_completed}, "
            f"controller_completed={controller_pim_completed}, expected={expected_pim}"
        )

    gpu_completion = int(frontend_stats["gpu_completion_cycles"])
    pim_completion = int(frontend_stats["pim_completion_cycles"])
    if gpu_read_transactions > 0 and gpu_completion <= 0:
        raise RuntimeError("mixed workload did not record GPU completion")
    kv_completion = int(frontend_stats["kv_completion_cycles"])
    kv_injected = int(frontend_stats["kv_injected_requests"])
    kv_completed = int(frontend_stats["kv_completed_requests"])
    if kv_injected != kv_read_transactions or kv_completed != kv_read_transactions:
        raise RuntimeError(
            "incomplete mixed KV stream: "
            f"injected={kv_injected}, completed={kv_completed}, expected={kv_read_transactions}"
        )
    if kv_read_transactions > 0 and kv_completion <= 0:
        raise RuntimeError("mixed workload did not record KV completion")
    if expected_pim > 0 and pim_completion <= 0:
        raise RuntimeError("mixed workload did not record PIM completion")

    def total(key: str) -> int:
        return sum(int(row[key]) for row in controller_stats)

    gpu_residence = int(frontend_stats["gpu_request_residence_cycles"])
    kv_residence = int(frontend_stats["kv_request_residence_cycles"])
    if gpu_residence + kv_residence != total("read_latency"):
        raise RuntimeError("KV/Expert residence counters do not match the controller read latency sum")
    if (gpu_residence < gpu_completed * read_latency_cycles
            or kv_residence < kv_completed * read_latency_cycles):
        raise RuntimeError("normal READ residence is shorter than DRAM service latency")

    return KVReadWorkloadResult(
        tick_ps=tick_ps,
        gpu_read_transactions=gpu_read_transactions,
        pim_gwrite_waves=pim_gwrite_waves,
        pim_mac_waves=pim_mac_waves,
        pim_read_waves=pim_read_waves,
        gpu_completion_cycles=gpu_completion,
        pim_completion_cycles=pim_completion,
        pim_gwrite_completion_cycles=int(frontend_stats["pim_gwrite_completion_cycles"]),
        pim_mac_completion_cycles=int(frontend_stats["pim_mac_completion_cycles"]),
        pim_read_completion_cycles=int(frontend_stats["pim_read_completion_cycles"]),
        total_completion_cycles=max(gpu_completion, kv_completion, pim_completion),
        gpu_injected_requests=gpu_injected,
        gpu_completed_requests=gpu_completed,
        pim_injected_requests=pim_injected,
        pim_completed_requests=pim_completed,
        gpu_blocked_by_pim_cycles=total("gpu_blocked_by_pim_cycles"),
        pim_blocked_by_gpu_cycles=total("pim_blocked_by_gpu_cycles"),
        pim_queue_wait_cycles=total("pim_queue_wait_cycles"),
        gpu_column_issues=total("gpu_column_issues"),
        pim_column_issues=total("pim_column_issues"),
        pim_row_activations=total("pim_row_activations"),
        pim_row_conflicts=total("pim_row_conflicts"),
        kv_read_transactions=kv_read_transactions,
        kv_injected_requests=kv_injected,
        kv_completed_requests=kv_completed,
        kv_completion_cycles=kv_completion,
        controller_read_completed_requests=controller_gpu_completed,
        read_latency_cycles=read_latency_cycles,
        gpu_request_residence_cycles=gpu_residence,
        kv_request_residence_cycles=kv_residence,
        gpu_max_request_residence_cycles=int(frontend_stats["gpu_max_request_residence_cycles"]),
        kv_max_request_residence_cycles=int(frontend_stats["kv_max_request_residence_cycles"]),
        gpu_injection_rejected_attempts=int(frontend_stats["gpu_injection_rejected_attempts"]),
        kv_injection_rejected_attempts=int(frontend_stats["kv_injection_rejected_attempts"]),
    )
