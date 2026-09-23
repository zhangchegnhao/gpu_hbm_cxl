from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..report.writer import sha256_file
from .cxl_kv_workload import _cxl_controller_config
from .cxl_pim_attention import CXLPIMTopology
from .microbenchmark import _load_ramulator
from .mixed_workload import SieveCycleV1Config


@dataclass(frozen=True)
class CXLPIMPipelineResult:
    tick_ps: int
    pim_channels: int
    link_channels: int
    pseudo_channels_per_channel: int
    query_link_transactions: int
    pim_gwrite_waves: int
    pim_mac_waves: int
    pim_read_waves: int
    result_link_transactions: int
    query_link_injected_requests: int
    query_link_completed_requests: int
    query_link_completion_cycles: int
    query_link_rejected_attempts: int
    pim_gwrite_injected_requests: int
    pim_gwrite_completed_requests: int
    pim_gwrite_start_cycles: int
    pim_gwrite_completion_cycles: int
    pim_mac_injected_requests: int
    pim_mac_completed_requests: int
    pim_mac_start_cycles: int
    pim_mac_completion_cycles: int
    pim_read_injected_requests: int
    pim_read_completed_requests: int
    pim_read_start_cycles: int
    pim_read_completion_cycles: int
    result_link_injected_requests: int
    result_link_completed_requests: int
    result_link_start_cycles: int
    result_link_completion_cycles: int
    result_link_rejected_attempts: int
    pipeline_completion_cycles: int
    controller_pim_gwrite_requests: int
    controller_pim_mac_requests: int
    controller_pim_read_requests: int
    controller_link_requests: int
    link_queue_wait_cycles: int
    link_max_queue_wait_cycles: int
    link_request_residence_cycles: int
    link_max_request_residence_cycles: int
    link_busy_cycles: int

    def cycles_to_us(self, cycles: int) -> float:
        return cycles * self.tick_ps / 1_000_000.0

    @property
    def pipeline_completion_us(self) -> float:
        return self.cycles_to_us(self.pipeline_completion_cycles)

    @property
    def query_link_us(self) -> float:
        return self.cycles_to_us(self.query_link_completion_cycles)

    @property
    def pim_gwrite_us(self) -> float:
        return self.cycles_to_us(
            self.pim_gwrite_completion_cycles - self.pim_gwrite_start_cycles
        )

    @property
    def pim_mac_us(self) -> float:
        return self.cycles_to_us(
            self.pim_mac_completion_cycles - self.pim_mac_start_cycles
        )

    @property
    def pim_read_us(self) -> float:
        return self.cycles_to_us(
            self.pim_read_completion_cycles - self.pim_read_start_cycles
        )

    @property
    def result_link_us(self) -> float:
        return self.cycles_to_us(
            self.result_link_completion_cycles - self.result_link_start_cycles
        )


def cxl_pim_pipeline_context(
    project_root: str | Path, cycle_config_path: str | Path
) -> dict[str, str]:
    root = Path(project_root)
    sources = [
        root
        / "ramulator/extensions/sieve_cxl/source/sieve_cxl_pim_pipeline_frontend.cpp",
        root / "ramulator/extensions/sieve_cxl/source/sieve_cxl_memory_controller.cpp",
        root
        / "ramulator/extensions/sieve_hbm_pim/source/sieve_hbm_pim_controller.cpp",
        root
        / "ramulator/extensions/sieve_hbm_pim/patches/ramulator2-cycle-v0.patch",
    ]
    digest = hashlib.sha256()
    for path in sorted(sources):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return {
        "model": "unified-cxl-pim-pipeline-v1",
        "cycle_config_sha256": sha256_file(Path(cycle_config_path)),
        "extension_sha256": digest.hexdigest(),
        "runner_sha256": sha256_file(Path(__file__)),
    }


def _pim_controller_config(
    ramulator: Any,
    cycle: SieveCycleV1Config,
    topology: CXLPIMTopology,
    total_pim_waves: int,
) -> dict[str, Any]:
    dram = ramulator.dram.HBM3(
        org_preset=cycle.dram_org_preset,
        timing_preset=cycle.dram_timing_preset,
    ).to_config()
    dram["org"]["count"] = [
        1,
        topology.pseudo_channels_per_channel,
        cycle.sid_per_pseudo_channel,
        cycle.bank_groups_per_pseudo_channel,
        cycle.banks_per_bank_group,
        cycle.rows,
        cycle.columns,
    ]
    return {
        "impl": "SieveHBMPIM",
        "pim_buffer_size": cycle.pim_buffer_size,
        "pim_mac_interval_ps": cycle.pim_mac_interval_ps,
        "pim_io_interval_ps": cycle.pim_io_interval_ps,
        "pim_milestone_waves": str(total_pim_waves),
        "scheduler": {"impl": "FRFCFS"},
        "refresh_manager": {"impl": "NoRefresh"},
        "row_policy": {"impl": "Open"},
        "addr_mapper": {"impl": "PassThroughAddrMapper"},
        "dram": dram,
    }


def run_cxl_pim_pipeline(
    ramulator_root: str | Path,
    cycle: SieveCycleV1Config,
    topology: CXLPIMTopology,
    *,
    query_link_transactions: int,
    pim_gwrite_waves: int,
    pim_mac_waves: int,
    pim_read_waves: int,
    result_link_transactions: int,
    cxl_bandwidth_bytes_per_second: float,
    cxl_latency_us: float,
    link_channels: int,
) -> CXLPIMPipelineResult:
    counts = {
        "query_link_transactions": query_link_transactions,
        "pim_gwrite_waves": pim_gwrite_waves,
        "pim_mac_waves": pim_mac_waves,
        "pim_read_waves": pim_read_waves,
        "result_link_transactions": result_link_transactions,
    }
    for name, value in counts.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if topology.pseudo_channels_per_channel != cycle.pseudo_channels_per_channel:
        raise ValueError("CXL-PIM topology and cycle PCH counts disagree")
    if topology.banks_per_pseudo_channel != cycle.banks_per_pseudo_channel:
        raise ValueError("CXL-PIM topology and cycle bank counts disagree")
    if cxl_bandwidth_bytes_per_second <= 0 or cxl_latency_us < 0:
        raise ValueError("CXL link bandwidth must be positive and latency nonnegative")
    if isinstance(link_channels, bool) or not isinstance(link_channels, int) or link_channels <= 0:
        raise ValueError("link_channels must be a positive integer")

    root = Path(ramulator_root).resolve()
    ramulator = _load_ramulator(root)
    total_pim_waves = pim_gwrite_waves + pim_mac_waves + pim_read_waves
    pim_controller = _pim_controller_config(
        ramulator, cycle, topology, total_pim_waves
    )
    link_controller = _cxl_controller_config(
        ramulator,
        cycle,
        bandwidth_bytes_per_second=cxl_bandwidth_bytes_per_second,
        access_latency_us=cxl_latency_us,
        cxl_channels=link_channels,
    )
    tick_ps = int(pim_controller["dram"]["timing"][-1])
    if tick_ps != cycle.expected_tick_ps:
        raise ValueError(
            f"Ramulator tick is {tick_ps} ps, expected {cycle.expected_tick_ps} ps"
        )

    frontend = {
        "impl": "SieveCXLPIMPipeline",
        "clock_ratio": 1,
        "pim_channels": topology.channels,
        "link_channels": link_channels,
        "pseudo_channels": topology.pseudo_channels_per_channel,
        "row_span_waves": cycle.row_span_waves,
        **counts,
    }
    memory_system = {
        "impl": "GenericDRAM",
        "clock_ratio": 1,
        "controllers": [pim_controller for _ in range(topology.channels)]
        + [link_controller for _ in range(link_channels)],
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
    pim_stats = controller_stats[: topology.channels]
    link_stats = controller_stats[topology.channels :]
    endpoints = topology.total_pseudo_channels
    expected = {
        "query_link": query_link_transactions,
        "pim_gwrite": pim_gwrite_waves * endpoints,
        "pim_mac": pim_mac_waves * endpoints,
        "pim_read": pim_read_waves * endpoints,
        "result_link": result_link_transactions,
    }
    for phase, phase_expected in expected.items():
        injected = int(frontend_stats[f"{phase}_injected_requests"])
        completed = int(frontend_stats[f"{phase}_completed_requests"])
        if injected != phase_expected or completed != phase_expected:
            raise RuntimeError(
                f"incomplete {phase} phase: injected={injected}, "
                f"completed={completed}, expected={phase_expected}"
            )

    controller_pim = {
        operation: sum(int(row[f"num_pim_{operation}_reqs"]) for row in pim_stats)
        for operation in ("gwrite", "mac", "read")
    }
    for operation in ("gwrite", "mac", "read"):
        if controller_pim[operation] != expected[f"pim_{operation}"]:
            raise RuntimeError(f"PIM controller {operation} count mismatch")
    controller_link = sum(
        int(row.get("cxl_requests_served", 0)) for row in link_stats
    )
    if controller_link != query_link_transactions + result_link_transactions:
        raise RuntimeError("CXL link controller request count mismatch")

    def total(key: str) -> int:
        return sum(int(row.get(key, 0)) for row in link_stats)

    def maximum(key: str) -> int:
        return max((int(row.get(key, 0)) for row in link_stats), default=0)

    fields = (
        "query_link_injected_requests",
        "query_link_completed_requests",
        "query_link_completion_cycles",
        "query_link_rejected_attempts",
        "pim_gwrite_injected_requests",
        "pim_gwrite_completed_requests",
        "pim_gwrite_start_cycles",
        "pim_gwrite_completion_cycles",
        "pim_mac_injected_requests",
        "pim_mac_completed_requests",
        "pim_mac_start_cycles",
        "pim_mac_completion_cycles",
        "pim_read_injected_requests",
        "pim_read_completed_requests",
        "pim_read_start_cycles",
        "pim_read_completion_cycles",
        "result_link_injected_requests",
        "result_link_completed_requests",
        "result_link_start_cycles",
        "result_link_completion_cycles",
        "result_link_rejected_attempts",
        "pipeline_completion_cycles",
    )
    return CXLPIMPipelineResult(
        tick_ps=tick_ps,
        pim_channels=topology.channels,
        link_channels=link_channels,
        pseudo_channels_per_channel=topology.pseudo_channels_per_channel,
        **counts,
        **{name: int(frontend_stats[name]) for name in fields},
        controller_pim_gwrite_requests=controller_pim["gwrite"],
        controller_pim_mac_requests=controller_pim["mac"],
        controller_pim_read_requests=controller_pim["read"],
        controller_link_requests=controller_link,
        link_queue_wait_cycles=total("cxl_queue_wait_cycles"),
        link_max_queue_wait_cycles=maximum("cxl_max_queue_wait_cycles"),
        link_request_residence_cycles=total("cxl_request_residence_cycles"),
        link_max_request_residence_cycles=maximum(
            "cxl_max_request_residence_cycles"
        ),
        link_busy_cycles=total("cxl_link_busy_cycles"),
    )
