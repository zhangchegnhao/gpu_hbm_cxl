from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..report.writer import sha256_file
from .cxl_kv_workload import _cxl_controller_config
from .cxl_pim_attention import CXLPIMTopology
from .cxl_pim_pipeline import _pim_controller_config
from .microbenchmark import _load_ramulator
from .mixed_workload import SieveCycleV1Config


_VECTOR_FIELDS = (
    "chunk_mac_waves",
    "chunk_mac_injected_requests",
    "chunk_mac_completed_requests",
    "chunk_mac_start_cycles",
    "chunk_mac_completion_cycles",
    "chunk_read_injected_requests",
    "chunk_read_completed_requests",
    "chunk_read_start_cycles",
    "chunk_read_completion_cycles",
    "chunk_result_link_injected_requests",
    "chunk_result_link_completed_requests",
    "chunk_result_link_start_cycles",
    "chunk_result_link_completion_cycles",
)


@dataclass(frozen=True)
class CXLPIMChunkPipelineResult:
    tick_ps: int
    pim_channels: int
    link_channels: int
    pseudo_channels_per_channel: int
    chunk_count: int
    buffer_slots: int
    query_link_transactions: int
    pim_gwrite_waves: int
    pim_mac_waves: int
    pim_read_waves_per_chunk: int
    result_link_transactions_per_chunk: int
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
    pim_read_injected_requests: int
    pim_read_completed_requests: int
    result_link_injected_requests: int
    result_link_completed_requests: int
    result_link_rejected_attempts: int
    pipeline_completion_cycles: int
    max_active_buffers: int
    buffer_stall_cycles: int
    chunk_mac_waves: tuple[int, ...]
    chunk_mac_injected_requests: tuple[int, ...]
    chunk_mac_completed_requests: tuple[int, ...]
    chunk_mac_start_cycles: tuple[int, ...]
    chunk_mac_completion_cycles: tuple[int, ...]
    chunk_read_injected_requests: tuple[int, ...]
    chunk_read_completed_requests: tuple[int, ...]
    chunk_read_start_cycles: tuple[int, ...]
    chunk_read_completion_cycles: tuple[int, ...]
    chunk_result_link_injected_requests: tuple[int, ...]
    chunk_result_link_completed_requests: tuple[int, ...]
    chunk_result_link_start_cycles: tuple[int, ...]
    chunk_result_link_completion_cycles: tuple[int, ...]
    controller_pim_gwrite_requests: int
    controller_pim_mac_requests: int
    controller_pim_read_requests: int
    controller_link_requests: int
    link_queue_wait_cycles: int
    link_max_queue_wait_cycles: int
    link_request_residence_cycles: int
    link_max_request_residence_cycles: int
    link_busy_cycles: int

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CXLPIMChunkPipelineResult":
        values = dict(raw)
        for name in _VECTOR_FIELDS:
            values[name] = tuple(int(value) for value in values[name])
        return cls(**values)

    def cycles_to_us(self, cycles: int) -> float:
        return cycles * self.tick_ps / 1_000_000.0

    @property
    def pipeline_completion_us(self) -> float:
        return self.cycles_to_us(self.pipeline_completion_cycles)

    @property
    def partial_ready_us(self) -> tuple[float, ...]:
        return tuple(
            self.cycles_to_us(value)
            for value in self.chunk_result_link_completion_cycles
        )


def cxl_pim_chunk_pipeline_context(
    project_root: str | Path, cycle_config_path: str | Path
) -> dict[str, str]:
    root = Path(project_root)
    sources = [
        root
        / "ramulator/extensions/sieve_cxl/source/sieve_cxl_pim_chunk_pipeline_frontend.cpp",
        root / "ramulator/extensions/sieve_cxl/source/sieve_cxl_memory_controller.cpp",
        root
        / "ramulator/extensions/sieve_hbm_pim/source/sieve_hbm_pim_controller.cpp",
        root
        / "ramulator/extensions/sieve_hbm_pim/patches/ramulator2-cycle-v0.patch",
        root / "src/sieve_replay/ramulator/cxl_pim_pipeline.py",
        Path(__file__),
    ]
    digest = hashlib.sha256()
    for path in sorted(sources):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return {
        "model": "unified-cxl-pim-chunk-pipeline-v1",
        "cycle_config_sha256": sha256_file(Path(cycle_config_path)),
        "extension_sha256": digest.hexdigest(),
        "runner_sha256": sha256_file(Path(__file__)),
    }


def run_cxl_pim_chunk_pipeline(
    ramulator_root: str | Path,
    cycle: SieveCycleV1Config,
    topology: CXLPIMTopology,
    *,
    chunk_count: int,
    buffer_slots: int,
    query_link_transactions: int,
    pim_gwrite_waves: int,
    pim_mac_waves: int,
    pim_read_waves_per_chunk: int,
    result_link_transactions_per_chunk: int,
    cxl_bandwidth_bytes_per_second: float,
    cxl_latency_us: float,
    link_channels: int,
) -> CXLPIMChunkPipelineResult:
    counts = {
        "chunk_count": chunk_count,
        "buffer_slots": buffer_slots,
        "query_link_transactions": query_link_transactions,
        "pim_gwrite_waves": pim_gwrite_waves,
        "pim_mac_waves": pim_mac_waves,
        "pim_read_waves_per_chunk": pim_read_waves_per_chunk,
        "result_link_transactions_per_chunk": result_link_transactions_per_chunk,
    }
    for name, value in counts.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if buffer_slots > chunk_count:
        raise ValueError("buffer_slots cannot exceed chunk_count")
    if pim_mac_waves < chunk_count:
        raise ValueError("each chunk must contain at least one PIM MAC wave")
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
    total_pim_waves = (
        pim_gwrite_waves
        + pim_mac_waves
        + chunk_count * pim_read_waves_per_chunk
    )
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
        "impl": "SieveCXLPIMChunkPipeline",
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
        "pim_read": chunk_count * pim_read_waves_per_chunk * endpoints,
        "result_link": chunk_count * result_link_transactions_per_chunk,
    }
    for phase, phase_expected in expected.items():
        injected = int(frontend_stats[f"{phase}_injected_requests"])
        completed = int(frontend_stats[f"{phase}_completed_requests"])
        if injected != phase_expected or completed != phase_expected:
            raise RuntimeError(
                f"incomplete {phase} phase: injected={injected}, "
                f"completed={completed}, expected={phase_expected}"
            )

    vector_values = {
        name: tuple(int(value) for value in frontend_stats[name])
        for name in _VECTOR_FIELDS
    }
    if any(len(values) != chunk_count for values in vector_values.values()):
        raise RuntimeError("chunk milestone vector length mismatch")
    if sum(vector_values["chunk_mac_waves"]) != pim_mac_waves:
        raise RuntimeError("chunk MAC waves do not preserve the full workload")
    for chunk in range(chunk_count):
        expected_mac = vector_values["chunk_mac_waves"][chunk] * endpoints
        expected_read = pim_read_waves_per_chunk * endpoints
        expected_result = result_link_transactions_per_chunk
        per_chunk_expected = {
            "chunk_mac_injected_requests": expected_mac,
            "chunk_mac_completed_requests": expected_mac,
            "chunk_read_injected_requests": expected_read,
            "chunk_read_completed_requests": expected_read,
            "chunk_result_link_injected_requests": expected_result,
            "chunk_result_link_completed_requests": expected_result,
        }
        for name, value in per_chunk_expected.items():
            if vector_values[name][chunk] != value:
                raise RuntimeError(f"chunk {chunk} request count mismatch for {name}")
        milestones = (
            vector_values["chunk_mac_start_cycles"][chunk],
            vector_values["chunk_mac_completion_cycles"][chunk],
            vector_values["chunk_read_start_cycles"][chunk],
            vector_values["chunk_read_completion_cycles"][chunk],
            vector_values["chunk_result_link_start_cycles"][chunk],
            vector_values["chunk_result_link_completion_cycles"][chunk],
        )
        if milestones != tuple(sorted(milestones)):
            raise RuntimeError(f"chunk {chunk} phase dependency mismatch")
    if tuple(sorted(vector_values["chunk_result_link_completion_cycles"])) != vector_values[
        "chunk_result_link_completion_cycles"
    ]:
        raise RuntimeError("partial results must complete in chunk order")
    if int(frontend_stats["pipeline_completion_cycles"]) != vector_values[
        "chunk_result_link_completion_cycles"
    ][-1]:
        raise RuntimeError("pipeline completion does not match final partial result")

    controller_pim = {
        operation: sum(int(row[f"num_pim_{operation}_reqs"]) for row in pim_stats)
        for operation in ("gwrite", "mac", "read")
    }
    for operation in ("gwrite", "mac", "read"):
        if controller_pim[operation] != expected[f"pim_{operation}"]:
            raise RuntimeError(f"PIM controller {operation} count mismatch")
    controller_link = sum(int(row.get("cxl_requests_served", 0)) for row in link_stats)
    if controller_link != expected["query_link"] + expected["result_link"]:
        raise RuntimeError("CXL link controller request count mismatch")

    def total(key: str) -> int:
        return sum(int(row.get(key, 0)) for row in link_stats)

    def maximum(key: str) -> int:
        return max((int(row.get(key, 0)) for row in link_stats), default=0)

    scalar_fields = (
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
        "pim_read_injected_requests",
        "pim_read_completed_requests",
        "result_link_injected_requests",
        "result_link_completed_requests",
        "result_link_rejected_attempts",
        "pipeline_completion_cycles",
        "max_active_buffers",
        "buffer_stall_cycles",
    )
    return CXLPIMChunkPipelineResult(
        tick_ps=tick_ps,
        pim_channels=topology.channels,
        link_channels=link_channels,
        pseudo_channels_per_channel=topology.pseudo_channels_per_channel,
        **counts,
        **{name: int(frontend_stats[name]) for name in scalar_fields},
        **vector_values,
        controller_pim_gwrite_requests=controller_pim["gwrite"],
        controller_pim_mac_requests=controller_pim["mac"],
        controller_pim_read_requests=controller_pim["read"],
        controller_link_requests=controller_link,
        link_queue_wait_cycles=total("cxl_queue_wait_cycles"),
        link_max_queue_wait_cycles=maximum("cxl_max_queue_wait_cycles"),
        link_request_residence_cycles=total("cxl_request_residence_cycles"),
        link_max_request_residence_cycles=maximum("cxl_max_request_residence_cycles"),
        link_busy_cycles=total("cxl_link_busy_cycles"),
    )
