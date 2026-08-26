from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..timing.ramulator_table import PINNED_RAMULATOR_COMMIT
from .microbenchmark import _load_ramulator


@dataclass(frozen=True)
class SieveCycleV1Config:
    ramulator_commit: str
    dram_org_preset: str
    dram_timing_preset: str
    hbm_pim_stacks: int
    channels_per_stack: int
    pseudo_channels_per_channel: int
    sid_per_pseudo_channel: int
    bank_groups_per_pseudo_channel: int
    banks_per_bank_group: int
    rows: int
    columns: int
    transaction_bytes: int
    internal_prefetch_size: int
    gpu_cachelines_per_row: int
    read_buffer_size: int
    pim_mac_interval_ps: int
    pim_io_interval_ps: int
    pim_buffer_size: int
    row_span_waves: int
    expected_tick_ps: int
    dual_row_buffer: bool

    @classmethod
    def load(cls, path: str | Path) -> "SieveCycleV1Config":
        config_path = Path(path)
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"cycle-v1 configuration does not exist: {config_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid cycle-v1 configuration JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError("cycle-v1 configuration root must be an object")
        fields = cls.__dataclass_fields__
        missing = sorted(set(fields) - set(raw))
        extra = sorted(set(raw) - set(fields))
        if missing or extra:
            raise ValueError(
                f"cycle-v1 configuration fields differ; missing={missing}, extra={extra}"
            )
        result = cls(**{name: raw[name] for name in fields})
        result.validate()
        return result

    def validate(self) -> None:
        strings = ("ramulator_commit", "dram_org_preset", "dram_timing_preset")
        for name in strings:
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"cycle-v1 field {name} must be a non-empty string")
        for name in self.__dataclass_fields__:
            if name in strings or name == "dual_row_buffer":
                continue
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"cycle-v1 field {name} must be a positive integer")
        if not isinstance(self.dual_row_buffer, bool):
            raise ValueError("cycle-v1 field dual_row_buffer must be boolean")
        if self.ramulator_commit != PINNED_RAMULATOR_COMMIT:
            raise ValueError("cycle-v1 configuration uses an unpinned Ramulator revision")
        if self.total_pseudo_channels != 256:
            raise ValueError("cycle-v1 currently requires exactly 256 pseudo-channels")
        if self.banks_per_pseudo_channel != 24:
            raise ValueError("cycle-v1 currently requires exactly 24 banks per pseudo-channel")
        if self.columns // self.internal_prefetch_size != self.gpu_cachelines_per_row:
            raise ValueError("gpu_cachelines_per_row must match columns/internal_prefetch_size")

    @property
    def total_channels(self) -> int:
        return self.hbm_pim_stacks * self.channels_per_stack

    @property
    def total_pseudo_channels(self) -> int:
        return self.total_channels * self.pseudo_channels_per_channel

    @property
    def banks_per_pseudo_channel(self) -> int:
        return (
            self.sid_per_pseudo_channel
            * self.bank_groups_per_pseudo_channel
            * self.banks_per_bank_group
        )

    @property
    def organization_count(self) -> list[int]:
        return [
            1,
            self.pseudo_channels_per_channel,
            self.sid_per_pseudo_channel,
            self.bank_groups_per_pseudo_channel,
            self.banks_per_bank_group,
            self.rows,
            self.columns,
        ]


@dataclass(frozen=True)
class MixedWorkloadResult:
    tick_ps: int
    gpu_read_transactions: int
    pim_gwrite_waves: int
    pim_mac_waves: int
    pim_read_waves: int
    gpu_completion_cycles: int
    pim_completion_cycles: int
    pim_gwrite_completion_cycles: int
    pim_mac_completion_cycles: int
    pim_read_completion_cycles: int
    total_completion_cycles: int
    gpu_injected_requests: int
    gpu_completed_requests: int
    pim_injected_requests: int
    pim_completed_requests: int
    gpu_blocked_by_pim_cycles: int
    pim_blocked_by_gpu_cycles: int
    pim_queue_wait_cycles: int
    gpu_column_issues: int
    pim_column_issues: int
    pim_row_activations: int
    pim_row_conflicts: int

    @property
    def gpu_completion_us(self) -> float:
        return self.gpu_completion_cycles * self.tick_ps / 1_000_000.0

    @property
    def pim_completion_us(self) -> float:
        return self.pim_completion_cycles * self.tick_ps / 1_000_000.0

    @property
    def total_completion_us(self) -> float:
        return self.total_completion_cycles * self.tick_ps / 1_000_000.0

    @property
    def pim_gwrite_completion_us(self) -> float:
        return self.pim_gwrite_completion_cycles * self.tick_ps / 1_000_000.0

    @property
    def pim_mac_completion_us(self) -> float:
        return self.pim_mac_completion_cycles * self.tick_ps / 1_000_000.0

    @property
    def pim_read_completion_us(self) -> float:
        return self.pim_read_completion_cycles * self.tick_ps / 1_000_000.0


def _controller_config(ramulator: Any, cycle: SieveCycleV1Config) -> dict[str, Any]:
    dram = ramulator.dram.HBM3(
        org_preset=cycle.dram_org_preset,
        timing_preset=cycle.dram_timing_preset,
    ).to_config()
    dram["org"]["count"] = cycle.organization_count
    return {
        "impl": "SieveHBMPIMV1",
        "read_buffer_size": cycle.read_buffer_size,
        "pim_buffer_size": cycle.pim_buffer_size,
        "pim_mac_interval_ps": cycle.pim_mac_interval_ps,
        "pim_io_interval_ps": cycle.pim_io_interval_ps,
        "dual_row_buffer": cycle.dual_row_buffer,
        "scheduler": {"impl": "FRFCFS"},
        "refresh_manager": {"impl": "NoRefresh"},
        "row_policy": {"impl": "Open"},
        "addr_mapper": {"impl": "PassThroughAddrMapper"},
        "dram": dram,
    }


def run_mixed_workload(
    ramulator_root: str | Path,
    cycle: SieveCycleV1Config,
    *,
    gpu_read_transactions: int,
    pim_gwrite_waves: int,
    pim_mac_waves: int,
    pim_read_waves: int,
) -> MixedWorkloadResult:
    values = {
        "gpu_read_transactions": gpu_read_transactions,
        "pim_gwrite_waves": pim_gwrite_waves,
        "pim_mac_waves": pim_mac_waves,
        "pim_read_waves": pim_read_waves,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if not any(values.values()):
        raise ValueError("mixed workload must contain GPU reads or PIM waves")

    ramulator = _load_ramulator(Path(ramulator_root).resolve())
    controller = _controller_config(ramulator, cycle)
    tick_ps = int(controller["dram"]["timing"][-1])
    if tick_ps != cycle.expected_tick_ps:
        raise ValueError(f"Ramulator tick is {tick_ps} ps, expected {cycle.expected_tick_ps} ps")

    frontend = {
        "impl": "SieveMixed",
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

    expected_pim = (
        pim_gwrite_waves + pim_mac_waves + pim_read_waves
    ) * cycle.total_pseudo_channels
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
        or controller_gpu_completed != gpu_read_transactions
    ):
        raise RuntimeError(
            "incomplete mixed GPU stream: "
            f"injected={gpu_injected}, frontend_completed={gpu_completed}, "
            f"controller_completed={controller_gpu_completed}, expected={gpu_read_transactions}"
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
    if expected_pim > 0 and pim_completion <= 0:
        raise RuntimeError("mixed workload did not record PIM completion")

    def total(key: str) -> int:
        return sum(int(row[key]) for row in controller_stats)

    return MixedWorkloadResult(
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
        total_completion_cycles=max(gpu_completion, pim_completion),
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
    )
