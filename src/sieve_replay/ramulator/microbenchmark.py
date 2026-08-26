from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SieveCycleConfig:
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
    pim_mac_interval_ps: int
    pim_io_interval_ps: int
    pim_buffer_size: int
    row_span_waves: int
    expected_tick_ps: int

    @classmethod
    def load(cls, path: str | Path) -> "SieveCycleConfig":
        config_path = Path(path)
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"cycle configuration does not exist: {config_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid cycle configuration JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError("cycle configuration root must be an object")
        fields = cls.__dataclass_fields__
        missing = sorted(set(fields) - set(raw))
        if missing:
            raise ValueError(f"cycle configuration missing fields: {', '.join(missing)}")
        result = cls(**{name: raw[name] for name in fields})
        result.validate()
        return result

    def validate(self) -> None:
        for name in self.__dataclass_fields__:
            if name in {"ramulator_commit", "dram_org_preset", "dram_timing_preset"}:
                if not isinstance(getattr(self, name), str) or not getattr(self, name):
                    raise ValueError(f"cycle configuration field {name} must be a non-empty string")
            elif not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"cycle configuration field {name} must be a positive integer")
        if self.total_pseudo_channels != 256:
            raise ValueError("cycle-v0 currently requires exactly 256 pseudo-channels")
        if self.banks_per_pseudo_channel != 24:
            raise ValueError("cycle-v0 currently requires exactly 24 banks per pseudo-channel")

    @property
    def total_channels(self) -> int:
        return self.hbm_pim_stacks * self.channels_per_stack

    @property
    def total_pseudo_channels(self) -> int:
        return self.total_channels * self.pseudo_channels_per_channel

    @property
    def banks_per_pseudo_channel(self) -> int:
        return self.sid_per_pseudo_channel * self.bank_groups_per_pseudo_channel * self.banks_per_bank_group

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
class MicrobenchmarkResult:
    operation: str
    max_waves: int
    tick_ps: int
    duration_us_by_waves: dict[int, float]
    cycles_by_waves: dict[int, int]
    injected_requests: int
    completed_requests: int


def _load_ramulator(ramulator_root: Path) -> Any:
    python_root = ramulator_root / "python"
    binding = next((python_root / "ramulator").glob("_ramulator*.so"), None)
    if binding is None:
        raise ValueError(
            f"Ramulator Python binding is not built under {python_root}; "
            "run scripts/build_sieve_ramulator.sh first"
        )
    python_root_str = str(python_root.resolve())
    if python_root_str not in sys.path:
        sys.path.insert(0, python_root_str)
    return importlib.import_module("ramulator")


def _controller_config(ramulator: Any, cycle: SieveCycleConfig, milestones: tuple[int, ...]) -> dict[str, Any]:
    dram = ramulator.dram.HBM3(
        org_preset=cycle.dram_org_preset,
        timing_preset=cycle.dram_timing_preset,
    ).to_config()
    dram["org"]["count"] = cycle.organization_count
    return {
        "impl": "SieveHBMPIM",
        "pim_buffer_size": cycle.pim_buffer_size,
        "pim_mac_interval_ps": cycle.pim_mac_interval_ps,
        "pim_io_interval_ps": cycle.pim_io_interval_ps,
        "pim_milestone_waves": ",".join(str(wave) for wave in milestones),
        "scheduler": {"impl": "FRFCFS"},
        "refresh_manager": {"impl": "NoRefresh"},
        "row_policy": {"impl": "Open"},
        "addr_mapper": {"impl": "PassThroughAddrMapper"},
        "dram": dram,
    }


def run_milestones(
    ramulator_root: str | Path,
    cycle: SieveCycleConfig,
    operation: str,
    waves: list[int] | tuple[int, ...] | set[int],
) -> MicrobenchmarkResult:
    milestones = tuple(sorted(set(waves)))
    if not milestones or milestones[0] <= 0:
        raise ValueError("microbenchmark milestones must contain positive wave counts")
    if operation not in {"PIM_GWRITE", "PIM_MAC", "PIM_READ"}:
        raise ValueError(f"unsupported microbenchmark operation: {operation}")

    root = Path(ramulator_root).resolve()
    ramulator = _load_ramulator(root)
    controller = _controller_config(ramulator, cycle, milestones)
    tick_ps = int(controller["dram"]["timing"][-1])
    if tick_ps != cycle.expected_tick_ps:
        raise ValueError(f"Ramulator tick is {tick_ps} ps, expected {cycle.expected_tick_ps} ps")

    frontend = {
        "impl": "SievePIM",
        "clock_ratio": 1,
        "operation": operation,
        "waves": milestones[-1],
        "channels": cycle.total_channels,
        "pseudo_channels": cycle.pseudo_channels_per_channel,
        "row_span_waves": cycle.row_span_waves,
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

    controller_stats = stats["memory_system"]["controller"]
    if not isinstance(controller_stats, list):
        controller_stats = [controller_stats]
    counter_key = {
        "PIM_GWRITE": "num_pim_gwrite_reqs",
        "PIM_MAC": "num_pim_mac_reqs",
        "PIM_READ": "num_pim_read_reqs",
    }[operation]
    completed = sum(int(row[counter_key]) for row in controller_stats)
    injected = int(stats["frontend"]["injected_requests"])
    expected = milestones[-1] * cycle.total_pseudo_channels
    if injected != expected or completed != expected:
        raise RuntimeError(
            f"incomplete {operation} microbenchmark: injected={injected}, "
            f"completed={completed}, expected={expected}"
        )

    cycles_by_waves: dict[int, int] = {}
    duration_us_by_waves: dict[int, float] = {}
    for wave in milestones:
        key = f"pim_milestone_{wave}_cycles"
        cycles = max(int(row[key]) for row in controller_stats)
        if cycles <= 0:
            raise RuntimeError(f"Ramulator did not record milestone {wave} for {operation}")
        cycles_by_waves[wave] = cycles
        duration_us_by_waves[wave] = cycles * tick_ps / 1_000_000.0

    return MicrobenchmarkResult(
        operation=operation,
        max_waves=milestones[-1],
        tick_ps=tick_ps,
        duration_us_by_waves=duration_us_by_waves,
        cycles_by_waves=cycles_by_waves,
        injected_requests=injected,
        completed_requests=completed,
    )
