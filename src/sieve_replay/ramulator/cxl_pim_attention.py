from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import ModelConfig
from ..report.writer import sha256_file
from .microbenchmark import MicrobenchmarkResult, run_milestones
from .mixed_workload import SieveCycleV1Config


@dataclass(frozen=True)
class CXLPIMTopology:
    """Explicit assumptions for the first isolated CXL-PIM Attention model."""

    channels: int = 4
    pseudo_channels_per_channel: int = 2
    banks_per_pseudo_channel: int = 24
    partial_scalar_bytes: int = 4

    def __post_init__(self) -> None:
        for value in asdict(self).values():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("CXL-PIM topology fields must be positive integers")

    @property
    def total_pseudo_channels(self) -> int:
        return self.channels * self.pseudo_channels_per_channel


@dataclass(frozen=True)
class CXLPIMAttentionShape:
    spilled_kv_bytes_per_layer: int
    kv_bytes_per_context_position: int
    spilled_context_positions: int
    query_bytes: int
    partial_result_bytes: int
    query_link_transactions: int
    result_link_transactions: int
    pim_gwrite_waves: int
    pim_mac_waves: int
    pim_read_waves: int
    mac_operations: int
    mac_operations_per_wave: int


@dataclass(frozen=True)
class _CXLPIMCycle:
    dram_org_preset: str
    dram_timing_preset: str
    channels: int
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

    @property
    def total_channels(self) -> int:
        return self.channels

    @property
    def total_pseudo_channels(self) -> int:
        return self.channels * self.pseudo_channels_per_channel

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


def build_cxl_pim_attention_shape(
    model: ModelConfig,
    cycle: SieveCycleV1Config,
    *,
    batch_size: int,
    spilled_kv_bytes_per_layer: int,
    topology: CXLPIMTopology,
) -> CXLPIMAttentionShape:
    if batch_size <= 0 or spilled_kv_bytes_per_layer <= 0:
        raise ValueError("batch size and spilled KV bytes must be positive")
    if topology.banks_per_pseudo_channel != cycle.banks_per_pseudo_channel:
        raise ValueError("CXL-PIM and cycle configurations disagree on banks per PCH")
    kv_bytes_per_position = (
        2 * model.num_key_value_heads * model.head_dim * model.dtype_bytes
    )
    spilled_positions = _ceil_div(
        spilled_kv_bytes_per_layer, kv_bytes_per_position
    )
    query_bytes = (
        batch_size
        * model.num_attention_heads
        * model.head_dim
        * model.dtype_bytes
    )
    partial_bytes = (
        batch_size
        * model.num_attention_heads
        * (model.head_dim + 2)
        * topology.partial_scalar_bytes
        * topology.total_pseudo_channels
    )
    mac_operations = (
        2
        * spilled_positions
        * model.num_attention_heads
        * model.head_dim
    )
    operations_per_wave = (
        topology.total_pseudo_channels
        * topology.banks_per_pseudo_channel
        * cycle.transaction_bytes
    )
    return CXLPIMAttentionShape(
        spilled_kv_bytes_per_layer=spilled_kv_bytes_per_layer,
        kv_bytes_per_context_position=kv_bytes_per_position,
        spilled_context_positions=spilled_positions,
        query_bytes=query_bytes,
        partial_result_bytes=partial_bytes,
        query_link_transactions=_ceil_div(query_bytes, cycle.transaction_bytes),
        result_link_transactions=_ceil_div(partial_bytes, cycle.transaction_bytes),
        # One 32-byte segment is broadcast to every CXL-PIM PCH per GWRITE wave.
        pim_gwrite_waves=_ceil_div(query_bytes, cycle.transaction_bytes),
        pim_mac_waves=_ceil_div(mac_operations, operations_per_wave),
        # Each PIM_READ wave returns one transaction from every PCH.
        pim_read_waves=_ceil_div(
            partial_bytes,
            topology.total_pseudo_channels * cycle.transaction_bytes,
        ),
        mac_operations=mac_operations,
        mac_operations_per_wave=operations_per_wave,
    )


def run_cxl_pim_operation(
    ramulator_root: str | Path,
    cycle: SieveCycleV1Config,
    topology: CXLPIMTopology,
    operation: str,
    waves: int,
) -> MicrobenchmarkResult:
    if waves <= 0:
        raise ValueError("CXL-PIM operation requires positive waves")
    cxl_cycle = _CXLPIMCycle(
        dram_org_preset=cycle.dram_org_preset,
        dram_timing_preset=cycle.dram_timing_preset,
        channels=topology.channels,
        pseudo_channels_per_channel=topology.pseudo_channels_per_channel,
        sid_per_pseudo_channel=cycle.sid_per_pseudo_channel,
        bank_groups_per_pseudo_channel=cycle.bank_groups_per_pseudo_channel,
        banks_per_bank_group=cycle.banks_per_bank_group,
        rows=cycle.rows,
        columns=cycle.columns,
        transaction_bytes=cycle.transaction_bytes,
        pim_mac_interval_ps=cycle.pim_mac_interval_ps,
        pim_io_interval_ps=cycle.pim_io_interval_ps,
        pim_buffer_size=cycle.pim_buffer_size,
        row_span_waves=cycle.row_span_waves,
        expected_tick_ps=cycle.expected_tick_ps,
    )
    return run_milestones(ramulator_root, cxl_cycle, operation, (waves,))


def cxl_pim_attention_context(
    project_root: str | Path, cycle_config_path: str | Path
) -> dict[str, str]:
    root = Path(project_root)
    sources = [
        root / "ramulator/extensions/sieve_hbm_pim/source/sieve_pim_frontend.cpp",
        root / "ramulator/extensions/sieve_hbm_pim/source/sieve_hbm_pim_controller.cpp",
        root / "ramulator/extensions/sieve_hbm_pim/patches/ramulator2-cycle-v0.patch",
    ]
    digest = hashlib.sha256()
    for path in sorted(sources):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return {
        "model": "isolated-cxl-pim-attention-v1",
        "cycle_config_sha256": sha256_file(Path(cycle_config_path)),
        "pim_extension_sha256": digest.hexdigest(),
        "runner_sha256": sha256_file(Path(__file__)),
        "microbenchmark_sha256": sha256_file(
            root / "src/sieve_replay/ramulator/microbenchmark.py"
        ),
    }


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator
