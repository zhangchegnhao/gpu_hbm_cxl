from __future__ import annotations

from collections.abc import Iterable

from ..config import HardwareConfig, ModelConfig
from ..types import ExpertLoad, TimingEstimate
from .analytic import AnalyticTimingModel
from .ramulator_table import RamulatorTimingTable


class RamulatorTableTimingModel(AnalyticTimingModel):
    """Analytic GPU timing combined with strict Ramulator PIM table lookups."""

    def __init__(
        self,
        model: ModelConfig,
        hardware: HardwareConfig,
        table: RamulatorTimingTable,
    ) -> None:
        super().__init__(model, hardware)
        if hardware.timing_backend not in {"ramulator-table-v0", "ramulator-contention-v1"}:
            raise ValueError(f"unsupported Ramulator table backend: {hardware.timing_backend}")
        self.table = table

    @staticmethod
    def _token_count(loads: Iterable[ExpertLoad]) -> int:
        return sum(load.token_count for load in loads)

    def pim_attention(self, context_lengths: tuple[int, ...]) -> TimingEstimate:
        if not context_lengths or len(set(context_lengths)) != 1:
            raise ValueError("Ramulator timing table v1 requires uniform context lengths")
        duration = self.table.attention_us(len(context_lengths), context_lengths[0])
        operations, bytes_accessed = self._attention_shape(context_lengths)
        return TimingEstimate(
            duration,
            operations,
            bytes_accessed,
            "decode attention from Ramulator cycle-v0 table",
        )

    def pim_token_write(self, loads: Iterable[ExpertLoad]) -> TimingEstimate:
        token_count = self._token_count(loads)
        if token_count == 0:
            return TimingEstimate(0.0, description="PIM_GWRITE token operands")
        timing = self.table.expert_us(token_count)
        bytes_accessed = token_count * self.model.hidden_size * self.model.dtype_bytes
        return TimingEstimate(
            timing.gwrite_us,
            0.0,
            bytes_accessed,
            "PIM_GWRITE from Ramulator cycle-v0 table",
        )

    def pim_expert_compute(self, loads: Iterable[ExpertLoad]) -> TimingEstimate:
        token_count = self._token_count(loads)
        if token_count == 0:
            return TimingEstimate(0.0, description="serialized PIM expert GEMV")
        timing = self.table.expert_us(token_count)
        operations = 6 * token_count * self.model.hidden_size * self.model.moe_intermediate_size
        bytes_accessed = token_count * self.model.expert_weight_elements * self.model.dtype_bytes
        return TimingEstimate(
            timing.compute_us,
            operations,
            bytes_accessed,
            "expert GEMV from Ramulator cycle-v0 table",
        )

    def pim_result_read(self, loads: Iterable[ExpertLoad]) -> TimingEstimate:
        token_count = self._token_count(loads)
        if token_count == 0:
            return TimingEstimate(0.0, description="PIM expert result read")
        timing = self.table.expert_us(token_count)
        bytes_accessed = token_count * self.model.hidden_size * self.model.dtype_bytes
        return TimingEstimate(
            timing.read_us,
            0.0,
            bytes_accessed,
            "PIM result read from Ramulator cycle-v0 table",
        )
