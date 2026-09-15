from __future__ import annotations

from ..config import HardwareConfig, ModelConfig
from ..types import ExpertLoad, TimingEstimate
from .contention_table import ExpertContentionTiming, RamulatorContentionTable
from .coupled import CoupledExpertTiming
from .ramulator_model import RamulatorTableTimingModel
from .ramulator_table import RamulatorTimingTable
from .runtime_model import RuntimeExpertTimingModel


class RamulatorContentionTimingModel(RamulatorTableTimingModel):
    """Cycle-v1 mixed HBM/PIM expert timing with isolated timing for policy search."""

    def __init__(
        self,
        model: ModelConfig,
        hardware: HardwareConfig,
        isolated_table: RamulatorTimingTable,
        contention_table: RamulatorContentionTable,
        runtime_expert_timing: RuntimeExpertTimingModel | None = None,
    ) -> None:
        super().__init__(model, hardware, isolated_table, runtime_expert_timing)
        if hardware.timing_backend != "ramulator-contention-v1":
            raise ValueError(f"unsupported contention backend: {hardware.timing_backend}")
        self.contention_table = contention_table
        self.last_contention_report: dict[str, object] | None = None

    def placement_candidates(
        self,
        loads: tuple[ExpertLoad, ...],
    ) -> tuple[ExpertContentionTiming, ...]:
        return self.contention_table.candidate_timings(loads)

    def coupled_expert_timing(
        self,
        gpu_loads: tuple[ExpertLoad, ...],
        pim_loads: tuple[ExpertLoad, ...],
    ) -> CoupledExpertTiming:
        row = self.contention_table.expert_timing(gpu_loads, pim_loads)
        self.last_contention_report = row.as_report()
        gpu_bytes = len(gpu_loads) * self.model.expert_weight_elements * self.model.dtype_bytes
        pim_tokens = sum(load.token_count for load in pim_loads)
        pim_input_bytes = pim_tokens * self.model.hidden_size * self.model.dtype_bytes
        pim_operations = 6 * pim_tokens * self.model.hidden_size * self.model.moe_intermediate_size
        pim_output_bytes = pim_input_bytes
        return CoupledExpertTiming(
            gpu_weight_load=TimingEstimate(
                row.gpu_contended_us,
                0.0,
                gpu_bytes,
                "ordinary GPU HBM reads from mixed Ramulator cycle-v1 run",
            ),
            pim_token_write=TimingEstimate(
                row.pim_gwrite_contended_us,
                0.0,
                pim_input_bytes,
                "PIM_GWRITE milestone from mixed Ramulator cycle-v1 run",
            ),
            pim_expert_compute=TimingEstimate(
                row.pim_compute_contended_us,
                pim_operations,
                pim_tokens * self.model.expert_weight_elements * self.model.dtype_bytes,
                "PIM_MAC milestone delta from mixed Ramulator cycle-v1 run",
            ),
            pim_result_read=TimingEstimate(
                row.pim_read_contended_us,
                0.0,
                pim_output_bytes,
                "PIM_READ milestone delta from mixed Ramulator cycle-v1 run",
            ),
            report=row.as_report(),
        )
