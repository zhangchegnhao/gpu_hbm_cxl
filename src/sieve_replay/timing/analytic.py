from __future__ import annotations

from collections.abc import Iterable

from ..config import HardwareConfig, ModelConfig
from ..types import ExpertLoad, TimingEstimate
from .coupled import CoupledExpertTiming


class AnalyticTimingModel:
    """Ideal roofline model used only for functional trace-replay validation."""

    def __init__(self, model: ModelConfig, hardware: HardwareConfig) -> None:
        if hardware.timing_backend not in {
            "analytic-v0",
            "ramulator-table-v0",
            "ramulator-contention-v1",
        }:
            raise ValueError(f"unsupported timing backend: {hardware.timing_backend}")
        self.model = model
        self.hardware = hardware

    def coupled_expert_timing(
        self,
        gpu_loads: tuple[ExpertLoad, ...],
        pim_loads: tuple[ExpertLoad, ...],
    ) -> CoupledExpertTiming | None:
        return None

    @property
    def gpu_flops_per_second(self) -> float:
        return self.hardware.gpu_peak_tflops * 1e12 * self.hardware.gpu_compute_efficiency

    @property
    def hbm_bytes_per_second(self) -> float:
        return self.hardware.hbm_bandwidth_tb_s * 1e12 * self.hardware.hbm_bandwidth_efficiency

    @property
    def pim_ops_per_second(self) -> float:
        return (
            self.hardware.hbm_bandwidth_tb_s
            * 1e12
            * self.hardware.pim_compute_density_ops_per_byte
            * self.hardware.pim_compute_efficiency
        )

    @staticmethod
    def _seconds_to_us(seconds: float) -> float:
        return seconds * 1e6

    def _gpu_roofline(
        self, flops: float, bytes_accessed: float, kernels: int, description: str
    ) -> TimingEstimate:
        compute = flops / self.gpu_flops_per_second
        memory = bytes_accessed / self.hbm_bytes_per_second
        duration = self._seconds_to_us(max(compute, memory)) + kernels * self.hardware.gpu_kernel_overhead_us
        return TimingEstimate(duration, flops, bytes_accessed, description)

    def _pim_roofline(
        self, operations: float, bytes_accessed: float, command_count: int, description: str
    ) -> TimingEstimate:
        compute = operations / self.pim_ops_per_second
        memory = bytes_accessed / self.hbm_bytes_per_second
        duration = self._seconds_to_us(max(compute, memory))
        duration += command_count * self.hardware.pim_command_overhead_us
        return TimingEstimate(duration, operations, bytes_accessed, description)

    def norm(self, batch_size: int) -> TimingEstimate:
        h = self.model.hidden_size
        return self._gpu_roofline(5 * batch_size * h, 2 * batch_size * h * self.model.dtype_bytes, 1, "RMSNorm")

    def qkv_projection(self, batch_size: int) -> TimingEstimate:
        h = self.model.hidden_size
        out = self.model.q_projection_size + 2 * self.model.kv_projection_size
        flops = 2 * batch_size * h * out
        weight_bytes = h * out * self.model.dtype_bytes
        activation_bytes = batch_size * (h + out) * self.model.dtype_bytes
        return self._gpu_roofline(flops, weight_bytes + activation_bytes, 1, "QKV projection")

    def rope(self, batch_size: int) -> TimingEstimate:
        width = self.model.q_projection_size + self.model.kv_projection_size
        return self._gpu_roofline(6 * batch_size * width, 2 * batch_size * width * self.model.dtype_bytes, 1, "QK norm and RoPE")

    def _attention_shape(self, context_lengths: tuple[int, ...]) -> tuple[float, float]:
        tokens = sum(context_lengths)
        flops = 4 * tokens * self.model.num_attention_heads * self.model.head_dim
        kv_bytes = 2 * tokens * self.model.num_key_value_heads * self.model.head_dim * self.model.dtype_bytes
        query_output_bytes = 2 * len(context_lengths) * self.model.q_projection_size * self.model.dtype_bytes
        return float(flops), float(kv_bytes + query_output_bytes)

    def gpu_attention(self, context_lengths: tuple[int, ...]) -> TimingEstimate:
        flops, bytes_accessed = self._attention_shape(context_lengths)
        return self._gpu_roofline(flops, bytes_accessed, 1, "decode attention on GPU")

    def pim_attention(self, context_lengths: tuple[int, ...]) -> TimingEstimate:
        operations, bytes_accessed = self._attention_shape(context_lengths)
        command_count = 2 * len(context_lengths)
        return self._pim_roofline(operations, bytes_accessed, command_count, "decode attention on PIM")

    def output_projection(self, batch_size: int) -> TimingEstimate:
        in_features = self.model.q_projection_size
        out_features = self.model.hidden_size
        flops = 2 * batch_size * in_features * out_features
        weight_bytes = in_features * out_features * self.model.dtype_bytes
        activations = batch_size * (in_features + out_features) * self.model.dtype_bytes
        return self._gpu_roofline(flops, weight_bytes + activations, 1, "attention output projection")

    def residual(self, batch_size: int) -> TimingEstimate:
        elements = batch_size * self.model.hidden_size
        return self._gpu_roofline(elements, 3 * elements * self.model.dtype_bytes, 1, "residual add")

    def router(self, batch_size: int) -> TimingEstimate:
        h = self.model.hidden_size
        experts = self.model.num_experts
        flops = 2 * batch_size * h * experts
        bytes_accessed = (h * experts + batch_size * (h + experts)) * self.model.dtype_bytes
        estimate = self._gpu_roofline(flops, bytes_accessed, 1, "MoE router")
        return TimingEstimate(
            estimate.duration_us + self.hardware.router_overhead_us,
            estimate.flops,
            estimate.bytes_accessed,
            estimate.description,
        )

    def metadata(self, total_assignments: int) -> TimingEstimate:
        bytes_accessed = total_assignments * 16
        duration = self._seconds_to_us(bytes_accessed / self.hbm_bytes_per_second)
        return TimingEstimate(duration + self.hardware.metadata_overhead_us, 0.0, bytes_accessed, "routing metadata")

    def scheduler(self, policy: str) -> TimingEstimate:
        overhead = (
            self.hardware.sieve_scheduler_overhead_us
            if policy in {
                "sieve",
                "sieve-cycle-v1",
                "sieve-fixed-16-cycle-v1",
            }
            else self.hardware.static_policy_overhead_us
        )
        return TimingEstimate(overhead, description=f"{policy} placement policy")

    def dispatch(self) -> TimingEstimate:
        return TimingEstimate(0.0, description="single-GPU dispatch placeholder")

    def gpu_expert_weight_load(self, loads: Iterable[ExpertLoad]) -> TimingEstimate:
        active = sum(1 for load in loads if load.token_count > 0)
        bytes_accessed = active * self.model.expert_weight_elements * self.model.dtype_bytes
        duration = self._seconds_to_us(bytes_accessed / self.hbm_bytes_per_second)
        return TimingEstimate(duration, 0.0, bytes_accessed, "expert weights HBM to GPU")

    def gpu_expert_compute(self, loads: Iterable[ExpertLoad]) -> TimingEstimate:
        token_count = sum(load.token_count for load in loads)
        flops = 6 * token_count * self.model.hidden_size * self.model.moe_intermediate_size
        duration = self._seconds_to_us(flops / self.gpu_flops_per_second)
        kernels = 3 if token_count else 0
        duration += kernels * self.hardware.gpu_kernel_overhead_us
        activations = token_count * (2 * self.model.hidden_size + 2 * self.model.moe_intermediate_size)
        return TimingEstimate(duration, flops, activations * self.model.dtype_bytes, "grouped expert GEMM")

    def pim_token_write(self, loads: Iterable[ExpertLoad]) -> TimingEstimate:
        token_count = sum(load.token_count for load in loads)
        bytes_accessed = token_count * self.model.hidden_size * self.model.dtype_bytes
        duration = self._seconds_to_us(bytes_accessed / self.hbm_bytes_per_second)
        if token_count:
            duration += self.hardware.pim_gwrite_overhead_us
        return TimingEstimate(duration, 0.0, bytes_accessed, "PIM_GWRITE token operands")

    def pim_expert_compute(self, loads: Iterable[ExpertLoad]) -> TimingEstimate:
        token_count = sum(load.token_count for load in loads)
        operations = 6 * token_count * self.model.hidden_size * self.model.moe_intermediate_size
        bytes_accessed = token_count * self.model.expert_weight_elements * self.model.dtype_bytes
        command_count = 3 * token_count
        return self._pim_roofline(operations, bytes_accessed, command_count, "serialized PIM expert GEMV")

    def pim_result_read(self, loads: Iterable[ExpertLoad]) -> TimingEstimate:
        token_count = sum(load.token_count for load in loads)
        bytes_accessed = token_count * self.model.hidden_size * self.model.dtype_bytes
        duration = self._seconds_to_us(bytes_accessed / self.hbm_bytes_per_second)
        if token_count:
            duration += self.hardware.pim_read_overhead_us
        return TimingEstimate(duration, 0.0, bytes_accessed, "PIM expert result read")

    def combine(self, batch_size: int) -> TimingEstimate:
        assignments = batch_size * self.model.num_experts_per_tok
        flops = 2 * assignments * self.model.hidden_size
        bytes_accessed = (assignments + batch_size) * self.model.hidden_size * self.model.dtype_bytes
        return self._gpu_roofline(flops, bytes_accessed, 1, "weighted expert combine")
