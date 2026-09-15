from __future__ import annotations

from abc import ABC, abstractmethod

from ..timing import (
    AnalyticTimingModel,
    ExpertContentionTiming,
    RamulatorContentionTimingModel,
)
from ..trace import TraceBatch
from ..types import AttentionTarget, ExpertLoad, PlacementDecision


def _loads_by_id(loads: tuple[ExpertLoad, ...]) -> dict[int, ExpertLoad]:
    return {load.expert_id: load for load in loads}


class PlacementPolicy(ABC):
    name: str
    attention_target: AttentionTarget

    def __init__(self, timing: AnalyticTimingModel) -> None:
        self.timing = timing

    @abstractmethod
    def place(self, trace: TraceBatch) -> PlacementDecision:
        raise NotImplementedError

    def _decision(
        self,
        loads: tuple[ExpertLoad, ...],
        gpu_experts: tuple[int, ...],
        pim_experts: tuple[int, ...],
        trace: TraceBatch,
    ) -> PlacementDecision:
        objective = self._objective(loads, gpu_experts, pim_experts, trace)
        return PlacementDecision(
            policy=self.name,
            attention_target=self.attention_target,
            gpu_experts=tuple(sorted(gpu_experts)),
            pim_experts=tuple(sorted(pim_experts)),
            estimated_objective_us=objective,
        )

    def _objective(
        self,
        loads: tuple[ExpertLoad, ...],
        gpu_experts: tuple[int, ...],
        pim_experts: tuple[int, ...],
        trace: TraceBatch,
    ) -> float:
        by_id = _loads_by_id(loads)
        gpu_loads = tuple(by_id[expert] for expert in gpu_experts)
        pim_loads = tuple(by_id[expert] for expert in pim_experts)
        gpu_time = (
            self.timing.gpu_expert_weight_load(gpu_loads).duration_us
            + self.timing.gpu_expert_compute(gpu_loads).duration_us
        )
        pim_time = (
            self.timing.pim_expert_compute(pim_loads).duration_us
            + self.timing.pim_token_write(pim_loads).duration_us
            + self.timing.pim_result_read(pim_loads).duration_us
        )
        if self.attention_target is AttentionTarget.PIM:
            pim_time += self.timing.pim_attention(trace.context_lengths).duration_us
        return max(gpu_time, pim_time)


class GpuOnlyPolicy(PlacementPolicy):
    name = "gpu-only"
    attention_target = AttentionTarget.GPU

    def place(self, trace: TraceBatch) -> PlacementDecision:
        active = tuple(load.expert_id for load in trace.expert_loads)
        return self._decision(trace.expert_loads, active, (), trace)


class NoExpPolicy(PlacementPolicy):
    name = "noexp"
    attention_target = AttentionTarget.PIM

    def place(self, trace: TraceBatch) -> PlacementDecision:
        active = tuple(load.expert_id for load in trace.expert_loads)
        return self._decision(trace.expert_loads, active, (), trace)


class AllExpPolicy(PlacementPolicy):
    name = "allexp"
    attention_target = AttentionTarget.PIM

    def place(self, trace: TraceBatch) -> PlacementDecision:
        active = tuple(load.expert_id for load in trace.expert_loads)
        return self._decision(trace.expert_loads, (), active, trace)


class PimOePolicy(PlacementPolicy):
    name = "pimoe"
    attention_target = AttentionTarget.PIM

    def place(self, trace: TraceBatch) -> PlacementDecision:
        threshold = self.timing.hardware.pimoe_token_threshold
        gpu = tuple(load.expert_id for load in trace.expert_loads if load.token_count >= threshold)
        pim = tuple(load.expert_id for load in trace.expert_loads if load.token_count < threshold)
        return self._decision(trace.expert_loads, gpu, pim, trace)


class SievePolicy(PlacementPolicy):
    name = "sieve"
    attention_target = AttentionTarget.PIM

    def place(self, trace: TraceBatch) -> PlacementDecision:
        loads = trace.expert_loads
        ordered = sorted(loads, key=lambda load: (-load.token_count, load.expert_id))
        gpu: list[int] = []
        pim: list[int] = [load.expert_id for load in loads]
        current = self._objective(loads, tuple(gpu), tuple(pim), trace)
        for load in ordered:
            candidate_gpu = (*gpu, load.expert_id)
            candidate_pim = tuple(expert for expert in pim if expert != load.expert_id)
            candidate = self._objective(loads, candidate_gpu, candidate_pim, trace)
            if candidate + 1e-12 >= current:
                break
            gpu.append(load.expert_id)
            pim.remove(load.expert_id)
            current = candidate
        return PlacementDecision(
            policy=self.name,
            attention_target=self.attention_target,
            gpu_experts=tuple(sorted(gpu)),
            pim_experts=tuple(sorted(pim)),
            estimated_objective_us=current,
        )


class SieveCycleV1Policy(PlacementPolicy):
    name = "sieve-cycle-v1"
    attention_target = AttentionTarget.PIM

    def place(self, trace: TraceBatch) -> PlacementDecision:
        if not isinstance(self.timing, RamulatorContentionTimingModel):
            raise ValueError("sieve-cycle-v1 requires the ramulator-contention-v1 backend")
        loads = trace.expert_loads
        by_id = _loads_by_id(loads)
        hot_order = tuple(
            load.expert_id
            for load in sorted(loads, key=lambda load: (-load.token_count, load.expert_id))
        )
        scheduler_us = self.timing.scheduler(self.name).duration_us
        evaluated: list[dict[str, object]] = []
        ranked: list[
            tuple[tuple[float, float, int, tuple[int, ...]], ExpertContentionTiming]
        ] = []
        for row in self.timing.placement_candidates(loads):
            prefix_length = len(row.gpu_experts)
            if set(row.gpu_experts) != set(hot_order[:prefix_length]):
                continue
            gpu_loads = tuple(by_id[expert] for expert in row.gpu_experts)
            gpu_compute_us = self.timing.gpu_expert_compute(gpu_loads).duration_us
            gpu_path_us = row.gpu_contended_us + gpu_compute_us
            pim_path_us = row.pim_contended_us
            objective_us = scheduler_us + max(gpu_path_us, pim_path_us)
            imbalance_us = abs(gpu_path_us - pim_path_us)
            candidate = {
                "gpu_prefix_length": prefix_length,
                "gpu_experts": list(row.gpu_experts),
                "gpu_tokens": row.gpu_token_count,
                "pim_tokens": row.pim_token_count,
                "gpu_memory_us": row.gpu_contended_us,
                "gpu_compute_us": gpu_compute_us,
                "gpu_path_us": gpu_path_us,
                "pim_path_us": pim_path_us,
                "scheduler_us": scheduler_us,
                "objective_us": objective_us,
                "isolation_available": (
                    row.gpu_isolated_us is not None and row.pim_isolated_us is not None
                ),
            }
            evaluated.append(candidate)
            rank = (objective_us, imbalance_us, prefix_length, row.gpu_experts)
            ranked.append((rank, row))
        if not ranked:
            raise ValueError(
                "contention table has no exact hot-prefix candidates for sieve-cycle-v1"
            )
        ranked.sort(key=lambda item: item[0])
        selected = ranked[0][1]
        selected_objective = ranked[0][0][0]
        selected_prefix = len(selected.gpu_experts)
        search_report = {
            "candidate_space": self.timing.contention_table.metadata.get(
                "candidate_space", "exact-table-hot-prefix"
            ),
            "method": self.timing.contention_table.metadata.get(
                "search_method", "enumerate-exact-table-v1"
            ),
            "objective": "scheduler + max(gpu_contended + gpu_compute, pim_contended)",
            "candidate_count": len(evaluated),
            "selected_gpu_prefix_length": selected_prefix,
            "candidates": sorted(
                evaluated, key=lambda candidate: int(candidate["gpu_prefix_length"])
            ),
        }
        return PlacementDecision(
            policy=self.name,
            attention_target=self.attention_target,
            gpu_experts=selected.gpu_experts,
            pim_experts=selected.pim_experts,
            estimated_objective_us=selected_objective,
            search_report=search_report,
        )


class SieveRuntimeV1Policy(PlacementPolicy):
    """Online hot-prefix search using only a small calibrated timing model."""

    name = "sieve-runtime-v1"
    attention_target = AttentionTarget.PIM

    def place(self, trace: TraceBatch) -> PlacementDecision:
        calibration = self.timing.runtime_expert_timing
        if calibration is None:
            raise ValueError("sieve-runtime-v1 requires runtime_calibration")
        loads = trace.expert_loads
        by_id = _loads_by_id(loads)
        hot_order = tuple(
            load.expert_id
            for load in sorted(loads, key=lambda load: (-load.token_count, load.expert_id))
        )
        scheduler_us = self.timing.scheduler(self.name).duration_us
        evaluated: list[dict[str, object]] = []
        ranked: list[tuple[tuple[float, float, int, tuple[int, ...]], dict[str, object]]] = []
        for prefix_length in range(len(hot_order) + 1):
            gpu_experts = tuple(sorted(hot_order[:prefix_length]))
            pim_experts = tuple(sorted(hot_order[prefix_length:]))
            gpu_loads = tuple(by_id[expert] for expert in gpu_experts)
            pim_tokens = sum(by_id[expert].token_count for expert in pim_experts)
            gpu_memory_us = calibration.gpu_expert_read_us(prefix_length)
            gpu_compute_us = self.timing.gpu_expert_compute(gpu_loads).duration_us
            gpu_path_us = gpu_memory_us + gpu_compute_us
            pim_path_us = calibration.pim_expert_pipeline_us(pim_tokens)
            gpu_extrapolated = calibration.gpu_prediction_is_extrapolated(prefix_length)
            pim_extrapolated = calibration.pim_prediction_is_extrapolated(pim_tokens)
            objective_us = scheduler_us + max(gpu_path_us, pim_path_us)
            candidate: dict[str, object] = {
                "gpu_prefix_length": prefix_length,
                "gpu_experts": list(gpu_experts),
                "gpu_tokens": sum(load.token_count for load in gpu_loads),
                "pim_tokens": pim_tokens,
                "gpu_memory_us": gpu_memory_us,
                "gpu_compute_us": gpu_compute_us,
                "gpu_path_us": gpu_path_us,
                "pim_path_us": pim_path_us,
                "scheduler_us": scheduler_us,
                "objective_us": objective_us,
                "isolation_available": False,
                "gpu_prediction_extrapolated": gpu_extrapolated,
                "pim_prediction_extrapolated": pim_extrapolated,
            }
            evaluated.append(candidate)
            rank = (
                objective_us,
                abs(gpu_path_us - pim_path_us),
                prefix_length,
                gpu_experts,
            )
            ranked.append((rank, candidate))
        ranked.sort(key=lambda item: item[0])
        selected = ranked[0][1]
        selected_gpu = tuple(int(value) for value in selected["gpu_experts"])
        selected_pim = tuple(
            expert for expert in sorted(by_id) if expert not in set(selected_gpu)
        )
        search_report = {
            "candidate_space": "runtime-hot-prefix",
            "method": calibration.metadata["method"],
            "objective": (
                "scheduler + max(predicted_gpu_memory + analytic_gpu_compute, "
                "predicted_pim_expert_pipeline)"
            ),
            "oracle_access_during_decision": False,
            "candidate_count": len(evaluated),
            "selected_gpu_prefix_length": len(selected_gpu),
            "selected_prediction_extrapolated": bool(
                selected["gpu_prediction_extrapolated"]
                or selected["pim_prediction_extrapolated"]
            ),
            "calibration": calibration.report(),
            "candidates": evaluated,
        }
        return PlacementDecision(
            policy=self.name,
            attention_target=self.attention_target,
            gpu_experts=selected_gpu,
            pim_experts=selected_pim,
            estimated_objective_us=float(selected["objective_us"]),
            search_report=search_report,
        )


class SieveFixed16CycleV1Policy(PlacementPolicy):
    """Fixed-prefix ablation with the same scheduler overhead as dynamic Sieve."""

    name = "sieve-fixed-16-cycle-v1"
    attention_target = AttentionTarget.PIM
    gpu_prefix_length = 16

    def place(self, trace: TraceBatch) -> PlacementDecision:
        if not isinstance(self.timing, RamulatorContentionTimingModel):
            raise ValueError(
                "sieve-fixed-16-cycle-v1 requires the ramulator-contention-v1 backend"
            )
        loads = trace.expert_loads
        by_id = _loads_by_id(loads)
        hot_order = tuple(
            load.expert_id
            for load in sorted(loads, key=lambda load: (-load.token_count, load.expert_id))
        )
        prefix_length = min(self.gpu_prefix_length, len(hot_order))
        expected_gpu = set(hot_order[:prefix_length])
        selected = next(
            (
                row
                for row in self.timing.placement_candidates(loads)
                if len(row.gpu_experts) == prefix_length
                and set(row.gpu_experts) == expected_gpu
            ),
            None,
        )
        if selected is None:
            raise ValueError(
                "contention table has no exact fixed-16 hot-prefix candidate"
            )
        gpu_loads = tuple(by_id[expert] for expert in selected.gpu_experts)
        gpu_compute_us = self.timing.gpu_expert_compute(gpu_loads).duration_us
        gpu_path_us = selected.gpu_contended_us + gpu_compute_us
        pim_path_us = selected.pim_contended_us
        scheduler_us = self.timing.scheduler(self.name).duration_us
        objective_us = scheduler_us + max(gpu_path_us, pim_path_us)
        return PlacementDecision(
            policy=self.name,
            attention_target=self.attention_target,
            gpu_experts=selected.gpu_experts,
            pim_experts=selected.pim_experts,
            estimated_objective_us=objective_us,
            search_report={
                "candidate_space": "fixed-hot-prefix-ablation",
                "method": "fixed-gpu-prefix-length",
                "objective": (
                    "scheduler + max(gpu_contended + gpu_compute, pim_contended)"
                ),
                "candidate_count": 1,
                "selected_gpu_prefix_length": prefix_length,
                "configured_gpu_prefix_length": self.gpu_prefix_length,
                "candidates": [
                    {
                        "gpu_prefix_length": prefix_length,
                        "gpu_experts": list(selected.gpu_experts),
                        "gpu_tokens": selected.gpu_token_count,
                        "pim_tokens": selected.pim_token_count,
                        "gpu_memory_us": selected.gpu_contended_us,
                        "gpu_compute_us": gpu_compute_us,
                        "gpu_path_us": gpu_path_us,
                        "pim_path_us": pim_path_us,
                        "scheduler_us": scheduler_us,
                        "objective_us": objective_us,
                    }
                ],
            },
        )


_POLICIES = {
    "gpu-only": GpuOnlyPolicy,
    "noexp": NoExpPolicy,
    "allexp": AllExpPolicy,
    "pimoe": PimOePolicy,
    "sieve": SievePolicy,
    "sieve-runtime-v1": SieveRuntimeV1Policy,
    "sieve-cycle-v1": SieveCycleV1Policy,
    "sieve-fixed-16-cycle-v1": SieveFixed16CycleV1Policy,
}


def available_policies() -> tuple[str, ...]:
    return tuple(_POLICIES)


def create_policy(name: str, timing: AnalyticTimingModel) -> PlacementPolicy:
    try:
        policy_type = _POLICIES[name]
    except KeyError as exc:
        raise ValueError(f"unknown policy {name!r}; choose from {', '.join(_POLICIES)}") from exc
    return policy_type(timing)
