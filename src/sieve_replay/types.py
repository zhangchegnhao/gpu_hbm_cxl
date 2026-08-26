from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class AttentionTarget(str, Enum):
    GPU = "gpu"
    PIM = "pim"


class ExpertTarget(str, Enum):
    GPU = "gpu"
    PIM = "pim"


@dataclass(frozen=True)
class ExpertLoad:
    expert_id: int
    token_count: int


@dataclass(frozen=True)
class PlacementDecision:
    policy: str
    attention_target: AttentionTarget
    gpu_experts: tuple[int, ...]
    pim_experts: tuple[int, ...]
    estimated_objective_us: float
    search_report: dict[str, Any] | None = None

    def target_for(self, expert_id: int) -> ExpertTarget:
        if expert_id in self.gpu_experts:
            return ExpertTarget.GPU
        if expert_id in self.pim_experts:
            return ExpertTarget.PIM
        raise KeyError(f"expert {expert_id} is absent from placement")


@dataclass(frozen=True)
class TimingEstimate:
    duration_us: float
    flops: float = 0.0
    bytes_accessed: float = 0.0
    description: str = ""
