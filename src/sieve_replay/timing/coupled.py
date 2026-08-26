from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..types import TimingEstimate


@dataclass(frozen=True)
class CoupledExpertTiming:
    gpu_weight_load: TimingEstimate
    pim_token_write: TimingEstimate
    pim_expert_compute: TimingEstimate
    pim_result_read: TimingEstimate
    report: dict[str, Any]
