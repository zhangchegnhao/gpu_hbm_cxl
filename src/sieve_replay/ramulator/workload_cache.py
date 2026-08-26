from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..config import ModelConfig
from ..report.writer import sha256_file
from ..types import ExpertLoad
from .mixed_workload import MixedWorkloadResult, SieveCycleV1Config


@dataclass(frozen=True)
class WorkloadShape:
    gpu_read_transactions: int
    pim_gwrite_waves: int
    pim_mac_waves: int
    pim_read_waves: int

    def __post_init__(self) -> None:
        if any(value < 0 for value in asdict(self).values()):
            raise ValueError("Ramulator workload shape values must be non-negative")


def cache_context(project_root: str | Path, cycle_config_path: str | Path) -> dict[str, str]:
    root = Path(project_root)
    extension_root = root / "ramulator/extensions/sieve_hbm_pim"
    extension_paths = [
        extension_root / "source/sieve_mixed_frontend.cpp",
        extension_root / "source/sieve_hbm_pim_controller_v1.cpp",
        extension_root / "patches/ramulator2-cycle-v1.patch",
    ]
    digest = hashlib.sha256()
    for path in sorted(extension_paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return {
        "cycle_config_sha256": sha256_file(Path(cycle_config_path)),
        "extension_sha256": digest.hexdigest(),
        "mixed_workload_sha256": sha256_file(
            root / "src/sieve_replay/ramulator/mixed_workload.py"
        ),
    }


class ContentionWorkloadCache:
    def __init__(self, root: str | Path, context: dict[str, str]) -> None:
        self.root = Path(root)
        self.context = dict(context)

    def key(self, shape: WorkloadShape) -> str:
        return hashlib.sha256(
            json.dumps(self._input(shape), sort_keys=True).encode("utf-8")
        ).hexdigest()

    def path(self, shape: WorkloadShape) -> Path:
        return self.root / f"{self.key(shape)}.json"

    def load(self, shape: WorkloadShape) -> MixedWorkloadResult | None:
        exact_path = self.path(shape)
        if exact_path.is_file():
            cached = json.loads(exact_path.read_text(encoding="utf-8"))
            if cached.get("cache_input") == self._input(shape):
                return MixedWorkloadResult(**cached["result"])
        if not self.root.is_dir():
            return None
        shape_dict = asdict(shape)
        for legacy_path in self.root.glob("*.json"):
            cached = json.loads(legacy_path.read_text(encoding="utf-8"))
            cache_input = cached.get("cache_input", {})
            legacy_context = cache_input.get("context", {})
            if cache_input.get("shape") != shape_dict:
                continue
            if legacy_context.get("cycle_config_sha256") != self.context["cycle_config_sha256"]:
                continue
            if legacy_context.get("extension_sha256") != self.context["extension_sha256"]:
                continue
            if legacy_context.get(
                "mixed_workload_sha256", self.context["mixed_workload_sha256"]
            ) != self.context["mixed_workload_sha256"]:
                continue
            result = MixedWorkloadResult(**cached["result"])
            self.store(shape, result)
            return result
        return None

    def store(self, shape: WorkloadShape, result: MixedWorkloadResult) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path(shape)
        path.write_text(
            json.dumps(
                {"cache_input": self._input(shape), "result": asdict(result)},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def _input(self, shape: WorkloadShape) -> dict[str, Any]:
        return {"context": self.context, "shape": asdict(shape)}


def shape_for_loads(
    gpu_loads: tuple[ExpertLoad, ...],
    pim_loads: tuple[ExpertLoad, ...],
    model: ModelConfig,
    cycle: SieveCycleV1Config,
) -> WorkloadShape:
    gpu_bytes = len(gpu_loads) * model.expert_weight_elements * model.dtype_bytes
    if gpu_bytes % cycle.transaction_bytes:
        raise ValueError("GPU expert weight bytes must divide into Ramulator transactions")
    pim_tokens = sum(load.token_count for load in pim_loads)
    operations_per_mac_wave = (
        cycle.total_pseudo_channels
        * cycle.banks_per_pseudo_channel
        * cycle.transaction_bytes
    )
    bytes_per_partitioned_wave = cycle.total_pseudo_channels * cycle.transaction_bytes
    return WorkloadShape(
        gpu_read_transactions=gpu_bytes // cycle.transaction_bytes,
        pim_gwrite_waves=(
            pim_tokens * _ceil_div(model.hidden_size * model.dtype_bytes, cycle.transaction_bytes)
        ),
        pim_mac_waves=(
            _ceil_div(
                6 * pim_tokens * model.hidden_size * model.moe_intermediate_size,
                operations_per_mac_wave,
            )
            if pim_tokens
            else 0
        ),
        pim_read_waves=(
            _ceil_div(
                pim_tokens * model.hidden_size * model.dtype_bytes,
                bytes_per_partitioned_wave,
            )
            if pim_tokens
            else 0
        ),
    )


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator
