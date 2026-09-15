from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"configuration file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"configuration root must be an object: {path}")
    return data


def find_project_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ValueError(f"cannot locate project root from {start}")


@dataclass(frozen=True)
class ModelConfig:
    name: str
    architecture: str
    dtype: str
    dtype_bytes: int
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_experts: int
    num_experts_per_tok: int
    vocab_size: int
    max_position_embeddings: int
    has_shared_expert: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        fields = cls.__dataclass_fields__
        missing = sorted(name for name in fields if name not in data)
        if missing:
            raise ValueError(f"model configuration missing fields: {', '.join(missing)}")
        result = cls(**{name: data[name] for name in fields})
        result.validate()
        return result

    def validate(self) -> None:
        positive = (
            "dtype_bytes",
            "num_hidden_layers",
            "hidden_size",
            "intermediate_size",
            "moe_intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "num_experts",
            "num_experts_per_tok",
            "vocab_size",
            "max_position_embeddings",
        )
        for field in positive:
            if getattr(self, field) <= 0:
                raise ValueError(f"model field {field} must be positive")
        if self.num_experts_per_tok > self.num_experts:
            raise ValueError("num_experts_per_tok cannot exceed num_experts")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

    @property
    def q_projection_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_projection_size(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def expert_weight_elements(self) -> int:
        return 3 * self.hidden_size * self.moe_intermediate_size


@dataclass(frozen=True)
class HardwareConfig:
    name: str
    timing_backend: str
    gpu_count: int
    gpu_peak_tflops: float
    gpu_compute_efficiency: float
    hbm_bandwidth_tb_s: float
    hbm_bandwidth_efficiency: float
    hbm_pim_stacks: int
    pseudo_channels_per_stack: int
    banks_per_pseudo_channel: int
    hbm_pim_capacity_gb: float
    pim_compute_density_ops_per_byte: float
    pim_compute_efficiency: float
    gpu_kernel_overhead_us: float
    pim_command_overhead_us: float
    pim_gwrite_overhead_us: float
    pim_read_overhead_us: float
    router_overhead_us: float
    metadata_overhead_us: float
    sieve_scheduler_overhead_us: float
    static_policy_overhead_us: float
    pimoe_token_threshold: int
    nvlink_bandwidth_gb_s_per_direction: float
    nvlink_latency_us: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HardwareConfig":
        fields = cls.__dataclass_fields__
        missing = sorted(name for name in fields if name not in data)
        if missing:
            raise ValueError(f"hardware configuration missing fields: {', '.join(missing)}")
        result = cls(**{name: data[name] for name in fields})
        result.validate()
        return result

    def validate(self) -> None:
        if self.gpu_count != 1:
            raise ValueError("the current milestone supports exactly one GPU")
        positive = (
            "gpu_peak_tflops",
            "gpu_compute_efficiency",
            "hbm_bandwidth_tb_s",
            "hbm_bandwidth_efficiency",
            "hbm_pim_stacks",
            "pseudo_channels_per_stack",
            "banks_per_pseudo_channel",
            "hbm_pim_capacity_gb",
            "pim_compute_density_ops_per_byte",
            "pim_compute_efficiency",
            "pimoe_token_threshold",
        )
        for field in positive:
            if getattr(self, field) <= 0:
                raise ValueError(f"hardware field {field} must be positive")
        for field in ("gpu_compute_efficiency", "hbm_bandwidth_efficiency", "pim_compute_efficiency"):
            if getattr(self, field) > 1.0:
                raise ValueError(f"hardware efficiency {field} cannot exceed 1.0")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model_path: Path
    hardware_path: Path
    trace_path: Path
    pim_timing_table_path: Path | None
    contention_timing_table_path: Path | None
    ramulator_cycle_config_path: Path | None
    trace_manifest_path: Path | None
    runtime_calibration_path: Path | None
    layers: tuple[int, ...]
    steps: tuple[int, ...]
    policies: tuple[str, ...]
    raw: dict[str, Any]

    @property
    def layer(self) -> int:
        if len(self.layers) != 1:
            raise ValueError("this operation requires exactly one selected layer")
        return self.layers[0]

    @property
    def step(self) -> int:
        if len(self.steps) != 1:
            raise ValueError("this operation requires exactly one selected decode step")
        return self.steps[0]

    @property
    def is_single_layer_step(self) -> bool:
        return len(self.layers) == 1 and len(self.steps) == 1


@dataclass(frozen=True)
class LoadedConfiguration:
    experiment: ExperimentConfig
    model: ModelConfig
    hardware: HardwareConfig
    model_raw: dict[str, Any]
    hardware_raw: dict[str, Any]


def load_configuration(experiment_path: str | Path) -> LoadedConfiguration:
    experiment_file = Path(experiment_path).resolve()
    raw = _load_json(experiment_file)
    required = {"name", "model", "hardware", "trace", "policies"}
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"experiment configuration missing fields: {', '.join(missing)}")

    project_root = find_project_root(experiment_file)

    def resolve(value: str) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (project_root / path).resolve()

    model_path = resolve(raw["model"])
    hardware_path = resolve(raw["hardware"])
    trace_path = resolve(raw["trace"])
    pim_timing_table_path = resolve(raw["pim_timing_table"]) if "pim_timing_table" in raw else None
    contention_timing_table_path = (
        resolve(raw["contention_timing_table"])
        if "contention_timing_table" in raw
        else None
    )
    ramulator_cycle_config_path = (
        resolve(raw["ramulator_cycle_config"])
        if "ramulator_cycle_config" in raw
        else None
    )
    trace_manifest_path = (
        resolve(raw["trace_manifest"]) if "trace_manifest" in raw else None
    )
    runtime_calibration_path = (
        resolve(raw["runtime_calibration"])
        if "runtime_calibration" in raw
        else None
    )
    model_raw = _load_json(model_path)
    hardware_raw = _load_json(hardware_path)
    model = ModelConfig.from_dict(model_raw)
    hardware = HardwareConfig.from_dict(hardware_raw)
    layers = _parse_selection(raw, "layer", "layers", upper_bound=model.num_hidden_layers)
    steps = _parse_selection(raw, "step", "steps")
    experiment = ExperimentConfig(
        name=str(raw["name"]),
        model_path=model_path,
        hardware_path=hardware_path,
        trace_path=trace_path,
        pim_timing_table_path=pim_timing_table_path,
        contention_timing_table_path=contention_timing_table_path,
        ramulator_cycle_config_path=ramulator_cycle_config_path,
        trace_manifest_path=trace_manifest_path,
        runtime_calibration_path=runtime_calibration_path,
        layers=layers,
        steps=steps,
        policies=tuple(str(policy) for policy in raw["policies"]),
        raw=raw,
    )
    return LoadedConfiguration(experiment, model, hardware, model_raw, hardware_raw)


def _parse_selection(
    raw: dict[str, Any],
    singular: str,
    plural: str,
    upper_bound: int | None = None,
) -> tuple[int, ...]:
    has_singular = singular in raw
    has_plural = plural in raw
    if has_singular == has_plural:
        raise ValueError(f"experiment must define exactly one of {singular!r} or {plural!r}")
    value = raw[singular] if has_singular else raw[plural]
    if has_singular:
        values = (value,)
    elif plural == "layers" and value == "all":
        if upper_bound is None:
            raise ValueError("layers='all' requires a known model layer count")
        values = tuple(range(upper_bound))
    else:
        if not isinstance(value, list) or not value:
            raise ValueError(f"experiment field {plural} must be a non-empty array")
        values = tuple(value)

    parsed: list[int] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"experiment field {plural} must contain integers")
        if item < 0:
            raise ValueError(f"experiment field {plural} cannot contain negative values")
        if upper_bound is not None and item >= upper_bound:
            raise ValueError(f"{singular} {item} is outside model {singular} range")
        parsed.append(item)
    result = tuple(parsed)
    if result != tuple(sorted(set(result))):
        raise ValueError(f"experiment field {plural} must be sorted and unique")
    return result
