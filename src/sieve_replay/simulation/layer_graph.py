from __future__ import annotations

from ..timing import AnalyticTimingModel
from ..trace import TraceBatch
from ..types import AttentionTarget, ExpertLoad, PlacementDecision, TimingEstimate
from .event import Event


GPU_COMPUTE = "GPU_COMPUTE"
HBM_MEM_PATH = "HBM_MEM_PATH"
PIM_COMPUTE = "PIM_COMPUTE"
GPU_PIM_IO = "GPU_PIM_IO"


def _event(
    name: str,
    category: str,
    resources: tuple[str, ...],
    dependencies: tuple[str, ...],
    timing: TimingEstimate,
) -> Event:
    return Event(
        name=name,
        category=category,
        resources=resources,
        dependencies=dependencies,
        duration_us=timing.duration_us,
        description=timing.description,
    )


def build_layer_graph(
    trace: TraceBatch,
    decision: PlacementDecision,
    timing: AnalyticTimingModel,
) -> tuple[Event, ...]:
    by_id = {load.expert_id: load for load in trace.expert_loads}
    gpu_loads = tuple(by_id[expert] for expert in decision.gpu_experts)
    pim_loads = tuple(by_id[expert] for expert in decision.pim_experts)
    batch = trace.batch_size
    coupled = timing.coupled_expert_timing(gpu_loads, pim_loads)
    if coupled is None:
        gpu_weight_timing = timing.gpu_expert_weight_load(gpu_loads)
        pim_write_timing = timing.pim_token_write(pim_loads)
        pim_compute_timing = timing.pim_expert_compute(pim_loads)
        pim_read_timing = timing.pim_result_read(pim_loads)
        gpu_weight_resources = (HBM_MEM_PATH,)
        pim_io_resources = (GPU_PIM_IO,)
        pim_compute_resources = (PIM_COMPUTE,)
    else:
        gpu_weight_timing = coupled.gpu_weight_load
        pim_write_timing = coupled.pim_token_write
        pim_compute_timing = coupled.pim_expert_compute
        pim_read_timing = coupled.pim_result_read
        # The shared-resource arbitration is already resolved inside one mixed
        # Ramulator run; adding Python resources here would serialize it twice.
        gpu_weight_resources = ()
        pim_io_resources = ()
        pim_compute_resources = ()

    attention_timing = (
        timing.gpu_attention(trace.context_lengths)
        if decision.attention_target is AttentionTarget.GPU
        else timing.pim_attention(trace.context_lengths)
    )
    attention_resource = (
        (GPU_COMPUTE,) if decision.attention_target is AttentionTarget.GPU else (PIM_COMPUTE,)
    )

    events = (
        _event("norm1", "norm", (GPU_COMPUTE,), (), timing.norm(batch)),
        _event("qkv", "qkv", (GPU_COMPUTE,), ("norm1",), timing.qkv_projection(batch)),
        _event("rope", "rope", (GPU_COMPUTE,), ("qkv",), timing.rope(batch)),
        _event("attention", "attention", attention_resource, ("rope",), attention_timing),
        _event(
            "o_proj",
            "attention_output",
            (GPU_COMPUTE,),
            ("attention",),
            timing.output_projection(batch),
        ),
        _event("residual1", "residual", (GPU_COMPUTE,), ("o_proj",), timing.residual(batch)),
        _event("norm2", "norm", (GPU_COMPUTE,), ("residual1",), timing.norm(batch)),
        _event("router", "router", (GPU_COMPUTE,), ("norm2",), timing.router(batch)),
        _event(
            "metadata",
            "metadata",
            (GPU_COMPUTE,),
            ("router",),
            timing.metadata(trace.total_expert_assignments),
        ),
        _event("dispatch", "dispatch", (), ("metadata",), timing.dispatch()),
        _event("scheduler", "scheduler", (GPU_COMPUTE,), ("metadata",), timing.scheduler(decision.policy)),
        _event(
            "gpu_weight_load",
            "gpu_weight_load",
            gpu_weight_resources,
            ("scheduler",),
            gpu_weight_timing,
        ),
        _event(
            "gpu_expert_compute",
            "gpu_expert",
            (GPU_COMPUTE,),
            ("gpu_weight_load", "dispatch"),
            timing.gpu_expert_compute(gpu_loads),
        ),
        _event(
            "pim_gwrite",
            "pim_write",
            pim_io_resources,
            ("scheduler", "dispatch"),
            pim_write_timing,
        ),
        _event(
            "pim_expert_compute",
            "pim_expert",
            pim_compute_resources,
            ("pim_gwrite",),
            pim_compute_timing,
        ),
        _event(
            "pim_read",
            "pim_read",
            pim_io_resources,
            ("pim_expert_compute",),
            pim_read_timing,
        ),
        _event(
            "combine",
            "combine",
            (GPU_COMPUTE,),
            ("gpu_expert_compute", "pim_read"),
            timing.combine(batch),
        ),
        _event("residual2", "residual", (GPU_COMPUTE,), ("combine",), timing.residual(batch)),
    )
    _validate_placement(trace.expert_loads, decision)
    return events


def _validate_placement(loads: tuple[ExpertLoad, ...], decision: PlacementDecision) -> None:
    active = {load.expert_id for load in loads}
    gpu = set(decision.gpu_experts)
    pim = set(decision.pim_experts)
    if gpu & pim:
        raise ValueError("GPU and PIM expert placements overlap")
    if gpu | pim != active:
        missing = active - (gpu | pim)
        extra = (gpu | pim) - active
        raise ValueError(f"placement does not cover active experts; missing={missing}, extra={extra}")
