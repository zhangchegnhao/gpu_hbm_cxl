from __future__ import annotations

from dataclasses import dataclass

from ..config import ModelConfig
from ..trace import TraceBatch


@dataclass(frozen=True)
class MemoryFootprint:
    model_weights_bytes: int
    kv_cache_bytes: int
    activation_bytes: int
    total_bytes: int


def estimate_memory_footprint(model: ModelConfig, trace: TraceBatch) -> MemoryFootprint:
    h = model.hidden_size
    q = model.q_projection_size
    kv = model.kv_projection_size

    embedding_elements = model.vocab_size * h
    attention_elements_per_layer = h * (q + 2 * kv) + q * h
    expert_elements_per_layer = model.num_experts * model.expert_weight_elements
    router_elements_per_layer = h * model.num_experts
    norm_elements_per_layer = 2 * h
    final_norm_elements = h
    lm_head_elements = model.vocab_size * h
    total_elements = (
        embedding_elements
        + model.num_hidden_layers
        * (attention_elements_per_layer + expert_elements_per_layer + router_elements_per_layer + norm_elements_per_layer)
        + final_norm_elements
        + lm_head_elements
    )
    if model.has_shared_expert:
        total_elements += model.num_hidden_layers * model.expert_weight_elements
    model_weights = total_elements * model.dtype_bytes

    kv_cache_per_layer = (
        2
        * sum(trace.context_lengths)
        * model.num_key_value_heads
        * model.head_dim
        * model.dtype_bytes
    )
    kv_cache = model.num_hidden_layers * kv_cache_per_layer
    activation = (
        trace.batch_size
        * model.num_experts_per_tok
        * (2 * h + 2 * model.moe_intermediate_size)
        * model.dtype_bytes
    )
    return MemoryFootprint(
        model_weights_bytes=model_weights,
        kv_cache_bytes=kv_cache,
        activation_bytes=activation,
        total_bytes=model_weights + kv_cache + activation,
    )
