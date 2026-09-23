from __future__ import annotations

import math
import random
import struct
from typing import Any, Iterable


FORMATS = ("fp32", "bf16", "fp16")


def quantize_scalar(value: float, scalar_format: str) -> float:
    if scalar_format == "fp32":
        return struct.unpack(">f", struct.pack(">f", value))[0]
    if scalar_format == "fp16":
        return struct.unpack(">e", struct.pack(">e", value))[0]
    if scalar_format != "bf16":
        raise ValueError(f"unsupported partial scalar format: {scalar_format}")
    bits = struct.unpack(">I", struct.pack(">f", value))[0]
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    rounded = (bits + rounding_bias) & 0xFFFF0000
    return struct.unpack(">f", struct.pack(">I", rounded))[0]


def _basis(position: int, seed: int) -> tuple[float, float, float, float]:
    angle = (position + 1) * (0.013 + (seed % 11) * 0.00017)
    saw = ((position * 1103515245 + seed * 12345) & 0xFFFF) / 32767.5 - 1.0
    alternating = 1.0 if (position + seed) % 2 == 0 else -1.0
    return math.sin(angle), math.cos(angle * 0.73), saw, alternating


def _coefficients(head_dim: int) -> list[tuple[float, float, float, float]]:
    return [
        (
            math.sin((dimension + 1) * 0.17),
            math.cos((dimension + 1) * 0.11),
            ((dimension * 37) % 29) / 14.0 - 1.0,
            ((dimension * 19) % 23) / 22.0 - 0.5,
        )
        for dimension in range(head_dim)
    ]


def _empty_state() -> list[float]:
    return [-math.inf, 0.0, 0.0, 0.0, 0.0, 0.0]


def _update_state(state: list[float], logit: float, basis: tuple[float, ...]) -> None:
    if logit > state[0]:
        scale = 0.0 if not math.isfinite(state[0]) else math.exp(state[0] - logit)
        state[1] = state[1] * scale + 1.0
        for index, value in enumerate(basis, start=2):
            state[index] = state[index] * scale + value
        state[0] = logit
    else:
        weight = math.exp(logit - state[0])
        state[1] += weight
        for index, value in enumerate(basis, start=2):
            state[index] += weight * value


def _numerator_vector(
    state: list[float], coefficients: list[tuple[float, float, float, float]]
) -> list[float]:
    return [
        sum(coefficient * state[index + 2] for index, coefficient in enumerate(row))
        for row in coefficients
    ]


def _direct_output(
    state: list[float], coefficients: list[tuple[float, float, float, float]]
) -> list[float]:
    return [value / state[1] for value in _numerator_vector(state, coefficients)]


def _merge_partials(
    states: list[list[float]],
    coefficients: list[tuple[float, float, float, float]],
    scalar_format: str,
) -> list[float]:
    partials: list[tuple[float, float, list[float]]] = []
    for state in states:
        partials.append(
            (
                quantize_scalar(state[0], scalar_format),
                quantize_scalar(state[1], scalar_format),
                [
                    quantize_scalar(value, scalar_format)
                    for value in _numerator_vector(state, coefficients)
                ],
            )
        )
    global_max = max(partial[0] for partial in partials)
    denominator = 0.0
    output = [0.0] * len(coefficients)
    for partial_max, partial_sum, partial_numerator in partials:
        scale = math.exp(partial_max - global_max)
        denominator += partial_sum * scale
        for dimension, value in enumerate(partial_numerator):
            output[dimension] += value * scale
    return [value / denominator for value in output]


def _metrics(actual: list[float], reference: list[float]) -> dict[str, float]:
    differences = [actual_value - reference_value for actual_value, reference_value in zip(actual, reference)]
    squared_error = sum(value * value for value in differences)
    reference_squared = sum(value * value for value in reference)
    actual_squared = sum(value * value for value in actual)
    dot = sum(actual_value * reference_value for actual_value, reference_value in zip(actual, reference))
    denominator = math.sqrt(actual_squared * reference_squared)
    return {
        "max_abs_error": max(abs(value) for value in differences),
        "mean_abs_error": sum(abs(value) for value in differences) / len(differences),
        "rmse": math.sqrt(squared_error / len(differences)),
        "relative_l2_error": math.sqrt(squared_error) / max(math.sqrt(reference_squared), 1e-30),
        "cosine_similarity": dot / denominator if denominator else 1.0,
    }


def run_partial_attention_trial(
    *,
    context_length: int,
    total_pseudo_channels: int,
    logit_std: float,
    seed: int,
    head_dim: int,
) -> list[dict[str, Any]]:
    integer_fields = {
        "context_length": context_length,
        "total_pseudo_channels": total_pseudo_channels,
        "seed": seed,
        "head_dim": head_dim,
    }
    for name, value in integer_fields.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if logit_std <= 0:
        raise ValueError("logit_std must be positive")
    if context_length < total_pseudo_channels:
        raise ValueError("context must cover every pseudo-channel")

    coefficients = _coefficients(head_dim)
    reference_state = _empty_state()
    partial_states = [_empty_state() for _ in range(total_pseudo_channels)]
    generator = random.Random(seed * 1_000_003 + context_length * 97 + int(logit_std * 1000))
    for position in range(context_length):
        logit = generator.gauss(0.0, logit_std)
        basis = _basis(position, seed)
        _update_state(reference_state, logit, basis)
        _update_state(
            partial_states[position % total_pseudo_channels], logit, basis
        )

    reference = _direct_output(reference_state, coefficients)
    outputs = {
        scalar_format: _merge_partials(
            partial_states, coefficients, scalar_format
        )
        for scalar_format in FORMATS
    }
    rows: list[dict[str, Any]] = []
    fp32_output = outputs["fp32"]
    for scalar_format, output in outputs.items():
        row = {
            **integer_fields,
            "logit_std": logit_std,
            "scalar_format": scalar_format,
            **_metrics(output, reference),
        }
        incremental = _metrics(output, fp32_output)
        row.update(
            {
                f"vs_fp32_partial_{name}": value
                for name, value in incremental.items()
            }
        )
        rows.append(row)
    return rows

def run_partial_attention_sweep(
    *,
    context_lengths: Iterable[int],
    total_pseudo_channels: Iterable[int],
    logit_stds: Iterable[float],
    seeds: Iterable[int],
    head_dim: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for context_length in context_lengths:
        for pseudo_channels in total_pseudo_channels:
            for logit_std in logit_stds:
                for seed in seeds:
                    rows.extend(
                        run_partial_attention_trial(
                            context_length=context_length,
                            total_pseudo_channels=pseudo_channels,
                            logit_std=logit_std,
                            seed=seed,
                            head_dim=head_dim,
                        )
                    )
    return rows
