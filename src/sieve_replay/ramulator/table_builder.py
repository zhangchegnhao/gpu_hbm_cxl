from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..config import LoadedConfiguration, load_configuration
from ..report.writer import sha256_file
from ..timing.ramulator_table import PINNED_RAMULATOR_COMMIT
from ..trace import load_trace
from .microbenchmark import MicrobenchmarkResult, SieveCycleConfig, run_milestones


def candidate_token_counts(configuration: LoadedConfiguration) -> tuple[int, ...]:
    trace = load_trace(
        configuration.experiment.trace_path,
        configuration.model,
        configuration.experiment.layer,
        configuration.experiment.step,
    )
    remaining = trace.total_expert_assignments
    counts = {remaining}
    for load in sorted(trace.expert_loads, key=lambda row: (-row.token_count, row.expert_id)):
        remaining -= load.token_count
        if remaining > 0:
            counts.add(remaining)
    return tuple(sorted(counts))


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _combined_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _extension_hash(project_root: Path) -> str:
    extension_root = project_root / "ramulator/extensions/sieve_hbm_pim"
    return _combined_sha256(sorted(extension_root.rglob("*.cpp")) + sorted(extension_root.rglob("*.patch")))


def _trace_generator_hash(project_root: Path) -> str:
    package_root = project_root / "src/sieve_replay/ramulator"
    return _combined_sha256([package_root / "microbenchmark.py", package_root / "table_builder.py"])


def _evidence_row(result: MicrobenchmarkResult) -> dict[str, Any]:
    return {
        "operation": result.operation,
        "max_waves": result.max_waves,
        "tick_ps": result.tick_ps,
        "injected_requests": result.injected_requests,
        "completed_requests": result.completed_requests,
        "cycles_by_waves": {str(key): value for key, value in result.cycles_by_waves.items()},
        "duration_us_by_waves": {
            str(key): value for key, value in result.duration_us_by_waves.items()
        },
    }


def build_timing_table(
    experiment_path: str | Path,
    cycle_config_path: str | Path,
    ramulator_root: str | Path,
    output_path: str | Path,
    evidence_path: str | Path | None = None,
) -> dict[str, Any]:
    configuration = load_configuration(experiment_path)
    cycle_path = Path(cycle_config_path).resolve()
    cycle = SieveCycleConfig.load(cycle_path)
    if cycle.ramulator_commit != PINNED_RAMULATOR_COMMIT:
        raise ValueError("cycle configuration uses an unpinned Ramulator revision")
    model = configuration.model
    trace = load_trace(
        configuration.experiment.trace_path,
        model,
        configuration.experiment.layer,
        configuration.experiment.step,
    )
    if len(set(trace.context_lengths)) != 1:
        raise ValueError("timing table schema v1 requires a uniform context length within the batch")

    token_counts = candidate_token_counts(configuration)
    total_pseudo_channels = cycle.total_pseudo_channels
    operations_per_mac_wave = (
        total_pseudo_channels * cycle.banks_per_pseudo_channel * cycle.transaction_bytes
    )
    bytes_per_partitioned_wave = total_pseudo_channels * cycle.transaction_bytes

    expert_gwrite_waves = {
        tokens: tokens * _ceil_div(model.hidden_size * model.dtype_bytes, cycle.transaction_bytes)
        for tokens in token_counts
    }
    expert_mac_waves = {
        tokens: _ceil_div(
            6 * tokens * model.hidden_size * model.moe_intermediate_size,
            operations_per_mac_wave,
        )
        for tokens in token_counts
    }
    expert_read_waves = {
        tokens: _ceil_div(tokens * model.hidden_size * model.dtype_bytes, bytes_per_partitioned_wave)
        for tokens in token_counts
    }

    batch = trace.batch_size
    context = trace.context_lengths[0]
    attention_gwrite_waves = batch * _ceil_div(
        model.q_projection_size * model.dtype_bytes, cycle.transaction_bytes
    )
    attention_operations = 4 * sum(trace.context_lengths) * model.num_attention_heads * model.head_dim
    attention_mac_waves = _ceil_div(attention_operations, operations_per_mac_wave)
    attention_read_waves = _ceil_div(
        batch * model.q_projection_size * model.dtype_bytes, bytes_per_partitioned_wave
    )

    gwrite_milestones = set(expert_gwrite_waves.values()) | {attention_gwrite_waves}
    mac_milestones = set(expert_mac_waves.values()) | {attention_mac_waves}
    read_milestones = set(expert_read_waves.values()) | {attention_read_waves}
    gwrite = run_milestones(ramulator_root, cycle, "PIM_GWRITE", gwrite_milestones)
    mac = run_milestones(ramulator_root, cycle, "PIM_MAC", mac_milestones)
    read = run_milestones(ramulator_root, cycle, "PIM_READ", read_milestones)

    project_root = Path(__file__).resolve().parents[3]
    table = {
        "schema_version": 1,
        "metadata": {
            "units": "us",
            "ramulator_commit": cycle.ramulator_commit,
            "extension_commit": f"cycle-v0-sha256:{_extension_hash(project_root)}",
            "hardware_config_sha256": sha256_file(cycle_path),
            "trace_generator_sha256": _trace_generator_hash(project_root),
        },
        "attention": [
            {
                "batch_size": batch,
                "context_length": context,
                "duration_us": (
                    gwrite.duration_us_by_waves[attention_gwrite_waves]
                    + mac.duration_us_by_waves[attention_mac_waves]
                    + read.duration_us_by_waves[attention_read_waves]
                ),
            }
        ],
        "expert_gemv": [
            {
                "token_count": tokens,
                "gwrite_us": gwrite.duration_us_by_waves[expert_gwrite_waves[tokens]],
                "compute_us": mac.duration_us_by_waves[expert_mac_waves[tokens]],
                "read_us": read.duration_us_by_waves[expert_read_waves[tokens]],
            }
            for tokens in token_counts
        ],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def portable_path(path: str | Path) -> str:
        resolved = Path(path).resolve()
        try:
            return resolved.relative_to(project_root).as_posix()
        except ValueError:
            return str(resolved)

    evidence = {
        "schema_version": 1,
        "cycle_config": asdict(cycle),
        "experiment": portable_path(experiment_path),
        "token_counts": list(token_counts),
        "wave_model": {
            "operations_per_mac_wave": operations_per_mac_wave,
            "bytes_per_partitioned_wave": bytes_per_partitioned_wave,
            "attention_gwrite_waves": attention_gwrite_waves,
            "attention_mac_waves": attention_mac_waves,
            "attention_read_waves": attention_read_waves,
            "expert_gwrite_waves": expert_gwrite_waves,
            "expert_mac_waves": expert_mac_waves,
            "expert_read_waves": expert_read_waves,
        },
        "runs": [_evidence_row(gwrite), _evidence_row(mac), _evidence_row(read)],
        "table": portable_path(output),
    }
    evidence_output = (
        Path(evidence_path)
        if evidence_path is not None
        else output.with_name(f"{output.stem}.evidence.json")
    )
    evidence_output.parent.mkdir(parents=True, exist_ok=True)
    evidence_output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return table
