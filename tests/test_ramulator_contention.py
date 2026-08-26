from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from sieve_replay.ramulator import SieveCycleV1Config, run_mixed_workload
from sieve_replay.timing import RamulatorContentionTable
from sieve_replay.timing.ramulator_table import PINNED_RAMULATOR_COMMIT
from sieve_replay.types import ExpertLoad


ROOT = Path(__file__).resolve().parents[1]
CYCLE_CONFIG = ROOT / "configs/ramulator/sieve_hbm3e_cycle_v1.json"
RAMULATOR_ROOT = ROOT / "third_party/ramulator2"
BINDING_ROOT = RAMULATOR_ROOT / "python/ramulator"


def _table() -> dict[str, object]:
    return {
        "schema_version": 1,
        "metadata": {
            "units": "us",
            "ramulator_commit": PINNED_RAMULATOR_COMMIT,
            "extension_commit": "cycle-v1-test",
            "cycle_config_sha256": "a" * 64,
            "isolated_timing_table_sha256": "b" * 64,
            "trace_sha256": "c" * 64,
            "generator_sha256": "d" * 64,
            "dual_row_buffer": True,
        },
        "expert_contention": [
            {
                "gpu_experts": [0],
                "pim_experts": [1],
                "gpu_token_count": 3,
                "pim_token_count": 2,
                "gpu_read_transactions": 16,
                "pim_gwrite_waves": 2,
                "pim_mac_waves": 3,
                "pim_read_waves": 1,
                "gpu_isolated_us": 1.0,
                "pim_isolated_us": 2.0,
                "gpu_contended_us": 1.1,
                "pim_contended_us": 2.1,
                "pim_gwrite_contended_us": 0.2,
                "pim_compute_contended_us": 1.8,
                "pim_read_contended_us": 0.1,
                "total_memory_phase_us": 2.1,
                "gpu_blocked_by_pim_cycles": 4,
                "pim_blocked_by_gpu_cycles": 5,
                "pim_queue_wait_cycles": 6,
                "gpu_column_issues": 16,
                "pim_column_issues": 1536,
                "pim_row_activations": 256,
                "pim_row_conflicts": 0,
            }
        ],
    }


class RamulatorContentionTableTest(unittest.TestCase):
    def test_strict_exact_placement_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "contention.json"
            path.write_text(json.dumps(_table()), encoding="utf-8")
            table = RamulatorContentionTable.load(path)
            row = table.expert_timing(
                (ExpertLoad(expert_id=0, token_count=3),),
                (ExpertLoad(expert_id=1, token_count=2),),
            )
            self.assertEqual(row.total_memory_phase_us, 2.1)
            with self.assertRaises(ValueError):
                table.expert_timing(
                    (ExpertLoad(expert_id=0, token_count=2),),
                    (ExpertLoad(expert_id=1, token_count=3),),
                )

    def test_formal_table_rejects_single_row_buffer(self) -> None:
        raw = _table()
        raw["metadata"]["dual_row_buffer"] = False  # type: ignore[index]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "contention.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ValueError):
                RamulatorContentionTable.load(path)

    def test_schema_v2_supports_search_candidates_without_isolated_runs(self) -> None:
        raw = _table()
        raw["schema_version"] = 2
        raw["metadata"].update(  # type: ignore[union-attr]
            {
                "candidate_space": "hot-prefix",
                "search_method": "monotonic-crossing-bisection-v1",
                "model_sha256": "e" * 64,
                "hardware_sha256": "f" * 64,
                "search_total_prefixes": 3,
                "search_evaluated_prefixes": [0, 1, 2],
                "search_selected_prefix": 1,
                "search_measured_monotonicity_verified": True,
            }
        )
        row = raw["expert_contention"][0]  # type: ignore[index]
        row["gpu_isolated_us"] = None
        row["pim_isolated_us"] = None
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "contention-v2.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            table = RamulatorContentionTable.load(path)
            candidates = table.candidate_timings(
                (
                    ExpertLoad(expert_id=0, token_count=3),
                    ExpertLoad(expert_id=1, token_count=2),
                )
            )
            self.assertEqual(len(candidates), 1)
            self.assertIsNone(candidates[0].gpu_isolated_us)
            self.assertIsNone(candidates[0].as_report()["gpu_contention_delta_us"])

    def test_table_rejects_stale_input_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "contention.json"
            path.write_text(json.dumps(_table()), encoding="utf-8")
            for name in ("isolated.json", "trace.jsonl", "cycle.json"):
                (root / name).write_text("stale\n", encoding="utf-8")
            table = RamulatorContentionTable.load(path)
            with self.assertRaises(ValueError):
                table.validate_inputs(
                    root / "isolated.json",
                    root / "trace.jsonl",
                    root / "cycle.json",
                )


@unittest.skipUnless(
    any(BINDING_ROOT.glob("_ramulator*.so")),
    "Ramulator binding is not built",
)
class RamulatorMixedCycleTest(unittest.TestCase):
    def test_mixed_requests_complete_with_dual_row_buffers(self) -> None:
        cycle = SieveCycleV1Config.load(CYCLE_CONFIG)
        result = run_mixed_workload(
            RAMULATOR_ROOT,
            cycle,
            gpu_read_transactions=256,
            pim_gwrite_waves=1,
            pim_mac_waves=64,
            pim_read_waves=1,
        )
        self.assertEqual(result.gpu_injected_requests, result.gpu_completed_requests)
        self.assertEqual(result.pim_injected_requests, result.pim_completed_requests)
        self.assertGreater(result.gpu_blocked_by_pim_cycles, 0)
        self.assertGreater(result.pim_row_activations, 0)
        self.assertGreater(result.pim_row_conflicts, 0)

    def test_per_pseudo_channel_fifo_crosses_many_rows(self) -> None:
        cycle = SieveCycleV1Config.load(CYCLE_CONFIG)
        result = run_mixed_workload(
            RAMULATOR_ROOT,
            cycle,
            gpu_read_transactions=0,
            pim_gwrite_waves=0,
            pim_mac_waves=800,
            pim_read_waves=0,
        )
        self.assertEqual(result.pim_completed_requests, 800 * cycle.total_pseudo_channels)
        self.assertEqual(result.pim_row_activations, 25 * cycle.total_pseudo_channels)

    def test_single_row_buffer_ablation_is_supported(self) -> None:
        cycle = replace(SieveCycleV1Config.load(CYCLE_CONFIG), dual_row_buffer=False)
        result = run_mixed_workload(
            RAMULATOR_ROOT,
            cycle,
            gpu_read_transactions=0,
            pim_gwrite_waves=0,
            pim_mac_waves=64,
            pim_read_waves=0,
        )
        self.assertEqual(result.pim_completed_requests, 64 * cycle.total_pseudo_channels)
        self.assertGreater(result.pim_row_conflicts, 0)
