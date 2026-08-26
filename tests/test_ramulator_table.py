from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from sieve_replay.config import load_configuration
from sieve_replay.timing import ExpertGemvTiming, RamulatorTableTimingModel
from sieve_replay.timing.ramulator_table import PINNED_RAMULATOR_COMMIT, RamulatorTimingTable
from sieve_replay.types import ExpertLoad


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "configs/experiments/single_layer_smoke.json"


class RamulatorTimingTableTest(unittest.TestCase):
    def test_strict_table_lookup(self) -> None:
        raw = {
            "schema_version": 1,
            "metadata": {
                "units": "us",
                "ramulator_commit": PINNED_RAMULATOR_COMMIT,
                "extension_commit": "test-extension",
                "hardware_config_sha256": "a" * 64,
                "trace_generator_sha256": "b" * 64,
            },
            "attention": [{"batch_size": 8, "context_length": 1024, "duration_us": 12.5}],
            "expert_gemv": [
                {"token_count": 1, "gwrite_us": 0.1, "compute_us": 2.0, "read_us": 0.2}
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "table.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            table = RamulatorTimingTable.load(path)
            self.assertEqual(table.attention_us(8, 1024), 12.5)
            self.assertEqual(table.expert_us(1).compute_us, 2.0)
            with self.assertRaises(ValueError):
                table.attention_us(4, 1024)

    def test_rejects_wrong_ramulator_revision(self) -> None:
        raw = {
            "schema_version": 1,
            "metadata": {
                "units": "us",
                "ramulator_commit": "0" * 40,
                "extension_commit": "test-extension",
                "hardware_config_sha256": "a" * 64,
                "trace_generator_sha256": "b" * 64,
            },
            "attention": [{"batch_size": 1, "context_length": 1, "duration_us": 1.0}],
            "expert_gemv": [
                {"token_count": 1, "gwrite_us": 0.0, "compute_us": 1.0, "read_us": 0.0}
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "table.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ValueError):
                RamulatorTimingTable.load(path)

    def test_cycle_model_uses_exact_table_entries(self) -> None:
        configuration = load_configuration(EXPERIMENT)
        hardware = replace(configuration.hardware, timing_backend="ramulator-table-v0")
        table = RamulatorTimingTable(
            metadata={},
            attention={(8, 1024): 12.5},
            expert_gemv={3: ExpertGemvTiming(0.3, 4.0, 0.2)},
        )
        timing = RamulatorTableTimingModel(configuration.model, hardware, table)
        loads = (ExpertLoad(expert_id=0, token_count=3),)
        self.assertEqual(timing.pim_attention((1024,) * 8).duration_us, 12.5)
        self.assertEqual(timing.pim_token_write(loads).duration_us, 0.3)
        self.assertEqual(timing.pim_expert_compute(loads).duration_us, 4.0)
        self.assertEqual(timing.pim_result_read(loads).duration_us, 0.2)
        with self.assertRaises(ValueError):
            timing.pim_expert_compute((ExpertLoad(expert_id=0, token_count=2),))


if __name__ == "__main__":
    unittest.main()
