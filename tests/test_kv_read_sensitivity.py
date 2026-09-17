from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from scripts.run_kv_read_sensitivity import _write_comparison, variants
from sieve_replay.ramulator.kv_read_workload import KVReadWorkloadResult
from sieve_replay.ramulator.mixed_workload import SieveCycleV1Config


ROOT = Path(__file__).resolve().parents[1]


class KvReadSensitivityTest(unittest.TestCase):
    def test_variants_include_unchanged_and_queue_ablation(self) -> None:
        cycle = SieveCycleV1Config.load(ROOT / "configs/ramulator/sieve_hbm3e_cycle_v1.json")
        values = variants(cycle)
        self.assertEqual(values[0]["read_buffer_size"], cycle.read_buffer_size)
        self.assertEqual(values[0]["dual_row_buffer"], cycle.dual_row_buffer)
        self.assertEqual({item["read_buffer_size"] for item in values}, {32, 64, 128, 256})
        self.assertTrue(any(item["dual_row_buffer"] is False for item in values))

    def test_comparison_records_baseline_delta_and_provenance(self) -> None:
        fields = {name: 0 for name in KVReadWorkloadResult.__dataclass_fields__}
        fields.update({"tick_ps": 312, "gpu_read_transactions": 1,
                       "gpu_injected_requests": 1, "gpu_completed_requests": 1,
                       "gpu_completion_cycles": 10, "total_completion_cycles": 10,
                       "controller_read_completed_requests": 1,
                       "read_latency_cycles": 2,
                       "gpu_request_residence_cycles": 2})
        result = asdict(KVReadWorkloadResult(**fields))
        shape = {"gpu_read_transactions": 1, "pim_gwrite_waves": 0,
                 "pim_mac_waves": 0, "pim_read_waves": 0, "kv_read_transactions": 0}
        entries = {
            "baseline-rb256-dual:abc": {"variant": {"name": "baseline-rb256-dual"}, "shape_key": "abc", "shape": shape, "result": result},
            "rb32-dual:abc": {"variant": {"name": "rb32-dual"}, "shape_key": "abc", "shape": shape, "result": result},
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            exact_path = output / "kv_read_sensitivity_exact_results.json"
            exact_path.write_text(json.dumps({"entries": entries}), encoding="utf-8")
            _write_comparison(output, {"classification": "test"}, {"plan_sha256": "plan", "entries": entries})
            comparison = json.loads((output / "kv_read_sensitivity_comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(len(comparison["rows"]), 2)
            self.assertEqual(comparison["rows"][1]["delta_total_completion_us"], 0.0)
            self.assertIn("frozen_baseline_sha256", comparison)


if __name__ == "__main__":
    unittest.main()
