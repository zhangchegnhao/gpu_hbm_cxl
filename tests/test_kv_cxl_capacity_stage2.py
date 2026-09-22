from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_kv_cxl_capacity_stage2.py"
SPEC = importlib.util.spec_from_file_location("run_kv_cxl_capacity_stage2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CXL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CXL)


class KvCxlCapacityStage2Test(unittest.TestCase):
    def test_stage2_matrix_separates_local_domains_and_cxl_budget(self) -> None:
        experiment = ROOT / "configs/experiments/full_decode_real_pilot_cycle_v0.json"
        with tempfile.TemporaryDirectory() as temporary:
            payload = CXL.run_cxl_capacity_stage2(experiment, temporary)
            rows = payload["rows"]
            self.assertEqual(len(rows), 4 * 2 * 4)
            self.assertTrue(payload["metadata"]["manifest_validated"])
            self.assertEqual(payload["metadata"]["cxl_budget_gb"], [0, 16, 32, 64])
            self.assertTrue((Path(temporary) / "cxl_capacity_states.csv").is_file())
            self.assertTrue((Path(temporary) / "stage2_report.md").is_file())

    def test_zero_budget_matches_formal_capacity_admission(self) -> None:
        experiment = ROOT / "configs/experiments/full_decode_real_pilot_cycle_v0.json"
        with tempfile.TemporaryDirectory() as temporary:
            payload = CXL.run_cxl_capacity_stage2(experiment, temporary, [0])
        rows = payload["rows"]
        self.assertEqual(len(rows), 8)
        for row in rows:
            self.assertEqual(row["cxl_capacity_bytes"], 0)
            self.assertEqual(row["spill_bytes"], 0)
            self.assertEqual(row["spill_destination"], "none")
        self.assertEqual(
            [row["state"] for row in rows if row["domain"] == "a800"],
            ["oom"] * 4,
        )
        self.assertEqual(
            [row["state"] for row in rows if row["domain"] == "simulated"],
            ["resident", "resident", "oom", "oom"],
        )

    def test_cxl_budget_can_admit_spill_without_changing_local_capacity(self) -> None:
        experiment = ROOT / "configs/experiments/full_decode_real_pilot_cycle_v0.json"
        with tempfile.TemporaryDirectory() as temporary:
            payload = CXL.run_cxl_capacity_stage2(experiment, temporary, [32, 64])
        row = next(
            row
            for row in payload["rows"]
            if row["domain"] == "a800"
            and row["batch_size"] == 8
            and row["context_length"] == 32768
            and row["cxl_capacity_bytes"] == 64_000_000_000
        )
        self.assertEqual(row["state"], "spill")
        self.assertTrue(row["feasible"])
        self.assertEqual(row["local_capacity_bytes"], 80_000_000_000)
        self.assertGreater(row["spill_bytes"], 0)
        self.assertEqual(row["unallocated_bytes"], 0)

    def test_metadata_records_hashes_and_memory_only_limits(self) -> None:
        experiment = ROOT / "configs/experiments/full_decode_real_pilot_cycle_v0.json"
        with tempfile.TemporaryDirectory() as temporary:
            payload = CXL.run_cxl_capacity_stage2(experiment, temporary, [0])
            saved = json.loads((Path(temporary) / "cxl_capacity_states.json").read_text())
        self.assertEqual(saved["metadata"]["method"], "analytic-kv-cxl-capacity-admission-v1")
        self.assertEqual(set(saved["metadata"]["input_hashes"]), {"experiment", "trace", "trace_manifest", "prompts", "model", "hardware"})
        self.assertIn("No CXL link", " ".join(payload["metadata"]["limitations"]))
        self.assertEqual(saved["metadata"]["rows"], 8)


if __name__ == "__main__":
    unittest.main()
