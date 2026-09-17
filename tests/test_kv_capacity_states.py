from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from sieve_replay.model.capacity import classify_capacity

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_kv_capacity_states.py"
SPEC = importlib.util.spec_from_file_location("run_kv_capacity_states", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CAPACITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAPACITY)


class KvCapacityStatesTest(unittest.TestCase):
    def test_default_zero_spill_and_exact_boundary(self) -> None:
        resident = classify_capacity(80, 80)
        self.assertEqual(resident["state"], "resident")
        self.assertTrue(resident["feasible"])
        overflow = classify_capacity(81, 80)
        self.assertEqual(overflow["state"], "oom")
        self.assertFalse(overflow["feasible"])
        self.assertEqual(overflow["admission_status"], "infeasible")
        self.assertEqual(overflow["unallocated_bytes"], 1)
        self.assertEqual(overflow["spill_bytes"], 0)

    def test_explicit_hypothetical_spill_budget(self) -> None:
        # These are unit-test budgets, not new hardware assumptions.
        admitted = classify_capacity(95, 80, 20)
        self.assertEqual(admitted["state"], "spill")
        self.assertTrue(admitted["feasible"])
        self.assertEqual(admitted["resident_bytes"], 80)
        self.assertEqual(admitted["spill_bytes"], 15)
        self.assertEqual(admitted["spill_headroom_bytes"], 5)
        overflow = classify_capacity(101, 80, 20)
        self.assertEqual(overflow["state"], "oom")
        self.assertEqual(overflow["unallocated_bytes"], 1)
        for total in (0, 80, 95, 100, 101):
            result = classify_capacity(total, 80, 20)
            self.assertEqual(result["resident_bytes"] + result["spill_bytes"] + result["unallocated_bytes"], total)

    def test_invalid_byte_counts(self) -> None:
        for arguments in ((-1, 80), (1, -1), (1, 80, -1), (True, 80), (1, 80.0)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                classify_capacity(*arguments)

    def test_stress_matrix_separates_device_domains(self) -> None:
        experiment = ROOT / "configs/experiments/full_decode_real_pilot_cycle_v0.json"
        with tempfile.TemporaryDirectory() as temporary:
            result = CAPACITY.run_capacity_states(experiment, temporary)
            rows = result["rows"]
            self.assertEqual([(r["batch_size"], r["context_length"]) for r in rows], list(CAPACITY.CAPACITY_MATRIX))
            self.assertEqual([r["a800_state"] for r in rows], ["oom"] * 4)
            self.assertEqual([r["simulated_state"] for r in rows], ["resident", "resident", "oom", "oom"])
            self.assertTrue(all(r["a800_admission_status"] == "infeasible" for r in rows))
            self.assertTrue(all(r["a800_spill_capacity_bytes"] == r["simulated_spill_capacity_bytes"] == 0 for r in rows))
            self.assertTrue(all(not r["timing_modelled"] for r in rows))
            self.assertTrue(all("overall_state" not in r for r in rows))
            payload = json.loads((Path(temporary) / "capacity_states.json").read_text())
            metadata = payload["metadata"]
            self.assertEqual(metadata["capacity_domains"]["a800"]["resident_capacity_bytes"], 80_000_000_000)
            self.assertEqual(metadata["capacity_domains"]["simulated"]["resident_capacity_bytes"], 96_000_000_000)
            self.assertTrue(metadata["manifest_validated"])
            self.assertEqual(set(metadata["input_hashes"]), {"experiment", "trace", "trace_manifest", "prompts", "model", "hardware"})
            self.assertEqual(set(metadata["config_snapshot"]), {"experiment", "model", "hardware"})
            self.assertTrue((Path(temporary) / "capacity_states.csv").is_file())

    def test_manifest_hash_mismatch_fails_before_output(self) -> None:
        experiment = ROOT / "configs/experiments/full_decode_real_pilot_cycle_v0.json"
        configuration = CAPACITY.load_configuration(experiment)
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            original = configuration.experiment.trace_manifest_path
            assert original is not None
            manifest = json.loads(original.read_text())
            manifest["trace_file"] = str(configuration.experiment.trace_path)
            manifest["trace_sha256"] = "0" * 64
            manifest_path = work / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            invalid = replace(configuration, experiment=replace(configuration.experiment, trace_manifest_path=manifest_path))
            with mock.patch.object(CAPACITY, "load_configuration", return_value=invalid), self.assertRaisesRegex(ValueError, "SHA-256"):
                CAPACITY.run_capacity_states(experiment, work / "output")
            self.assertFalse((work / "output").exists())


if __name__ == "__main__":
    unittest.main()
