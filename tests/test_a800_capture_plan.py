from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sieve_replay.capture.a800_metrics import CASES, build_plan, write_plan


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class A800CapturePlanTest(unittest.TestCase):
    def test_plan_validates_all_five_existing_inputs_without_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan = build_plan(PROJECT_ROOT, output_root=Path(temporary) / "cases")
        self.assertFalse(plan["execution_implemented"])
        self.assertEqual(plan["status"], "planned")
        self.assertEqual([case["case"] for case in plan["cases"]], [entry[0] for entry in CASES])
        for case in plan["cases"]:
            self.assertEqual(case["prompt_count"], case["batch_size"])
            self.assertEqual(case["router_records"], case["batch_size"] * 48 * 8)
            self.assertEqual(len(case["input_artifacts"]["router"]["sha256"]), 64)

    def test_write_plan_is_json_and_declares_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = write_plan(PROJECT_ROOT, Path(temporary) / "plan.json")
            raw = json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(raw["capture_kind"], "a800-timing-memory-v1")
        self.assertEqual(raw["measurement_protocol"]["repeats"], 3)
        self.assertFalse(raw["execution_implemented"])
        self.assertIn("timing.jsonl", raw["required_outputs"])

    def test_unknown_case_is_rejected(self) -> None:
        from sieve_replay.capture.a800_metrics import validate_case

        with self.assertRaisesRegex(ValueError, "unsupported A800 case"):
            validate_case(PROJECT_ROOT, "b1_c1k", output_root="/tmp/unused")


if __name__ == "__main__":
    unittest.main()
