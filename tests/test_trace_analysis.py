from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from sieve_replay.config import load_configuration
from sieve_replay.trace import load_trace_set
from sieve_replay.trace_analysis import analyze_router_trace, write_router_trace_analysis


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "configs/experiments/full_decode_synthetic.json"


class TraceAnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.configuration = load_configuration(EXPERIMENT)
        experiment = self.configuration.experiment
        self.trace_set = load_trace_set(
            experiment.trace_path,
            self.configuration.model,
            experiment.layers,
            experiment.steps,
        )

    def test_synthetic_trace_exposes_repeated_load_shape(self) -> None:
        analysis = analyze_router_trace(
            self.trace_set,
            self.configuration.model,
            self.configuration.experiment.trace_path,
            trace_kind="synthetic-validation",
        )
        self.assertEqual(analysis.summary["selection"]["layer_batches"], 96)
        self.assertEqual(analysis.summary["selection"]["records"], 768)
        self.assertEqual(analysis.summary["load_shape"]["unique_signatures"], 1)
        self.assertEqual(
            analysis.summary["load_shape"]["dominant_signature_fraction"], 1.0
        )
        self.assertEqual(len(analysis.transitions), 48)
        self.assertTrue(all(row["same_load_signature"] for row in analysis.transitions))

    def test_analysis_writes_machine_readable_tables(self) -> None:
        analysis = analyze_router_trace(
            self.trace_set,
            self.configuration.model,
            self.configuration.experiment.trace_path,
            trace_kind="synthetic-validation",
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            write_router_trace_analysis(output, analysis)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            with (output / "batches.csv").open(encoding="utf-8", newline="") as handle:
                batches = list(csv.DictReader(handle))
            with (output / "transitions.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                transitions = list(csv.DictReader(handle))
        self.assertEqual(summary["trace_kind"], "synthetic-validation")
        self.assertEqual(len(batches), 96)
        self.assertEqual(len(transitions), 48)


if __name__ == "__main__":
    unittest.main()
