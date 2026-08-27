from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from sieve_replay.cli import main
from sieve_replay.config import load_configuration
from sieve_replay.replay import run_experiment
from sieve_replay.trace import load_trace_set


ROOT = Path(__file__).resolve().parents[1]
FULL_DECODE = ROOT / "configs/experiments/full_decode_synthetic.json"
FULL_CYCLE_V1 = ROOT / "configs/experiments/full_decode_cycle_v1.json"
SINGLE_LAYER = ROOT / "configs/experiments/single_layer_smoke.json"


class DecodeReplayTest(unittest.TestCase):
    def test_full_decode_trace_contains_every_selected_batch(self) -> None:
        loaded = load_configuration(FULL_DECODE)
        trace_set = load_trace_set(
            loaded.experiment.trace_path,
            loaded.model,
            loaded.experiment.layers,
            loaded.experiment.steps,
        )
        self.assertEqual(len(trace_set.batches), 96)
        self.assertEqual(trace_set.layers, tuple(range(48)))
        self.assertEqual(trace_set.steps, (0, 1))
        self.assertEqual(trace_set.batch_size, 8)

    def test_decode_replay_serializes_layers_and_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = run_experiment(FULL_DECODE, "sieve", temporary)
            self.assertEqual(len(result.units), 96)
            self.assertEqual(len(result.events), 96 * 18)
            by_name = {event.event.name: event for event in result.events}
            self.assertEqual(
                by_name["step0.layer1.norm1"].event.dependencies,
                ("step0.layer0.residual2",),
            )
            self.assertEqual(
                by_name["step1.layer0.norm1"].event.dependencies,
                ("step0.layer47.residual2",),
            )
            self.assertEqual(len(result.summary["steps"]), 2)
            self.assertEqual(len(result.summary["layers"]), 96)
            self.assertTrue((Path(temporary) / "steps.csv").is_file())
            self.assertTrue((Path(temporary) / "layers.csv").is_file())

    def test_plural_single_selection_preserves_legacy_result(self) -> None:
        with tempfile.TemporaryDirectory() as legacy_output:
            legacy = run_experiment(SINGLE_LAYER, "sieve", legacy_output)
            raw = json.loads(SINGLE_LAYER.read_text(encoding="utf-8"))
            raw.pop("layer")
            raw.pop("step")
            raw["layers"] = [0]
            raw["steps"] = [0]
            with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
                experiment = Path(temporary) / "plural_single.json"
                experiment.write_text(json.dumps(raw), encoding="utf-8")
                result = run_experiment(
                    experiment, "sieve", Path(temporary) / "output"
                )
        self.assertEqual(result.summary["total_latency_us"], legacy.summary["total_latency_us"])
        self.assertEqual(result.summary["placement"], legacy.summary["placement"])

    def test_full_cycle_v1_writes_layer_and_policy_contention_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            self.assertEqual(
                main(
                    [
                        "run-all",
                        "--experiment",
                        str(FULL_CYCLE_V1),
                        "--output",
                        str(output),
                    ]
                ),
                0,
            )
            with (output / "contention.csv").open(encoding="utf-8", newline="") as handle:
                policy_rows = list(csv.DictReader(handle))
            self.assertEqual(len(policy_rows), 6)
            self.assertTrue(all(row["layer_records"] == "96" for row in policy_rows))

            summary = json.loads(
                (output / "sieve-cycle-v1/summary.json").read_text(encoding="utf-8")
            )
            by_layer = summary["contention_by_layer"]
            aggregate = summary["contention_summary"]
            self.assertEqual(len(by_layer), 96)
            self.assertEqual(aggregate["layer_records"], 96)
            self.assertAlmostEqual(
                aggregate["totals"]["gpu_contended_us"],
                sum(row["gpu_contended_us"] for row in by_layer),
            )
            with (output / "sieve-cycle-v1/contention.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 96)


if __name__ == "__main__":
    unittest.main()
