from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sieve_replay.cli import main
from sieve_replay.replay import run_experiment


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "configs/experiments/single_layer_smoke.json"
CYCLE_V1_EXPERIMENT = ROOT / "configs/experiments/single_layer_cycle_v1.json"


class ReplayIntegrationTest(unittest.TestCase):
    def test_sieve_replay_writes_reproducible_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = run_experiment(EXPERIMENT, "sieve", output)
            self.assertGreater(result.summary["total_latency_us"], 0.0)
            self.assertTrue(result.summary["memory"]["feasible"])
            self.assertTrue((output / "events.csv").is_file())
            self.assertTrue((output / "placement.csv").is_file())
            self.assertTrue((output / "run_manifest.json").is_file())
            persisted = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted, result.summary)

    def test_combine_waits_for_both_expert_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = run_experiment(EXPERIMENT, "sieve", temporary)
            events = {event.event.name: event for event in result.events}
            expected_start = max(
                events["gpu_expert_compute"].end_us,
                events["pim_read"].end_us,
            )
            self.assertAlmostEqual(events["combine"].start_us, expected_start)

    def test_run_all_writes_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            exit_code = main(
                [
                    "run-all",
                    "--experiment",
                    str(EXPERIMENT),
                    "--output",
                    temporary,
                ]
            )
            self.assertEqual(exit_code, 0)
            comparison = json.loads(
                (Path(temporary) / "comparison.json").read_text(encoding="utf-8")
            )
            self.assertEqual({row["policy"] for row in comparison}, {
                "gpu-only", "noexp", "allexp", "pimoe", "sieve"
            })

    def test_cycle_v1_replay_uses_coupled_expert_milestones(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = run_experiment(CYCLE_V1_EXPERIMENT, "sieve", temporary)
            events = {event.event.name: event for event in result.events}
            pim_path = sum(
                events[name].event.duration_us
                for name in ("pim_gwrite", "pim_expert_compute", "pim_read")
            )
            self.assertAlmostEqual(pim_path, result.summary["contention"]["pim_contended_us"])
            self.assertAlmostEqual(
                events["gpu_weight_load"].event.duration_us,
                result.summary["contention"]["gpu_contended_us"],
            )
            self.assertEqual(events["gpu_weight_load"].event.resources, ())

    def test_cycle_v1_search_policy_reports_candidate_objectives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = run_experiment(
                CYCLE_V1_EXPERIMENT, "sieve-cycle-v1", temporary
            )
            search = result.summary["placement_search"]
            self.assertEqual(
                search["objective"],
                "scheduler + max(gpu_contended + gpu_compute, pim_contended)",
            )
            self.assertGreaterEqual(search["candidate_count"], 2)
            self.assertAlmostEqual(
                result.summary["estimated_policy_objective_us"],
                min(candidate["objective_us"] for candidate in search["candidates"]),
            )
            self.assertTrue((Path(temporary) / "placement_search.csv").is_file())


if __name__ == "__main__":
    unittest.main()
