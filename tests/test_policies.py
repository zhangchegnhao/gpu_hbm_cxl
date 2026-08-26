from __future__ import annotations

import unittest
from pathlib import Path

from sieve_replay.config import load_configuration
from sieve_replay.policy import create_policy
from sieve_replay.timing import (
    AnalyticTimingModel,
    RamulatorContentionTable,
    RamulatorContentionTimingModel,
    RamulatorTimingTable,
)
from sieve_replay.trace import load_trace


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "configs/experiments/single_layer_smoke.json"
CYCLE_V1_EXPERIMENT = ROOT / "configs/experiments/single_layer_cycle_v1.json"


class PolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.loaded = load_configuration(EXPERIMENT)
        cls.trace = load_trace(
            cls.loaded.experiment.trace_path,
            cls.loaded.model,
            cls.loaded.experiment.layer,
            cls.loaded.experiment.step,
        )
        cls.timing = AnalyticTimingModel(cls.loaded.model, cls.loaded.hardware)

    def test_every_policy_partitions_each_active_expert_once(self) -> None:
        active = {load.expert_id for load in self.trace.expert_loads}
        for name in self.loaded.experiment.policies:
            with self.subTest(policy=name):
                decision = create_policy(name, self.timing).place(self.trace)
                gpu = set(decision.gpu_experts)
                pim = set(decision.pim_experts)
                self.assertFalse(gpu & pim)
                self.assertEqual(gpu | pim, active)

    def test_baseline_semantics(self) -> None:
        active_count = len(self.trace.expert_loads)
        gpu_only = create_policy("gpu-only", self.timing).place(self.trace)
        noexp = create_policy("noexp", self.timing).place(self.trace)
        allexp = create_policy("allexp", self.timing).place(self.trace)
        self.assertEqual(len(gpu_only.gpu_experts), active_count)
        self.assertEqual(len(noexp.gpu_experts), active_count)
        self.assertEqual(len(allexp.pim_experts), active_count)
        self.assertEqual(gpu_only.attention_target.value, "gpu")
        self.assertEqual(noexp.attention_target.value, "pim")

    def test_cycle_v1_policy_uses_exact_table_candidates(self) -> None:
        loaded = load_configuration(CYCLE_V1_EXPERIMENT)
        trace = load_trace(
            loaded.experiment.trace_path,
            loaded.model,
            loaded.experiment.layer,
            loaded.experiment.step,
        )
        assert loaded.experiment.pim_timing_table_path is not None
        assert loaded.experiment.contention_timing_table_path is not None
        timing = RamulatorContentionTimingModel(
            loaded.model,
            loaded.hardware,
            RamulatorTimingTable.load(loaded.experiment.pim_timing_table_path),
            RamulatorContentionTable.load(loaded.experiment.contention_timing_table_path),
        )
        decision = create_policy("sieve-cycle-v1", timing).place(trace)
        self.assertIsNotNone(decision.search_report)
        assert decision.search_report is not None
        self.assertGreaterEqual(decision.search_report["candidate_count"], 2)
        self.assertEqual(
            set(decision.gpu_experts) | set(decision.pim_experts),
            {load.expert_id for load in trace.expert_loads},
        )

    def test_cycle_v1_policy_rejects_analytic_backend(self) -> None:
        with self.assertRaises(ValueError):
            create_policy("sieve-cycle-v1", self.timing).place(self.trace)


if __name__ == "__main__":
    unittest.main()
