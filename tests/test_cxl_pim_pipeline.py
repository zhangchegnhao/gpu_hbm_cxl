from __future__ import annotations

import unittest
from pathlib import Path

from sieve_replay.ramulator.cxl_pim_attention import CXLPIMTopology
from sieve_replay.ramulator.cxl_pim_pipeline import (
    cxl_pim_pipeline_context,
    run_cxl_pim_pipeline,
)
from sieve_replay.ramulator.mixed_workload import SieveCycleV1Config


ROOT = Path(__file__).resolve().parents[1]
CYCLE = ROOT / "configs/ramulator/sieve_hbm3e_cycle_v1.json"
RAMULATOR = ROOT / "third_party/ramulator2_cxl"
BINDING = RAMULATOR / "python/ramulator"


class CXLPIMPipelineTest(unittest.TestCase):
    def test_context_binds_unified_pipeline_sources(self) -> None:
        context = cxl_pim_pipeline_context(ROOT, CYCLE)
        self.assertEqual(context["model"], "unified-cxl-pim-pipeline-v1")
        self.assertEqual(len(context["extension_sha256"]), 64)
        self.assertEqual(len(context["runner_sha256"]), 64)


@unittest.skipUnless(
    any(BINDING.glob("_ramulator*.so")),
    "CXL Ramulator binding is not built",
)
class CXLPIMPipelineRamulatorTest(unittest.TestCase):
    def test_five_phases_complete_in_dependency_order(self) -> None:
        cycle = SieveCycleV1Config.load(CYCLE)
        topology = CXLPIMTopology(channels=2)
        result = run_cxl_pim_pipeline(
            RAMULATOR,
            cycle,
            topology,
            query_link_transactions=17,
            pim_gwrite_waves=2,
            pim_mac_waves=3,
            pim_read_waves=2,
            result_link_transactions=19,
            cxl_bandwidth_bytes_per_second=64e9,
            cxl_latency_us=0.25,
            link_channels=2,
        )
        self.assertEqual(result.query_link_completed_requests, 17)
        self.assertEqual(result.pim_gwrite_completed_requests, 8)
        self.assertEqual(result.pim_mac_completed_requests, 12)
        self.assertEqual(result.pim_read_completed_requests, 8)
        self.assertEqual(result.result_link_completed_requests, 19)
        self.assertEqual(result.controller_link_requests, 36)
        self.assertEqual(result.controller_pim_gwrite_requests, 8)
        self.assertEqual(result.controller_pim_mac_requests, 12)
        self.assertEqual(result.controller_pim_read_requests, 8)
        milestones = (
            result.query_link_completion_cycles,
            result.pim_gwrite_start_cycles,
            result.pim_gwrite_completion_cycles,
            result.pim_mac_start_cycles,
            result.pim_mac_completion_cycles,
            result.pim_read_start_cycles,
            result.pim_read_completion_cycles,
            result.result_link_start_cycles,
            result.result_link_completion_cycles,
        )
        self.assertEqual(tuple(sorted(milestones)), milestones)
        self.assertEqual(
            result.pipeline_completion_cycles,
            result.result_link_completion_cycles,
        )


if __name__ == "__main__":
    unittest.main()
