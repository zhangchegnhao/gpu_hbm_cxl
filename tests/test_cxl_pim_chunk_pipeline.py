from __future__ import annotations

import unittest
from pathlib import Path

from sieve_replay.ramulator.cxl_pim_attention import CXLPIMTopology
from sieve_replay.ramulator.cxl_pim_chunk_pipeline import (
    cxl_pim_chunk_pipeline_context,
    run_cxl_pim_chunk_pipeline,
)
from sieve_replay.ramulator.cxl_pim_pipeline import run_cxl_pim_pipeline
from sieve_replay.ramulator.mixed_workload import SieveCycleV1Config


ROOT = Path(__file__).resolve().parents[1]
CYCLE = ROOT / "configs/ramulator/sieve_hbm3e_cycle_v1.json"
RAMULATOR = ROOT / "third_party/ramulator2_cxl"
BINDING = RAMULATOR / "python/ramulator"


class CXLPIMChunkPipelineTest(unittest.TestCase):
    def test_context_binds_chunk_pipeline_sources(self) -> None:
        context = cxl_pim_chunk_pipeline_context(ROOT, CYCLE)
        self.assertEqual(context["model"], "unified-cxl-pim-chunk-pipeline-v1")
        self.assertEqual(len(context["extension_sha256"]), 64)
        self.assertEqual(len(context["runner_sha256"]), 64)

    def test_rejects_invalid_buffer_and_chunk_counts(self) -> None:
        cycle = SieveCycleV1Config.load(CYCLE)
        topology = CXLPIMTopology(channels=2)
        common = dict(
            query_link_transactions=17,
            pim_gwrite_waves=2,
            pim_mac_waves=3,
            pim_read_waves_per_chunk=2,
            result_link_transactions_per_chunk=19,
            cxl_bandwidth_bytes_per_second=64e9,
            cxl_latency_us=0.25,
            link_channels=2,
        )
        with self.assertRaisesRegex(ValueError, "buffer_slots cannot exceed"):
            run_cxl_pim_chunk_pipeline(
                RAMULATOR,
                cycle,
                topology,
                chunk_count=1,
                buffer_slots=2,
                **common,
            )
        with self.assertRaisesRegex(ValueError, "each chunk"):
            run_cxl_pim_chunk_pipeline(
                RAMULATOR,
                cycle,
                topology,
                chunk_count=4,
                buffer_slots=1,
                **common,
            )


@unittest.skipUnless(
    any(BINDING.glob("_ramulator*.so")),
    "CXL Ramulator binding is not built",
)
class CXLPIMChunkPipelineRamulatorTest(unittest.TestCase):
    def test_one_chunk_is_cycle_identical_to_stage10_pipeline(self) -> None:
        cycle = SieveCycleV1Config.load(CYCLE)
        topology = CXLPIMTopology(channels=2)
        common = dict(
            query_link_transactions=17,
            pim_gwrite_waves=2,
            pim_mac_waves=3,
            cxl_bandwidth_bytes_per_second=64e9,
            cxl_latency_us=0.25,
            link_channels=2,
        )
        stage10 = run_cxl_pim_pipeline(
            RAMULATOR,
            cycle,
            topology,
            pim_read_waves=2,
            result_link_transactions=19,
            **common,
        )
        chunked = run_cxl_pim_chunk_pipeline(
            RAMULATOR,
            cycle,
            topology,
            chunk_count=1,
            buffer_slots=1,
            pim_read_waves_per_chunk=2,
            result_link_transactions_per_chunk=19,
            **common,
        )
        stage10_cycles = (
            stage10.query_link_completion_cycles,
            stage10.pim_gwrite_start_cycles,
            stage10.pim_gwrite_completion_cycles,
            stage10.pim_mac_start_cycles,
            stage10.pim_mac_completion_cycles,
            stage10.pim_read_start_cycles,
            stage10.pim_read_completion_cycles,
            stage10.result_link_start_cycles,
            stage10.result_link_completion_cycles,
            stage10.pipeline_completion_cycles,
        )
        chunk_cycles = (
            chunked.query_link_completion_cycles,
            chunked.pim_gwrite_start_cycles,
            chunked.pim_gwrite_completion_cycles,
            chunked.chunk_mac_start_cycles[0],
            chunked.chunk_mac_completion_cycles[0],
            chunked.chunk_read_start_cycles[0],
            chunked.chunk_read_completion_cycles[0],
            chunked.chunk_result_link_start_cycles[0],
            chunked.chunk_result_link_completion_cycles[0],
            chunked.pipeline_completion_cycles,
        )
        self.assertEqual(chunk_cycles, stage10_cycles)
        self.assertEqual(chunked.max_active_buffers, 1)

    def test_two_slots_overlap_pim_and_result_link(self) -> None:
        cycle = SieveCycleV1Config.load(CYCLE)
        topology = CXLPIMTopology(channels=2)
        result = run_cxl_pim_chunk_pipeline(
            RAMULATOR,
            cycle,
            topology,
            chunk_count=4,
            buffer_slots=2,
            query_link_transactions=17,
            pim_gwrite_waves=2,
            pim_mac_waves=8,
            pim_read_waves_per_chunk=2,
            result_link_transactions_per_chunk=19,
            cxl_bandwidth_bytes_per_second=64e9,
            cxl_latency_us=0.25,
            link_channels=2,
        )
        self.assertEqual(result.max_active_buffers, 2)
        self.assertEqual(len(result.partial_ready_us), 4)
        self.assertEqual(tuple(sorted(result.partial_ready_us)), result.partial_ready_us)
        self.assertLess(
            result.chunk_mac_start_cycles[1],
            result.chunk_result_link_completion_cycles[0],
        )


if __name__ == "__main__":
    unittest.main()
