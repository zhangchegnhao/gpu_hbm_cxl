from __future__ import annotations

import unittest
from pathlib import Path

from sieve_replay.ramulator.cxl_kv_workload import (
    cxl_kv_cache_context,
    run_cxl_kv_workload,
)
from sieve_replay.ramulator.mixed_workload import SieveCycleV1Config


ROOT = Path(__file__).resolve().parents[1]
CYCLE = ROOT / "configs/ramulator/sieve_hbm3e_cycle_v1.json"
RAMULATOR = ROOT / "third_party/ramulator2_cxl"
BINDING = RAMULATOR / "python/ramulator"


class CXLKVWorkloadTest(unittest.TestCase):
    def test_context_binds_cxl_sources(self) -> None:
        context = cxl_kv_cache_context(ROOT, CYCLE)
        self.assertEqual(context["model"], "cxl-kv-memory-only-request-v1")
        self.assertEqual(len(context["extension_sha256"]), 64)


@unittest.skipUnless(
    any(BINDING.glob("_ramulator*.so")),
    "Ramulator binding is not built",
)
class CXLKVRamulatorTest(unittest.TestCase):
    def test_independent_cxl_queue_completes_exactly(self) -> None:
        cycle = SieveCycleV1Config.load(CYCLE)
        result = run_cxl_kv_workload(
            RAMULATOR,
            cycle,
            gpu_read_transactions=256,
            local_kv_read_transactions=512,
            cxl_kv_read_transactions=513,
            pim_gwrite_waves=1,
            pim_mac_waves=2,
            pim_read_waves=1,
            cxl_bandwidth_bytes_per_second=64e9,
            cxl_latency_us=0.25,
            cxl_channels=2,
        )
        self.assertEqual(result.gpu_completed_requests, 256)
        self.assertEqual(result.local_kv_completed_requests, 512)
        self.assertEqual(result.cxl_kv_completed_requests, 513)
        self.assertEqual(result.pim_completed_requests, 4 * cycle.total_pseudo_channels)
        self.assertGreater(result.cxl_queue_wait_cycles, 0)
        self.assertGreater(result.cxl_link_busy_ratio, 0.0)


if __name__ == "__main__":
    unittest.main()
