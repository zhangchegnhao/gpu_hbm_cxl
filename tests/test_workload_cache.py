from __future__ import annotations

import tempfile
import unittest

from sieve_replay.ramulator.mixed_workload import MixedWorkloadResult
from sieve_replay.ramulator.workload_cache import (
    ContentionWorkloadCache,
    WorkloadShape,
)


class WorkloadCacheTest(unittest.TestCase):
    def test_cache_uses_exact_shape_and_provenance(self) -> None:
        context = {
            "cycle_config_sha256": "1" * 64,
            "extension_sha256": "2" * 64,
            "mixed_workload_sha256": "3" * 64,
        }
        shape = WorkloadShape(10, 20, 30, 40)
        result = MixedWorkloadResult(
            tick_ps=1000,
            gpu_read_transactions=10,
            pim_gwrite_waves=20,
            pim_mac_waves=30,
            pim_read_waves=40,
            gpu_completion_cycles=1,
            pim_completion_cycles=2,
            pim_gwrite_completion_cycles=1,
            pim_mac_completion_cycles=2,
            pim_read_completion_cycles=2,
            total_completion_cycles=2,
            gpu_injected_requests=10,
            gpu_completed_requests=10,
            pim_injected_requests=90,
            pim_completed_requests=90,
            gpu_blocked_by_pim_cycles=0,
            pim_blocked_by_gpu_cycles=0,
            pim_queue_wait_cycles=0,
            gpu_column_issues=10,
            pim_column_issues=90,
            pim_row_activations=1,
            pim_row_conflicts=0,
        )
        with tempfile.TemporaryDirectory() as temporary:
            cache = ContentionWorkloadCache(temporary, context)
            cache.store(shape, result)
            self.assertEqual(cache.load(shape), result)
            self.assertIsNone(cache.load(WorkloadShape(11, 20, 30, 40)))
            stale = ContentionWorkloadCache(
                temporary, {**context, "extension_sha256": "4" * 64}
            )
            self.assertIsNone(stale.load(shape))


if __name__ == "__main__":
    unittest.main()
