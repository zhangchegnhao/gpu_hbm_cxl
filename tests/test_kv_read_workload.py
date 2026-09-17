from __future__ import annotations

import unittest
from dataclasses import asdict, replace
from pathlib import Path

from sieve_replay.ramulator.kv_read_workload import (
    KVWorkloadShape,
    kv_read_cache_context,
    run_kv_read_workload,
)
from sieve_replay.ramulator.mixed_workload import SieveCycleV1Config, run_mixed_workload


ROOT = Path(__file__).resolve().parents[1]
CYCLE = ROOT / "configs/ramulator/sieve_hbm3e_cycle_v1.json"
RAMULATOR = ROOT / "third_party/ramulator2"
INSTALLED = RAMULATOR / "src/ramulator/frontend/impl/memory_trace/sieve_kv_read_frontend.cpp"


class KVReadShapeTest(unittest.TestCase):
    def test_separate_shape_defaults_and_rejects_invalid_counts(self) -> None:
        expert = KVWorkloadShape(1, 2, 3, 4)
        self.assertEqual(expert.kv_read_transactions, 0)
        self.assertNotEqual(expert, replace(expert, kv_read_transactions=1))
        for value in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                KVWorkloadShape(1, 0, 0, 0, value)

    def test_context_is_independent_of_legacy_workload_cache(self) -> None:
        context = kv_read_cache_context(ROOT, CYCLE)
        self.assertEqual(context["model"], "kv-read-concurrent-stress-v1")
        self.assertEqual(len(context["kv_read_workload_sha256"]), 64)
        self.assertEqual(len(context["microbenchmark_sha256"]), 64)

    def test_empty_workload_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must contain"):
            run_kv_read_workload(RAMULATOR, SieveCycleV1Config.load(CYCLE),
                                 gpu_read_transactions=0, pim_gwrite_waves=0,
                                 pim_mac_waves=0, pim_read_waves=0)


@unittest.skipUnless(
    INSTALLED.is_file() and any((RAMULATOR / "python/ramulator").glob("_ramulator*.so")),
    "KV READ Ramulator frontend is not built",
)
class KVReadCycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cycle = SieveCycleV1Config.load(CYCLE)

    def test_zero_kv_preserves_legacy_expert_timing(self) -> None:
        shape = dict(gpu_read_transactions=513, pim_gwrite_waves=1,
                     pim_mac_waves=64, pim_read_waves=1)
        old = run_mixed_workload(RAMULATOR, self.cycle, **shape)
        new = run_kv_read_workload(RAMULATOR, self.cycle, **shape)
        self.assertEqual(asdict(old), {key: asdict(new)[key] for key in asdict(old)})

    def test_kv_only_exact_counts_and_residence(self) -> None:
        result = run_kv_read_workload(
            RAMULATOR, self.cycle, gpu_read_transactions=0, pim_gwrite_waves=0,
            pim_mac_waves=0, pim_read_waves=0, kv_read_transactions=513,
        )
        self.assertEqual(result.kv_injected_requests, 513)
        self.assertEqual(result.kv_completed_requests, 513)
        self.assertEqual(result.controller_read_completed_requests, 513)
        self.assertEqual(result.gpu_completed_requests, 0)
        self.assertEqual(result.total_completion_cycles, result.kv_completion_cycles)
        self.assertGreater(result.kv_accepted_to_column_issue_cycles, 0)

    def test_shared_queue_has_exact_counts_and_both_streams_progress(self) -> None:
        result = run_kv_read_workload(
            RAMULATOR, replace(self.cycle, read_buffer_size=2),
            gpu_read_transactions=1024, pim_gwrite_waves=1, pim_mac_waves=64,
            pim_read_waves=1, kv_read_transactions=1024,
        )
        self.assertEqual(result.gpu_injected_requests, result.gpu_completed_requests)
        self.assertEqual(result.kv_injected_requests, result.kv_completed_requests)
        self.assertEqual(result.gpu_completed_requests, 1024)
        self.assertEqual(result.kv_completed_requests, 1024)
        self.assertEqual(result.controller_read_completed_requests, 2048)
        self.assertEqual(result.pim_completed_requests, 66 * self.cycle.total_pseudo_channels)
        self.assertGreater(result.gpu_injection_rejected_attempts, 0)
        self.assertGreater(result.kv_injection_rejected_attempts, 0)
        self.assertGreater(result.gpu_accepted_to_column_issue_cycles, 0)
        self.assertGreater(result.kv_accepted_to_column_issue_cycles, 0)


if __name__ == "__main__":
    unittest.main()
