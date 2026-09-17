from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from scripts.fill_kv_workloads import read_exact
from sieve_replay.ramulator.mixed_workload import (
    MixedWorkloadResult,
    SieveCycleV1Config,
)
from sieve_replay.ramulator.workload_cache import WorkloadShape


ROOT = Path(__file__).resolve().parents[1]
CYCLE_PATH = ROOT / "configs/ramulator/sieve_hbm3e_cycle_v1.json"


class KvWorkloadFillValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cycle = SieveCycleV1Config.load(CYCLE_PATH)
        self.context = {
            "cycle_config_sha256": "a" * 64,
            "extension_sha256": "b" * 64,
            "mixed_workload_sha256": "c" * 64,
        }
        self.shape = WorkloadShape(2, 1, 1, 1)

    def valid_result(self) -> MixedWorkloadResult:
        pim_requests = 3 * self.cycle.total_pseudo_channels
        return MixedWorkloadResult(
            tick_ps=self.cycle.expected_tick_ps,
            gpu_read_transactions=self.shape.gpu_read_transactions,
            pim_gwrite_waves=self.shape.pim_gwrite_waves,
            pim_mac_waves=self.shape.pim_mac_waves,
            pim_read_waves=self.shape.pim_read_waves,
            gpu_completion_cycles=8,
            pim_completion_cycles=7,
            pim_gwrite_completion_cycles=3,
            pim_mac_completion_cycles=5,
            pim_read_completion_cycles=7,
            total_completion_cycles=8,
            gpu_injected_requests=self.shape.gpu_read_transactions,
            gpu_completed_requests=self.shape.gpu_read_transactions,
            pim_injected_requests=pim_requests,
            pim_completed_requests=pim_requests,
            gpu_blocked_by_pim_cycles=0,
            pim_blocked_by_gpu_cycles=0,
            pim_queue_wait_cycles=0,
            gpu_column_issues=0,
            pim_column_issues=0,
            pim_row_activations=0,
            pim_row_conflicts=0,
        )

    def write_cache(self, payload: dict) -> Path:
        temporary = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        with temporary:
            json.dump(payload, temporary)
        return Path(temporary.name)

    def exact_payload(self, result: MixedWorkloadResult | None = None) -> dict:
        return {
            "cache_input": {"context": self.context, "shape": asdict(self.shape)},
            "result": asdict(result or self.valid_result()),
        }

    def test_accepts_exact_context_and_shape(self) -> None:
        path = self.write_cache(self.exact_payload())
        try:
            self.assertEqual(read_exact(path, self.shape, self.context, self.cycle), self.valid_result())
        finally:
            path.unlink()

    def test_rejects_context_with_missing_or_extra_provenance(self) -> None:
        for altered_context in (
            {"cycle_config_sha256": "a" * 64, "extension_sha256": "b" * 64},
            {**self.context, "trace_sha256": "d" * 64},
        ):
            payload = self.exact_payload()
            payload["cache_input"]["context"] = altered_context
            path = self.write_cache(payload)
            try:
                self.assertIsNone(read_exact(path, self.shape, self.context, self.cycle))
            finally:
                path.unlink()

    def test_rejects_result_for_different_shape(self) -> None:
        result = self.valid_result()
        payload = self.exact_payload(result)
        payload["result"]["gpu_read_transactions"] += 1
        path = self.write_cache(payload)
        try:
            with self.assertRaisesRegex(ValueError, "does not describe"):
                read_exact(path, self.shape, self.context, self.cycle)
        finally:
            path.unlink()

    def test_rejects_incomplete_request_counters(self) -> None:
        result = self.valid_result()
        payload = self.exact_payload(result)
        payload["result"]["pim_completed_requests"] -= 1
        path = self.write_cache(payload)
        try:
            with self.assertRaisesRegex(ValueError, "did not complete"):
                read_exact(path, self.shape, self.context, self.cycle)
        finally:
            path.unlink()


if __name__ == "__main__":
    unittest.main()
