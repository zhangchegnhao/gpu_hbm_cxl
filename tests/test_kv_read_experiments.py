from __future__ import annotations

import unittest
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from scripts.run_kv_read_experiments import kv_transactions, run_exact, validate_result
from scripts.plot_kv_read_experiments import _validate_matrix
from sieve_replay.config import load_configuration
from sieve_replay.ramulator.kv_read_workload import KVReadWorkloadResult, KVWorkloadShape
from sieve_replay.ramulator.mixed_workload import SieveCycleV1Config


ROOT = Path(__file__).resolve().parents[1]


class KvReadExperimentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_configuration(
            ROOT / "configs/experiments/full_decode_real_kv_b8_c4k.json"
        ).model
        cls.cycle = SieveCycleV1Config.load(
            ROOT / "configs/ramulator/sieve_hbm3e_cycle_v1.json"
        )

    def test_kv_transactions_use_k_and_v_per_layer(self) -> None:
        byte_count, transactions = kv_transactions((4096,) * 8, self.model, self.cycle)
        self.assertEqual(byte_count, 67_108_864)
        self.assertEqual(transactions, 2_097_152)

    def test_kv_transactions_round_up_to_hbm_transaction(self) -> None:
        byte_count, transactions = kv_transactions((1,), self.model, self.cycle)
        self.assertEqual(byte_count, 2_048)
        self.assertEqual(transactions, 64)

    def test_context_lengths_are_required_and_positive(self) -> None:
        with self.assertRaises(ValueError):
            kv_transactions((), self.model, self.cycle)
        with self.assertRaises(ValueError):
            kv_transactions((0,), self.model, self.cycle)

    def valid_result(self) -> KVReadWorkloadResult:
        fields = {name: 0 for name in KVReadWorkloadResult.__dataclass_fields__}
        fields.update({
            "tick_ps": self.cycle.expected_tick_ps,
            "gpu_read_transactions": 2,
            "kv_read_transactions": 3,
            "gpu_injected_requests": 2,
            "gpu_completed_requests": 2,
            "kv_injected_requests": 3,
            "kv_completed_requests": 3,
            "gpu_completion_cycles": 5,
            "kv_completion_cycles": 6,
            "total_completion_cycles": 6,
            "controller_read_completed_requests": 5,
            "read_latency_cycles": 2,
            "gpu_request_residence_cycles": 6,
            "kv_request_residence_cycles": 9,
        })
        return KVReadWorkloadResult(**fields)

    def test_exact_result_rejects_incomplete_kv_stream(self) -> None:
        shape = KVWorkloadShape(2, 0, 0, 0, 3)
        result = self.valid_result()
        validate_result(shape, result, self.cycle)
        with self.assertRaisesRegex(ValueError, "kv request counts"):
            validate_result(shape, replace(result, kv_completed_requests=2), self.cycle)

    def test_exact_result_rejects_wrong_controller_count_and_total(self) -> None:
        shape = KVWorkloadShape(2, 0, 0, 0, 3)
        result = self.valid_result()
        with self.assertRaisesRegex(ValueError, "controller"):
            validate_result(shape, replace(result, controller_read_completed_requests=4), self.cycle)
        with self.assertRaisesRegex(ValueError, "total completion"):
            validate_result(shape, replace(result, total_completion_cycles=5), self.cycle)

    def test_stale_plan_is_rejected_before_simulation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "plan.json"
            path.write_text(json.dumps({"method": "kv-read-representative-microbenchmark-v1", "planner_sha256": "stale"}))
            with patch("scripts.run_kv_read_experiments.CACHE_ROOT", root / "cache"), patch("scripts.run_kv_read_experiments.freeze_or_verify"), patch("scripts.run_kv_read_experiments._run_shape") as run:
                with self.assertRaisesRegex(ValueError, "stale"):
                    run_exact(path, root)
                run.assert_not_called()

    def test_plot_matrix_rejects_duplicate_or_missing_case(self) -> None:
        rows = [{"case": case, "placement": placement} for case in ("b8_c4k", "b8_c8k", "b8_c16k", "b16_c8k") for placement in ("gpu-only", "frozen-oracle", "fixed-half-prefix")]
        _validate_matrix(rows)
        with self.assertRaises(ValueError):
            _validate_matrix(rows[:-1] + [rows[0]])


if __name__ == "__main__":
    unittest.main()
