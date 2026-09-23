from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from sieve_replay.ramulator.cxl_kv_workload import CXLKVWorkloadResult


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


STAGE7 = _load("run_kv_cxl_a800_bridge_stage7", ROOT / "scripts/run_kv_cxl_a800_bridge_stage7.py")
VALIDATE = _load("validate_kv_cxl_a800_bridge_stage7", ROOT / "scripts/validate_kv_cxl_a800_bridge_stage7.py")


def _fake_result(**shape: int) -> CXLKVWorkloadResult:
    local = shape["local_kv_read_transactions"]
    cxl = shape["cxl_kv_read_transactions"]
    return CXLKVWorkloadResult(
        tick_ps=312,
        local_channels=128,
        cxl_channels=4,
        **shape,
        gpu_injected_requests=shape["gpu_read_transactions"],
        gpu_completed_requests=shape["gpu_read_transactions"],
        local_kv_injected_requests=local,
        local_kv_completed_requests=local,
        cxl_kv_injected_requests=cxl,
        cxl_kv_completed_requests=cxl,
        pim_injected_requests=0,
        pim_completed_requests=0,
        gpu_completion_cycles=0,
        local_kv_completion_cycles=600_000,
        cxl_kv_completion_cycles=0,
        pim_completion_cycles=0,
        total_completion_cycles=600_000,
        local_read_latency_cycles=local,
        cxl_read_latency_cycles=0,
        cxl_queue_wait_cycles=0,
        cxl_max_queue_wait_cycles=0,
        cxl_request_residence_cycles=0,
        cxl_max_request_residence_cycles=0,
        cxl_link_busy_cycles=0,
        cxl_controller_completion_cycles=0,
        gpu_injection_rejected_attempts=0,
        local_kv_injection_rejected_attempts=0,
        cxl_kv_injection_rejected_attempts=0,
    )


class Stage7Tests(unittest.TestCase):
    def test_a800_forward_holdout_is_bounded(self) -> None:
        calibration = STAGE7._load_a800_calibration()
        self.assertEqual(len(calibration["anchors"]), 3)
        self.assertLess(
            calibration["forward_holdout"]["absolute_percentage_error"], 0.05
        )
        self.assertGreater(calibration["fit"]["r_squared"], 0.99)

    def test_run_and_validator_bind_exact_resident_shape(self) -> None:
        def runner(_root, _cycle, **kwargs):
            shape = {
                key: int(kwargs[key])
                for key in (
                    "gpu_read_transactions",
                    "local_kv_read_transactions",
                    "cxl_kv_read_transactions",
                    "pim_gwrite_waves",
                    "pim_mac_waves",
                    "pim_read_waves",
                )
            }
            return _fake_result(**shape)

        with tempfile.TemporaryDirectory() as temporary:
            result = STAGE7.run_stage7(temporary, workload_runner=runner)
            validation = VALIDATE.validate(temporary)
            self.assertEqual(result["metadata"]["exact_new_runs"], 1)
            self.assertEqual(result["metadata"]["interpolation"], 0)
            self.assertEqual(validation["status"], "passed")
            self.assertNotIn(
                b"\r\n", (Path(temporary) / "comparison.csv").read_bytes()
            )
            self.assertEqual(
                result["rows"][0]["local_kv_read_transactions"], 16_777_216
            )
            self.assertEqual(result["rows"][0]["cxl_kv_read_transactions"], 0)


if __name__ == "__main__":
    unittest.main()
