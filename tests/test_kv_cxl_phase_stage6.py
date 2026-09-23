from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from sieve_replay.ramulator.cxl_kv_workload import CXLKVWorkloadResult


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_kv_cxl_phase_stage6.py"
SPEC = importlib.util.spec_from_file_location("run_kv_cxl_phase_stage6", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
STAGE6 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STAGE6)


def _fake_result(**shape: int) -> CXLKVWorkloadResult:
    pim_requests = (
        shape["pim_gwrite_waves"]
        + shape["pim_mac_waves"]
        + shape["pim_read_waves"]
    ) * 256
    return CXLKVWorkloadResult(
        tick_ps=312,
        local_channels=8,
        cxl_channels=4,
        gpu_read_transactions=shape["gpu_read_transactions"],
        local_kv_read_transactions=shape["local_kv_read_transactions"],
        cxl_kv_read_transactions=shape["cxl_kv_read_transactions"],
        pim_gwrite_waves=shape["pim_gwrite_waves"],
        pim_mac_waves=shape["pim_mac_waves"],
        pim_read_waves=shape["pim_read_waves"],
        gpu_injected_requests=shape["gpu_read_transactions"],
        gpu_completed_requests=shape["gpu_read_transactions"],
        local_kv_injected_requests=shape["local_kv_read_transactions"],
        local_kv_completed_requests=shape["local_kv_read_transactions"],
        cxl_kv_injected_requests=shape["cxl_kv_read_transactions"],
        cxl_kv_completed_requests=shape["cxl_kv_read_transactions"],
        pim_injected_requests=pim_requests,
        pim_completed_requests=pim_requests,
        gpu_completion_cycles=100,
        local_kv_completion_cycles=120,
        cxl_kv_completion_cycles=160,
        pim_completion_cycles=80 if pim_requests else 0,
        total_completion_cycles=160,
        local_read_latency_cycles=120,
        cxl_read_latency_cycles=160,
        cxl_queue_wait_cycles=320,
        cxl_max_queue_wait_cycles=32,
        cxl_request_residence_cycles=480,
        cxl_max_request_residence_cycles=48,
        cxl_link_busy_cycles=400,
        cxl_controller_completion_cycles=160,
        gpu_injection_rejected_attempts=0,
        local_kv_injection_rejected_attempts=0,
        cxl_kv_injection_rejected_attempts=0,
    )


class KvCxlPhaseStage6Test(unittest.TestCase):
    def test_phase_shapes_separate_kv_and_expert_requests(self) -> None:
        configuration, source = STAGE6.stage5._source_configuration()
        template = STAGE6.stage5._template(source, 8, 32768)
        cycle = STAGE6.SieveCycleV1Config.load(STAGE6.stage5.CYCLE)
        capacity = STAGE6.stage5._capacity_rows()[("a800", 8, 32768, 64_000_000_000)]
        timing = STAGE6.AnalyticTimingModel(configuration.model, configuration.hardware)
        decision = STAGE6.create_policy("gpu-only", timing).place(template)
        shapes = STAGE6._phase_shapes(template, decision, capacity, configuration, cycle, 4096)
        self.assertGreater(shapes["attention"]["local_kv_read_transactions"], 0)
        self.assertGreater(shapes["attention"]["cxl_kv_read_transactions"], 0)
        self.assertEqual(shapes["attention"]["gpu_read_transactions"], 0)
        self.assertGreater(shapes["expert"]["gpu_read_transactions"], 0)
        self.assertEqual(shapes["expert"]["local_kv_read_transactions"], 0)
        self.assertEqual(shapes["expert"]["cxl_kv_read_transactions"], 0)

    def test_shape_identity_does_not_include_policy_label(self) -> None:
        context = {"model": "x"}
        shape = {"gpu_read_transactions": 1, "local_kv_read_transactions": 2}
        first = STAGE6._shape_identity("attention", shape, 1, context)
        second = STAGE6._shape_identity("attention", shape, 1, context)
        self.assertEqual(first, second)
        self.assertNotIn("policy", first)

    def test_smoke_run_deduplicates_two_policy_consumers(self) -> None:
        def fake_runner(_root: Path, _cycle: object, **shape: int) -> CXLKVWorkloadResult:
            return _fake_result(**shape)

        with tempfile.TemporaryDirectory() as temporary:
            result = STAGE6.run_stage6(
                temporary,
                request_scales=(4096,),
                policies=("gpu-only", "sieve"),
                workload_runner=fake_runner,
            )
            self.assertEqual(result["metadata"]["unique_shapes"], 2)
            self.assertEqual(result["metadata"]["exact_runs"], 2)
            self.assertEqual(result["metadata"]["new_runs"], 2)
            self.assertEqual(len(result["rows"]), 2)
            self.assertEqual(result["metadata"]["cache_records"][0]["consumers"], ["gpu-only", "sieve"])

    def test_resume_from_exact_cache_directory(self) -> None:
        def fake_runner(_root: Path, _cycle: object, **shape: int) -> CXLKVWorkloadResult:
            return _fake_result(**shape)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            output.mkdir(exist_ok=True)
            (output / "exact_cache").mkdir()
            result = STAGE6.run_stage6(
                output,
                request_scales=(4096,),
                policies=("gpu-only", "sieve"),
                workload_runner=fake_runner,
            )
            self.assertEqual(result["metadata"]["new_runs"], 2)


if __name__ == "__main__":
    unittest.main()
