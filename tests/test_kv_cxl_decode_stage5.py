from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from sieve_replay.config import load_configuration
from sieve_replay.ramulator.cxl_kv_workload import CXLKVWorkloadResult
from sieve_replay.simulation import EventEngine
from sieve_replay.simulation.layer_graph import build_layer_graph
from sieve_replay.timing import CXLReadConfig


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_kv_cxl_decode_stage5.py"
SPEC = importlib.util.spec_from_file_location("run_kv_cxl_decode_stage5", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
STAGE5 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STAGE5)


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


class KvCxlDecodeStage5Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.configuration, cls.source = STAGE5._source_configuration()

    def test_template_tiles_routes_with_new_request_ids(self) -> None:
        trace = STAGE5._template(self.source, 16, 32768)
        self.assertEqual(trace.batch_size, 16)
        self.assertEqual(trace.context_lengths, (32768,) * 16)
        self.assertEqual(len({record.token_id for record in trace.records}), 16)
        self.assertEqual(trace.records[0].expert_ids, trace.records[8].expert_ids)

    def test_request_timing_adapter_injects_milestones(self) -> None:
        trace = STAGE5._template(self.source, 8, 32768)
        cycle = STAGE5.SieveCycleV1Config.load(STAGE5.CYCLE)
        timing_config = CXLReadConfig(
            mode="memory-only-v1",
            local_capacity_bytes=80_000_000_000,
            capacity_bytes=64_000_000_000,
            bandwidth_bytes_per_second=64e9,
            latency_us=0.25,
        )
        shape = {
            "gpu_read_transactions": 1,
            "local_kv_read_transactions": 2,
            "cxl_kv_read_transactions": 3,
            "pim_gwrite_waves": 0,
            "pim_mac_waves": 0,
            "pim_read_waves": 0,
            "request_scale": 4096,
        }
        result = _fake_result(**{key: value for key, value in shape.items() if key != "request_scale"})
        timing = STAGE5.RequestLevelTiming(
            self.configuration.model,
            self.configuration.hardware,
            timing_config,
            result,
            shape,
            cycle.transaction_bytes,
        )
        decision = STAGE5.create_policy("gpu-only", timing).place(trace)
        events = build_layer_graph(trace, decision, timing, "overlap-local-hbm-v1")
        by_name = {event.name: event for event in events}
        self.assertEqual(by_name["attention"].dependencies, ("rope",))
        self.assertEqual(by_name["o_proj"].dependencies, ("attention", "attention_kv_read", "cxl_kv_read"))
        self.assertAlmostEqual(by_name["cxl_kv_read"].duration_us, result.cxl_kv_completion_cycles * result.tick_ps / 1_000_000.0)
        self.assertGreater(EventEngine().run(events)[-1].end_us, 0.0)

    def test_smoke_run_writes_48_layer_graph(self) -> None:
        def fake_runner(_root: Path, _cycle: object, **shape: int) -> CXLKVWorkloadResult:
            return _fake_result(**shape)

        with tempfile.TemporaryDirectory() as temporary:
            result = STAGE5.run_stage5(
                temporary,
                cases=("b8_c32k",),
                policies=("gpu-only",),
                workload_runner=fake_runner,
            )
            self.assertEqual(result["stage_manifest"]["rows"], 1)
            self.assertEqual(result["stage_manifest"]["layer_rows"], 48)
            self.assertTrue((Path(temporary) / "layer_results.csv").is_file())
            self.assertTrue((Path(temporary) / "exact_cache").is_dir())


if __name__ == "__main__":
    unittest.main()
