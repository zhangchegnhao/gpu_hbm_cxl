from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from sieve_replay.config import ModelConfig
from sieve_replay.ramulator.cxl_kv_workload import CXLKVWorkloadResult
from sieve_replay.ramulator.cxl_pim_attention import (
    CXLPIMTopology,
    build_cxl_pim_attention_shape,
    run_cxl_pim_operation,
)
from sieve_replay.ramulator.microbenchmark import MicrobenchmarkResult
from sieve_replay.ramulator.mixed_workload import SieveCycleV1Config


ROOT = Path(__file__).resolve().parents[1]
RAMULATOR = ROOT / "third_party/ramulator2_cxl"
BINDING = RAMULATOR / "python/ramulator"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


STAGE8 = _load("run_kv_cxl_pim_attention_stage8", ROOT / "scripts/run_kv_cxl_pim_attention_stage8.py")
VALIDATE = _load("validate_kv_cxl_pim_attention_stage8", ROOT / "scripts/validate_kv_cxl_pim_attention_stage8.py")


def _fake_transfer(**shape: int) -> CXLKVWorkloadResult:
    count = shape["cxl_kv_read_transactions"]
    return CXLKVWorkloadResult(
        tick_ps=312,
        local_channels=128,
        cxl_channels=4,
        **shape,
        gpu_injected_requests=0,
        gpu_completed_requests=0,
        local_kv_injected_requests=0,
        local_kv_completed_requests=0,
        cxl_kv_injected_requests=count,
        cxl_kv_completed_requests=count,
        pim_injected_requests=0,
        pim_completed_requests=0,
        gpu_completion_cycles=0,
        local_kv_completion_cycles=0,
        cxl_kv_completion_cycles=max(1, count),
        pim_completion_cycles=0,
        total_completion_cycles=max(1, count),
        local_read_latency_cycles=0,
        cxl_read_latency_cycles=count,
        cxl_queue_wait_cycles=0,
        cxl_max_queue_wait_cycles=0,
        cxl_request_residence_cycles=count,
        cxl_max_request_residence_cycles=1,
        cxl_link_busy_cycles=count,
        cxl_controller_completion_cycles=max(1, count),
        gpu_injection_rejected_attempts=0,
        local_kv_injection_rejected_attempts=0,
        cxl_kv_injection_rejected_attempts=0,
    )


class Stage8Tests(unittest.TestCase):
    def test_shape_records_query_partial_and_compute_assumptions(self) -> None:
        model = ModelConfig.from_dict(
            json.loads((ROOT / "configs/models/qwen3_30b_a3b.json").read_text())
        )
        cycle = SieveCycleV1Config.load(ROOT / "configs/ramulator/sieve_hbm3e_cycle_v1.json")
        shape = build_cxl_pim_attention_shape(
            model,
            cycle,
            batch_size=8,
            spilled_kv_bytes_per_layer=142_390_528,
            topology=CXLPIMTopology(),
        )
        self.assertEqual(shape.query_bytes, 65_536)
        self.assertEqual(shape.query_link_transactions, 2_048)
        self.assertEqual(shape.partial_result_bytes, 1_064_960)
        self.assertEqual(shape.result_link_transactions, 33_280)
        self.assertGreater(shape.pim_mac_waves, shape.pim_read_waves)

    def test_fake_exact_matrix_and_validator(self) -> None:
        def transfer(_root, _cycle, **kwargs):
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
            return _fake_transfer(**shape)

        def pim(_root, cycle, topology, operation, waves):
            completed = waves * topology.total_pseudo_channels
            return MicrobenchmarkResult(
                operation=operation,
                max_waves=waves,
                tick_ps=cycle.expected_tick_ps,
                duration_us_by_waves={waves: waves * 0.001},
                cycles_by_waves={waves: waves},
                injected_requests=completed,
                completed_requests=completed,
            )

        with tempfile.TemporaryDirectory() as temporary:
            result = STAGE8.run_stage8(
                temporary, transfer_runner=transfer, pim_runner=pim
            )
            validation = VALIDATE.validate(temporary)
            self.assertEqual(result["metadata"]["exact_runs"], 5)
            self.assertEqual(len(result["rows"]), 4)
            self.assertEqual(validation["status"], "passed")
            self.assertNotIn(b"\r\n", (Path(temporary) / "comparison.csv").read_bytes())


@unittest.skipUnless(
    any(BINDING.glob("_ramulator*.so")),
    "CXL Ramulator binding is not built",
)
class Stage8RamulatorTests(unittest.TestCase):
    def test_one_cxl_pim_wave_completes_on_every_endpoint(self) -> None:
        cycle = SieveCycleV1Config.load(
            ROOT / "configs/ramulator/sieve_hbm3e_cycle_v1.json"
        )
        topology = CXLPIMTopology()
        result = run_cxl_pim_operation(
            RAMULATOR, cycle, topology, "PIM_MAC", 1
        )
        self.assertEqual(result.injected_requests, topology.total_pseudo_channels)
        self.assertEqual(result.completed_requests, topology.total_pseudo_channels)


if __name__ == "__main__":
    unittest.main()
