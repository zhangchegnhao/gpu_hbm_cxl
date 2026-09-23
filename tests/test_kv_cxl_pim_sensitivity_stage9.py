from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from sieve_replay.ramulator.cxl_kv_workload import CXLKVWorkloadResult
from sieve_replay.ramulator.microbenchmark import MicrobenchmarkResult


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


STAGE9 = _load("run_kv_cxl_pim_sensitivity_stage9", ROOT / "scripts/run_kv_cxl_pim_sensitivity_stage9.py")
VALIDATE = _load("validate_kv_cxl_pim_sensitivity_stage9", ROOT / "scripts/validate_kv_cxl_pim_sensitivity_stage9.py")


def _fake_transfer(**shape: int) -> CXLKVWorkloadResult:
    count = shape["cxl_kv_read_transactions"]
    return CXLKVWorkloadResult(
        tick_ps=312, local_channels=128, cxl_channels=4, **shape,
        gpu_injected_requests=0, gpu_completed_requests=0,
        local_kv_injected_requests=0, local_kv_completed_requests=0,
        cxl_kv_injected_requests=count, cxl_kv_completed_requests=count,
        pim_injected_requests=0, pim_completed_requests=0,
        gpu_completion_cycles=0, local_kv_completion_cycles=0,
        cxl_kv_completion_cycles=max(1, count), pim_completion_cycles=0,
        total_completion_cycles=max(1, count), local_read_latency_cycles=0,
        cxl_read_latency_cycles=count, cxl_queue_wait_cycles=0,
        cxl_max_queue_wait_cycles=0, cxl_request_residence_cycles=count,
        cxl_max_request_residence_cycles=1, cxl_link_busy_cycles=count,
        cxl_controller_completion_cycles=max(1, count),
        gpu_injection_rejected_attempts=0, local_kv_injection_rejected_attempts=0,
        cxl_kv_injection_rejected_attempts=0,
    )


class Stage9Tests(unittest.TestCase):
    def test_factorial_plan_deduplicates_to_expected_components(self) -> None:
        stage8 = STAGE9._load_stage8()
        configuration, _ = STAGE9.stage5._source_configuration()
        cycle = STAGE9.SieveCycleV1Config.load(STAGE9.stage5.CYCLE)
        designs = STAGE9._designs(
            configuration.model,
            cycle,
            int(stage8["derived"]["memory_only_cxl_link_bytes_per_layer"]),
        )
        planned, _ = STAGE9._component_plan(designs)
        self.assertEqual(len(designs), 18)
        self.assertEqual(len(planned), 23)

    def test_fake_matrix_and_validator(self) -> None:
        def transfer(_root, _cycle, **kwargs):
            shape = {key: int(kwargs[key]) for key in (
                "gpu_read_transactions", "local_kv_read_transactions",
                "cxl_kv_read_transactions", "pim_gwrite_waves",
                "pim_mac_waves", "pim_read_waves",
            )}
            return _fake_transfer(**shape)

        def pim(_root, cycle, topology, operation, waves):
            completed = waves * topology.total_pseudo_channels
            return MicrobenchmarkResult(
                operation=operation, max_waves=waves, tick_ps=cycle.expected_tick_ps,
                duration_us_by_waves={waves: waves * cycle.pim_mac_interval_ps / 1_000_000.0 if operation == "PIM_MAC" else waves * cycle.pim_io_interval_ps / 1_000_000.0},
                cycles_by_waves={waves: waves}, injected_requests=completed,
                completed_requests=completed,
            )

        with tempfile.TemporaryDirectory() as temporary:
            result = STAGE9.run_stage9(
                temporary, transfer_runner=transfer, pim_runner=pim
            )
            validation = VALIDATE.validate(temporary)
            self.assertEqual(result["metadata"]["design_points"], 18)
            self.assertEqual(result["metadata"]["rows"], 162)
            self.assertEqual(result["metadata"]["exact_components"], 23)
            self.assertEqual(validation["status"], "passed")
            self.assertNotIn(b"\r\n", (Path(temporary) / "sensitivity.csv").read_bytes())

            manifest_path = Path(temporary) / "stage_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            record = next(
                item
                for item in manifest["cache_records"]
                if item["source"] == "stage9-exact-cache"
            )
            cache_path = Path(record["cache_path"])
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            cached["cache_input"]["simulation_identity"]["waves"] += 1
            cache_path.write_text(
                json.dumps(cached, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            record["sha256"] = STAGE9.stage5._sha256(cache_path)
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "cache identity mismatch"):
                VALIDATE.validate(temporary)


if __name__ == "__main__":
    unittest.main()
