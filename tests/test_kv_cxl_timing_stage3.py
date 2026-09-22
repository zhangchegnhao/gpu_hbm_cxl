from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from sieve_replay.config import load_configuration
from sieve_replay.policy import create_policy
from sieve_replay.simulation import EventEngine
from sieve_replay.simulation.layer_graph import build_layer_graph
from sieve_replay.timing import AnalyticTimingModel, CXLReadConfig
from sieve_replay.trace import load_trace_set


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_kv_cxl_timing_stage3.py"
SPEC = importlib.util.spec_from_file_location("run_kv_cxl_timing_stage3", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
STAGE3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STAGE3)


class KvCxlTimingStage3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        configuration = load_configuration(ROOT / "configs/experiments/full_decode_real_pilot_cycle_v0.json")
        traces = load_trace_set(
            configuration.experiment.trace_path,
            configuration.model,
            configuration.experiment.layers,
            configuration.experiment.steps,
        )
        cls.configuration = configuration
        cls.template = STAGE3._template(traces.batches[0], 8, 32768)

    def test_cxl_config_rejects_positive_capacity_without_bandwidth(self) -> None:
        with self.assertRaises(ValueError):
            CXLReadConfig.from_raw(
                {
                    "cxl_mode": "memory-only-v1",
                    "cxl_local_capacity_gb": 80,
                    "cxl_capacity_gb": 16,
                    "cxl_bandwidth_gb_s": 0,
                },
                self.configuration.hardware,
            )

    def test_serial_cxl_read_joins_attention_dependencies(self) -> None:
        cxl = CXLReadConfig(
            mode="memory-only-v1",
            local_capacity_bytes=80_000_000_000,
            capacity_bytes=16_000_000_000,
            bandwidth_bytes_per_second=64_000_000_000,
            latency_us=0.25,
        )
        timing = AnalyticTimingModel(self.configuration.model, self.configuration.hardware, cxl_config=cxl)
        decision = create_policy("gpu-only", timing).place(self.template)
        events = build_layer_graph(self.template, decision, timing, "serial-local-hbm-v1")
        by_name = {event.name: event for event in events}
        self.assertEqual(len(events), 20)
        self.assertEqual(by_name["attention"].dependencies, ("attention_kv_read", "cxl_kv_read"))
        self.assertGreater(by_name["cxl_kv_read"].duration_us, 0.0)
        self.assertEqual(by_name["cxl_kv_read"].resources, ("CXL_MEM_PATH",))
        self.assertEqual(by_name["attention_kv_read"].resources, ("HBM_MEM_PATH",))

    def test_overlap_cxl_read_joins_at_output_projection(self) -> None:
        cxl = CXLReadConfig(
            mode="memory-only-v1",
            local_capacity_bytes=80_000_000_000,
            capacity_bytes=16_000_000_000,
            bandwidth_bytes_per_second=64_000_000_000,
            latency_us=0.25,
        )
        timing = AnalyticTimingModel(self.configuration.model, self.configuration.hardware, cxl_config=cxl)
        decision = create_policy("gpu-only", timing).place(self.template)
        events = build_layer_graph(self.template, decision, timing, "overlap-local-hbm-v1")
        by_name = {event.name: event for event in events}
        self.assertEqual(by_name["attention"].dependencies, ("rope",))
        self.assertEqual(
            by_name["o_proj"].dependencies,
            ("attention", "attention_kv_read", "cxl_kv_read"),
        )
        scheduled = EventEngine().run(events)
        self.assertLess(
            next(item.end_us for item in scheduled if item.event.name == "attention"),
            next(item.end_us for item in scheduled if item.event.name == "o_proj"),
        )

    def test_stage3_smoke_writes_audited_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = STAGE3.run_stage3(
                ROOT / "configs/experiments/full_decode_real_pilot_cycle_v0.json",
                temporary,
                pressure_matrix=(("b8_c32k", 8, 32768),),
                domains=(("simulated", 96.0),),
                budgets=(16,),
                profiles={"nominal-assumption": STAGE3.CXL_PROFILES["nominal-assumption"]},
                policies=("gpu-only",),
                modes=("cxl-memory-only-serial", "cxl-memory-only-overlap"),
            )
            self.assertEqual(result["stage_manifest"]["rows"], 2)
            self.assertEqual(result["stage_manifest"]["oom_rows"], 0)
            self.assertTrue((Path(temporary) / "comparison.csv").is_file())
            self.assertTrue((Path(temporary) / "stage_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
