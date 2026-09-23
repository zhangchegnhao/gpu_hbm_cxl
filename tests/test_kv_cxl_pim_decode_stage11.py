from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


STAGE11 = _load(
    "run_kv_cxl_pim_decode_stage11",
    ROOT / "scripts/run_kv_cxl_pim_decode_stage11.py",
)
VALIDATE = _load(
    "validate_kv_cxl_pim_decode_stage11",
    ROOT / "scripts/validate_kv_cxl_pim_decode_stage11.py",
)


class Stage11Tests(unittest.TestCase):
    def _graph(self, schedule_mode: str):
        stage6 = STAGE11._load_stage6()
        stage7 = STAGE11._load_stage7()
        configuration, source = STAGE11.stage5._source_configuration()
        template = STAGE11.stage5._template(
            source, STAGE11.BATCH_SIZE, STAGE11.CONTEXT_LENGTH
        )
        decision = STAGE11._decision(stage6["row"])
        cycle = STAGE11.stage6.SieveCycleV1Config.load(STAGE11.stage5.CYCLE)
        resident = stage7["rows"]["resident-local-counterfactual"]
        timing = STAGE11.Stage11Timing(
            configuration.model,
            configuration.hardware,
            float(resident["estimated_attention_ms"]) * 1000.0 / STAGE11.LAYERS,
            stage6["expert_result"],
            stage6["row"]["expert_shape"],
            cycle.transaction_bytes,
        )
        events, _ = STAGE11._build_layer_events(
            STAGE11.stage6._layer_trace(template, 0),
            decision,
            timing,
            None,
            external_kind="cxl-pim",
            schedule_mode=schedule_mode,
            external_us=100.0,
            non_attention_residual_us=0.0,
        )
        return {event.name: event for event in events}

    def test_attention_branch_dependencies_match_bounds(self) -> None:
        ideal = self._graph("ideal-overlap-bound")
        serial = self._graph("fully-serialized-bound")
        prefix = "step0.layer0."
        self.assertEqual(
            ideal[prefix + "cxl_pim_pipeline"].dependencies,
            (prefix + "rope",),
        )
        self.assertEqual(
            serial[prefix + "cxl_pim_pipeline"].dependencies,
            (prefix + "attention",),
        )
        self.assertEqual(
            set(ideal[prefix + "attention_join"].dependencies),
            {prefix + "attention", prefix + "cxl_pim_pipeline"},
        )
        self.assertEqual(
            ideal[prefix + "o_proj"].dependencies,
            (prefix + "attention_join",),
        )

    def test_full_replay_and_validator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = STAGE11.run_stage11(temporary)
            validation = VALIDATE.validate(temporary)
            self.assertEqual(result["metadata"]["new_ramulator_runs"], 0)
            self.assertEqual(result["metadata"]["reused_exact_cache_artifacts"], 6)
            self.assertEqual(len(result["rows"]), 8)
            self.assertEqual(validation["status"], "passed")
            memory = next(
                row for row in result["rows"] if row["scenario"] == "memory-only-cxl"
            )
            serial = {
                row["design_id"]: row
                for row in result["rows"]
                if row["scenario"] == "cxl-pim-attention"
                and row["schedule_mode"] == "fully-serialized-bound"
            }
            self.assertGreater(
                serial["c2_mac49152_p4"]["total_latency_us"],
                memory["total_latency_us"],
            )
            self.assertLess(
                serial["c4_mac24576_p4"]["total_latency_us"],
                memory["total_latency_us"],
            )
            self.assertLess(
                serial["c8_mac24576_p4"]["total_latency_us"],
                memory["total_latency_us"],
            )
            self.assertNotIn(
                b"\r\n", (Path(temporary) / "comparison.csv").read_bytes()
            )


if __name__ == "__main__":
    unittest.main()
