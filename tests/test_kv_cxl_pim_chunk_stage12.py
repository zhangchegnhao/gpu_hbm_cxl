from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


STAGE12 = _load_script(
    "run_kv_cxl_pim_chunk_stage12_test",
    ROOT / "scripts/run_kv_cxl_pim_chunk_stage12.py",
)
VALIDATE = _load_script(
    "validate_kv_cxl_pim_chunk_stage12_test",
    ROOT / "scripts/validate_kv_cxl_pim_chunk_stage12.py",
)


class KVCXLPIMChunkStage12Test(unittest.TestCase):
    def test_merge_model_is_memory_bound_and_includes_kernel_overhead(self) -> None:
        configuration, _ = STAGE12.stage5._source_configuration()
        cycle = STAGE12.stage6.SieveCycleV1Config.load(STAGE12.stage5.CYCLE)
        design = STAGE12._selected_designs(configuration, cycle)["c4_mac24576_p4"]
        merge = STAGE12._merge_model(configuration, design)
        self.assertEqual(
            merge["partial_states_per_chunk"],
            STAGE12.BATCH_SIZE
            * configuration.model.num_attention_heads
            * design["topology"].total_pseudo_channels,
        )
        self.assertGreater(merge["memory_us_per_chunk"], merge["compute_us_per_chunk"])
        self.assertAlmostEqual(
            merge["duration_us_per_chunk"],
            merge["memory_us_per_chunk"]
            + configuration.hardware.gpu_kernel_overhead_us,
        )

    def test_chunk_events_expose_readiness_and_share_gpu_compute(self) -> None:
        configuration, source = STAGE12.stage5._source_configuration()
        stage6_source = STAGE12.stage11._load_stage6()
        cycle = STAGE12.stage6.SieveCycleV1Config.load(STAGE12.stage5.CYCLE)
        template = STAGE12.stage5._template(
            source, STAGE12.BATCH_SIZE, STAGE12.CONTEXT_LENGTH
        )
        decision = STAGE12.stage11._decision(stage6_source["row"])
        stage10 = json.loads(
            STAGE12.stage11.STAGE10_COMPARISON.read_text(encoding="utf-8")
        )
        local_us = (
            float(stage10["metadata"]["baselines"]["local_gpu_attention_ms"])
            * 1000.0
            / STAGE12.LAYERS
        )
        timing = STAGE12.stage11.Stage11Timing(
            configuration.model,
            configuration.hardware,
            local_us,
            stage6_source["expert_result"],
            stage6_source["row"]["expert_shape"],
            cycle.transaction_bytes,
        )
        events, _ = STAGE12._build_chunk_layer_events(
            template,
            decision,
            timing,
            None,
            partial_ready_us=(100.0, 200.0),
            merge_us_per_chunk=3.0,
            non_attention_residual_us=0.0,
        )
        by_name = {event.name: event for event in events}
        prefix = "step0.layer0."
        self.assertEqual(
            by_name[prefix + "cxl_partial_ready_1"].dependencies,
            (prefix + "cxl_partial_ready_0",),
        )
        self.assertEqual(
            by_name[prefix + "gpu_partial_merge_1"].resources,
            ("GPU_COMPUTE",),
        )
        self.assertEqual(
            set(by_name[prefix + "attention_join"].dependencies),
            {prefix + "attention", prefix + "gpu_partial_merge_1"},
        )
        scheduled = STAGE12.EventEngine().run(events)
        scheduled_by_name = {item.event.name: item for item in scheduled}
        self.assertGreaterEqual(
            scheduled_by_name[prefix + "gpu_partial_merge_0"].start_us,
            scheduled_by_name[prefix + "attention"].end_us,
        )
        self.assertAlmostEqual(
            scheduled_by_name[prefix + "cxl_partial_ready_1"].end_us
            - scheduled_by_name[prefix + "rope"].end_us,
            200.0,
        )


@unittest.skipUnless(
    (STAGE12.DEFAULT_OUTPUT / "stage_manifest.json").is_file(),
    "Stage-12 formal results are not generated",
)
class KVCXLPIMChunkStage12ResultTest(unittest.TestCase):
    def test_formal_result_passes_independent_validator(self) -> None:
        validation = VALIDATE.validate(STAGE12.DEFAULT_OUTPUT)
        self.assertEqual(validation["status"], "passed")
        self.assertEqual(validation["checks"]["exact_runs"], 9)
        self.assertEqual(validation["checks"]["layer_rows"], 432)


if __name__ == "__main__":
    unittest.main()
