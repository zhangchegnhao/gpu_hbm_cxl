from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from sieve_replay.ramulator.cxl_pim_pipeline import CXLPIMPipelineResult


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


STAGE10 = _load(
    "run_kv_cxl_pim_pipeline_stage10",
    ROOT / "scripts/run_kv_cxl_pim_pipeline_stage10.py",
)
VALIDATE = _load(
    "validate_kv_cxl_pim_pipeline_stage10",
    ROOT / "scripts/validate_kv_cxl_pim_pipeline_stage10.py",
)


def _fake_pipeline(_root, cycle, topology, **values) -> CXLPIMPipelineResult:
    endpoints = topology.total_pseudo_channels
    query = int(values["query_link_transactions"])
    gwrite = int(values["pim_gwrite_waves"]) * endpoints
    mac = int(values["pim_mac_waves"]) * endpoints
    read = int(values["pim_read_waves"]) * endpoints
    result = int(values["result_link_transactions"])
    query_completion = max(1, query)
    gwrite_start = query_completion
    gwrite_completion = gwrite_start + max(1, int(values["pim_gwrite_waves"]))
    mac_start = gwrite_completion
    mac_completion = mac_start + max(1, int(values["pim_mac_waves"]))
    read_start = mac_completion
    read_completion = read_start + max(1, int(values["pim_read_waves"]))
    result_start = read_completion
    result_completion = result_start + max(1, result)
    return CXLPIMPipelineResult(
        tick_ps=cycle.expected_tick_ps,
        pim_channels=topology.channels,
        link_channels=int(values["link_channels"]),
        pseudo_channels_per_channel=topology.pseudo_channels_per_channel,
        query_link_transactions=query,
        pim_gwrite_waves=int(values["pim_gwrite_waves"]),
        pim_mac_waves=int(values["pim_mac_waves"]),
        pim_read_waves=int(values["pim_read_waves"]),
        result_link_transactions=result,
        query_link_injected_requests=query,
        query_link_completed_requests=query,
        query_link_completion_cycles=query_completion,
        query_link_rejected_attempts=0,
        pim_gwrite_injected_requests=gwrite,
        pim_gwrite_completed_requests=gwrite,
        pim_gwrite_start_cycles=gwrite_start,
        pim_gwrite_completion_cycles=gwrite_completion,
        pim_mac_injected_requests=mac,
        pim_mac_completed_requests=mac,
        pim_mac_start_cycles=mac_start,
        pim_mac_completion_cycles=mac_completion,
        pim_read_injected_requests=read,
        pim_read_completed_requests=read,
        pim_read_start_cycles=read_start,
        pim_read_completion_cycles=read_completion,
        result_link_injected_requests=result,
        result_link_completed_requests=result,
        result_link_start_cycles=result_start,
        result_link_completion_cycles=result_completion,
        result_link_rejected_attempts=0,
        pipeline_completion_cycles=result_completion,
        controller_pim_gwrite_requests=gwrite,
        controller_pim_mac_requests=mac,
        controller_pim_read_requests=read,
        controller_link_requests=query + result,
        link_queue_wait_cycles=0,
        link_max_queue_wait_cycles=0,
        link_request_residence_cycles=query + result,
        link_max_request_residence_cycles=1,
        link_busy_cycles=query + result,
    )


def _fake_precision(**matrix):
    rows = []
    for context in matrix["context_lengths"]:
        for pseudo_channels in matrix["total_pseudo_channels"]:
            for logit_std in matrix["logit_stds"]:
                for seed in matrix["seeds"]:
                    for scalar_format in ("fp32", "bf16", "fp16"):
                        error = 0.0 if scalar_format == "fp32" else 1e-3
                        rows.append(
                            {
                                "context_length": context,
                                "total_pseudo_channels": pseudo_channels,
                                "seed": seed,
                                "head_dim": matrix["head_dim"],
                                "logit_std": logit_std,
                                "scalar_format": scalar_format,
                                "max_abs_error": error,
                                "mean_abs_error": error / 2,
                                "rmse": error / 2,
                                "relative_l2_error": error,
                                "cosine_similarity": 1.0 - error,
                                "vs_fp32_partial_max_abs_error": error,
                                "vs_fp32_partial_mean_abs_error": error / 2,
                                "vs_fp32_partial_rmse": error / 2,
                                "vs_fp32_partial_relative_l2_error": error,
                                "vs_fp32_partial_cosine_similarity": 1.0 - error,
                            }
                        )
    return rows


class Stage10Tests(unittest.TestCase):
    def test_boundary_designs_are_present_in_stage9(self) -> None:
        source = STAGE10._load_stage9()
        designs = {row["design_id"] for row in source["design_summary"]}
        self.assertTrue(set(STAGE10.DESIGN_IDS).issubset(designs))

    def test_fake_pipeline_precision_matrix_and_validator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = STAGE10.run_stage10(
                temporary,
                pipeline_runner=_fake_pipeline,
                precision_runner=_fake_precision,
            )
            validation = VALIDATE.validate(temporary)
            self.assertEqual(result["metadata"]["exact_runs"], 3)
            self.assertEqual(result["metadata"]["precision_trials"], 432)
            self.assertEqual(validation["status"], "passed")
            self.assertNotIn(
                b"\r\n", (Path(temporary) / "pipeline_comparison.csv").read_bytes()
            )


if __name__ == "__main__":
    unittest.main()
