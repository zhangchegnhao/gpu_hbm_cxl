from __future__ import annotations

import importlib.util
import tempfile
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_kv_capacity_sweep.py"
SPEC = importlib.util.spec_from_file_location("run_kv_capacity_sweep", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
KV_SWEEP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(KV_SWEEP)


class KvSweepTest(unittest.TestCase):
    def test_controlled_matrix_and_capacity_states(self) -> None:
        experiment = ROOT / "configs/experiments/full_decode_real_pilot_cycle_v0.json"
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(KV_SWEEP, "_plot"):
            result = KV_SWEEP.run_sweep(experiment, temporary)
            import json
            baseline = json.loads((Path(temporary) / "baseline_breakdown.json").read_text())
        rows = result["rows"]
        self.assertEqual(len(rows), 13)
        self.assertEqual({row["batch_size"] for row in rows}, {8, 16, 32})
        row = next(row for row in rows if row["batch_size"] == 8 and row["context_length"] == 4096)
        self.assertEqual(row["kv_cache_bytes"], 3_221_225_472)
        self.assertEqual(row["capacity_state"], "feasible")
        self.assertEqual(
            next(row for row in rows if row["batch_size"] == 16 and row["context_length"] == 32768)["capacity_state"],
            "infeasible",
        )
        self.assertEqual(result["metadata"]["result_classification"], "解析敏感性分析；不是A800实测，也不是Ramulator模拟")
        self.assertEqual(row["layers"], 48)
        self.assertEqual(row["decode_steps"], 1)
        self.assertAlmostEqual(
            row["throughput_request_tokens_per_s"] * row["total_latency_us"],
            row["batch_size"] * 1e6, delta=1.0,
        )
        configuration = KV_SWEEP.load_configuration(experiment)
        template = KV_SWEEP.load_trace_set(
            configuration.experiment.trace_path, configuration.model,
            configuration.experiment.layers, configuration.experiment.steps,
        ).batches[0]
        for batch in (8, 16, 32):
            short = KV_SWEEP._route_template(template, batch, 512)
            long = KV_SWEEP._route_template(template, batch, 32768)
            self.assertEqual(short.expert_loads, long.expert_loads)
            self.assertEqual(short.total_expert_assignments, batch * 8)
        self.assertLess(baseline["expert_latency_us"], baseline["total_latency_us"])
        self.assertAlmostEqual(sum(baseline["critical_path_duration_us_by_category"].values()),
                               baseline["total_latency_us"], places=5)


if __name__ == "__main__":
    unittest.main()
