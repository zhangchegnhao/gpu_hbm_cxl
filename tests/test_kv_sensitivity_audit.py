from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_kv_read_sensitivity import (
    compare_modes, validate_matrix, check_comparison, verify_cache_entry,
)
from scripts.run_kv_read_sensitivity import digest
from sieve_replay.ramulator.kv_read_workload import KVReadWorkloadResult
from sieve_replay.ramulator.mixed_workload import SieveCycleV1Config
from sieve_replay.report.writer import sha256_file


ROOT = Path(__file__).resolve().parents[1]


class KvSensitivityAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        directory = ROOT / "results/kv_read_sensitivity_v1"
        cls.plan = json.loads((directory / "kv_read_sensitivity_plan.json").read_text())
        cls.exact = json.loads((directory / "kv_read_sensitivity_exact_results.json").read_text())
        cls.comparison = json.loads((directory / "kv_read_sensitivity_comparison.json").read_text())
        cls.base = json.loads((ROOT / "results/kv_read_experiment_v1/kv_read_plan.json").read_text())
        cls.cycle = SieveCycleV1Config.load(ROOT / "configs/ramulator/sieve_hbm3e_cycle_v1.json")

    def test_missing_workload_is_rejected_despite_complete_summary(self) -> None:
        exact = deepcopy(self.exact)
        exact["entries"].pop(next(iter(exact["entries"])))
        with self.assertRaisesRegex(ValueError, "coverage"):
            validate_matrix(self.plan, exact, self.base, self.cycle)

    def test_relabelled_controller_or_duplicate_consumer_is_rejected(self) -> None:
        plan = deepcopy(self.plan)
        plan["variants"][1]["read_buffer_size"] = 256
        with self.assertRaisesRegex(ValueError, "variants"):
            validate_matrix(plan, self.exact, self.base, self.cycle)
        base = deepcopy(self.base)
        base["consumers"][1] = base["consumers"][0]
        with self.assertRaisesRegex(ValueError, "consumer matrix"):
            validate_matrix(self.plan, self.exact, base, self.cycle)

    def test_derived_comparison_cannot_change_without_detecting_error(self) -> None:
        rows = deepcopy(self.comparison["rows"])
        rows[0]["delta_total_completion_us"] += 1
        with self.assertRaisesRegex(ValueError, "derived value"):
            check_comparison(rows, self.exact["entries"])

    def test_cache_rejected_when_recorded_hash_or_content_changes(self) -> None:
        entry = deepcopy(next(iter(self.exact["entries"].values())))
        context = self.exact["cache_context"]
        cache_input = {"context": context, "variant": entry["variant"], "shape": entry["shape"]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / (digest(cache_input) + ".json")
            cache = {"cache_input": cache_input, "result": entry["result"]}
            path.write_text(json.dumps(cache))
            entry["cache_file_sha256"] = sha256_file(path)
            verify_cache_entry(entry, context, root)
            changed = deepcopy(cache)
            changed["result"]["gpu_completed_requests"] += 1
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_cache_entry(entry, context, root)
            entry["cache_file_sha256"] = sha256_file(path)
            with self.assertRaisesRegex(ValueError, "cache content"):
                verify_cache_entry(entry, context, root)

    def test_modes_use_their_own_isolated_control_and_absent_stream_ratio(self) -> None:
        rows = {r["mode"]: r for r in self.base["consumers"]
                if r["case"] == "b8_c4k" and r["placement"] == "gpu-only"}
        def result(mode: str) -> KVReadWorkloadResult:
            return KVReadWorkloadResult(**self.exact["entries"][f"baseline-rb256-dual:{rows[mode]['shape_key']}"]["result"])
        expert, kv, combined = (result(m) for m in ("expert-only", "kv-only", "combined"))
        values = compare_modes(expert, kv, combined)
        self.assertIsNone(values["pim_expert_completion_ratio"])
        self.assertAlmostEqual(values["combined_minus_isolated_max_us"], 20.399496)
        # Different controller isolated timing must change the denominator,
        # even when combined completion happens to be identical.
        slower_kv = replace(kv, total_completion_cycles=kv.total_completion_cycles * 2)
        matched = compare_modes(expert, slower_kv, combined)
        self.assertAlmostEqual(matched["combined_minus_isolated_max_us"], combined.total_completion_us - slower_kv.total_completion_us)
        self.assertNotEqual(matched["combined_minus_isolated_max_us"], values["combined_minus_isolated_max_us"])


if __name__ == "__main__":
    unittest.main()
