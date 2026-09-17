import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_kv_cycle_v1 import summarize_replay, sha256


class KvSummaryValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.case = "b8_c4k"
        self.policy = "gpu-only"
        self.result = self.root / "results/full_decode_real_kv_cycle_v1" / self.case / self.policy
        self.result.mkdir(parents=True)
        self.paths = {
            "model": "model.json", "hardware": "hardware.json",
            "pim_timing_table": "pim.json", "contention_timing_table": "contention.json",
            "ramulator_cycle_config": "cycle.json",
        }
        for name in self.paths.values():
            (self.root / name).write_text("{}\n", encoding="utf-8")
        (self.root / "contention.evidence.json").write_text("{}\n", encoding="utf-8")
        (self.root / "pim.evidence.json").write_text("{}\n", encoding="utf-8")
        exact = self.root / "results/kv_workload_timing_v1/exact_workloads.json"
        exact.parent.mkdir(parents=True, exist_ok=True)
        exact.write_text(json.dumps({"summary": {"failed": 0, "remaining": 0, "completed": 1, "required_unique_shapes": 1}, "entries": {"fixture-shape": {}}, "interpolated_workloads": 0}), encoding="utf-8")
        self.trace_meta = {
            "case": self.case, "batch_size": 8, "context_length": 4096,
            "trace_sha256": "t" * 64, "manifest_sha256": "m" * 64,
            "prompt_sha256": "p" * 64, "trace_relpath": "trace.jsonl",
            "manifest_relpath": "manifest.json", "config_paths": self.paths,
            "config_snapshot": {},
        }
        rows = [{"step": s, "layer": l} for s in range(8) for l in range(48)]
        with (self.result / "layers.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "layer"]); writer.writeheader(); writer.writerows(rows)
        with (self.result / "events.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "layer", "category", "duration_us"]); writer.writeheader()
            writer.writerows({"step": s, "layer": l, "category": "attention", "duration_us": "1"} for s in range(8) for l in range(48))
        (self.result / "summary.json").write_text(json.dumps({
            "policy": self.policy, "timing_backend": "ramulator-contention-v1", "total_latency_us": 384,
            "throughput_request_tokens_per_s": 1, "critical_path_duration_us_by_category": {"attention": 384},
            "memory": {"peak_kv_cache_bytes": 1, "peak_total_bytes": 2},
        }), encoding="utf-8")
        experiment = {**self.paths, "trace": "trace.jsonl", "trace_manifest": "manifest.json"}
        hashes = {f"{key}_sha256": sha256(self.root / value) for key, value in self.paths.items()}
        hashes.update({"trace_sha256": self.trace_meta["trace_sha256"], "trace_manifest_sha256": self.trace_meta["manifest_sha256"]})
        self.manifest = {"policy": self.policy, "experiment": experiment, "input_hashes": hashes}
        self.write_manifest()
        stage = {"exact_workloads_sha256": sha256(exact), "cases": {self.case: {"configuration_snapshot": {},
            "pim_timing_table_sha256": sha256(self.root / "pim.json"), "pim_evidence_sha256": sha256(self.root / "pim.evidence.json"),
            "contention_timing_table_sha256": sha256(self.root / "contention.json"), "contention_evidence_sha256": sha256(self.root / "contention.evidence.json"), "replays": {self.policy: {
            "layers_sha256": sha256(self.result / "layers.csv"),
            "events_sha256": sha256(self.result / "events.csv"),
            "summary_sha256": sha256(self.result / "summary.json"),
            "run_manifest_sha256": sha256(self.result / "run_manifest.json"),
        }}}}}
        (self.result.parent.parent / "stage_manifest.json").write_text(json.dumps(stage), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def write_manifest(self):
        (self.result / "run_manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        stage_path = self.result.parent.parent / "stage_manifest.json"
        if stage_path.is_file():
            stage = json.loads(stage_path.read_text(encoding="utf-8"))
            stage["cases"][self.case]["replays"][self.policy]["run_manifest_sha256"] = sha256(self.result / "run_manifest.json")
            stage_path.write_text(json.dumps(stage), encoding="utf-8")

    def test_stale_contention_hash_rejected(self):
        self.manifest["input_hashes"]["contention_timing_table_sha256"] = "0" * 64
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "contention_timing_table_sha256 mismatch"):
            summarize_replay(self.root, self.case, self.policy, self.trace_meta)

    def test_complete_replay_is_summarized(self):
        row = summarize_replay(self.root, self.case, self.policy, self.trace_meta)
        self.assertEqual(row["attention_latency_us"], 384)
        self.assertEqual(row["attention_latency_share"], 1)
        self.assertEqual(row["expert_latency_us"], 0)

    def test_run_manifest_path_mismatch_rejected(self):
        self.manifest["experiment"]["trace"] = "stale.jsonl"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "trace path differs"):
            summarize_replay(self.root, self.case, self.policy, self.trace_meta)

    def test_duplicate_layer_rejected(self):
        path = self.result / "layers.csv"
        with path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        rows[-1]["step"], rows[-1]["layer"] = "0", "0"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "layer"]); writer.writeheader(); writer.writerows(rows)
        stage_path = self.result.parent.parent / "stage_manifest.json"
        stage = json.loads(stage_path.read_text(encoding="utf-8")); stage["cases"][self.case]["replays"][self.policy]["layers_sha256"] = sha256(path); stage_path.write_text(json.dumps(stage), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unique complete"):
            summarize_replay(self.root, self.case, self.policy, self.trace_meta)


if __name__ == "__main__":
    unittest.main()
