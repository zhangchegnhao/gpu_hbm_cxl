from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sieve_replay.config import load_configuration
from sieve_replay.trace import load_trace_set, sha256_file
from sieve_replay.trace_manifest import TraceManifest


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "configs/experiments/full_decode_synthetic.json"


class TraceManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.loaded = load_configuration(EXPERIMENT)
        self.trace_set = load_trace_set(
            self.loaded.experiment.trace_path,
            self.loaded.model,
            self.loaded.experiment.layers,
            self.loaded.experiment.steps,
        )

    def _manifest(self) -> dict[str, object]:
        model = self.loaded.model
        return {
            "schema_version": 1,
            "trace_file": str(self.loaded.experiment.trace_path),
            "trace_sha256": sha256_file(self.loaded.experiment.trace_path),
            "model": {
                "name": model.name,
                "revision": "test-revision",
                "dtype": model.dtype,
                "num_hidden_layers": model.num_hidden_layers,
                "num_experts": model.num_experts,
                "num_experts_per_tok": model.num_experts_per_tok,
            },
            "workload": {
                "dataset": "synthetic-test",
                "dataset_revision": "v1",
                "split": "test",
                "prompt_count": 8,
                "batch_size": 8,
                "decode_steps": 2,
                "seed": 0,
            },
            "capture": {
                "framework": "test",
                "framework_version": "1",
                "torch_version": "not-used",
                "device_map": "not-used",
                "created_at_utc": "2026-08-26T00:00:00Z",
                "timing_source": "routing-only; no hardware timing captured",
            },
        }

    def test_manifest_binds_trace_and_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps(self._manifest()), encoding="utf-8")
            manifest = TraceManifest.load_and_validate(
                path,
                self.loaded.experiment.trace_path,
                self.loaded.model,
                self.trace_set,
            )
            self.assertEqual(manifest.raw["schema_version"], 1)

    def test_manifest_rejects_stale_trace_hash(self) -> None:
        raw = self._manifest()
        raw["trace_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                TraceManifest.load_and_validate(
                    path,
                    self.loaded.experiment.trace_path,
                    self.loaded.model,
                    self.trace_set,
                )


if __name__ == "__main__":
    unittest.main()
