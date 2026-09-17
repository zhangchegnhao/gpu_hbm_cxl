from __future__ import annotations

import fcntl
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# replay_kv_cycle_v1.py is normally executed with scripts/ as argv[0], so its
# sibling imports are intentionally top-level. Mirror that launch environment.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from scripts import build_kv_isolated_tables as isolated_builder
from scripts import replay_kv_cycle_v1 as replay
from sieve_replay.timing import RamulatorTimingTable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _attestation(table: Path, expected: dict) -> dict:
    return {
        "input_hashes": expected["input_hashes"],
        "generator_sha256": expected["generator_sha256"],
        "extension_sha256": expected["extension_sha256"],
        "table_sha256": _sha256(table),
        "snapshot": expected["snapshot"],
        "interpolation": "forbidden",
        "exact_workload": True,
        "runs_request_counts_match": True,
        "runs": [{"injected_requests": 1, "completed_requests": 1}],
        "table_entries": {"attention": 1, "expert_gemv": 1},
    }


class KvReplayValidationTest(unittest.TestCase):
    def test_isolated_attestation_rejects_stale_or_missing_hash(self) -> None:
        expected = {
            "input_hashes": {"trace_sha256": "a" * 64},
            "generator_sha256": "b" * 64,
            "extension_sha256": "c" * 64,
            "snapshot": {"experiment": "case.json"},
        }
        with tempfile.TemporaryDirectory() as directory:
            table = Path(directory) / "table.json"
            evidence = Path(directory) / "table.evidence.json"
            table.write_text("table\n", encoding="utf-8")
            evidence.write_text(
                json.dumps(_attestation(table, expected)), encoding="utf-8"
            )
            self.assertTrue(isolated_builder.attestation_matches(evidence, table, expected))

            stale = json.loads(evidence.read_text(encoding="utf-8"))
            stale["input_hashes"]["trace_sha256"] = "d" * 64
            evidence.write_text(json.dumps(stale), encoding="utf-8")
            self.assertFalse(isolated_builder.attestation_matches(evidence, table, expected))

            missing = _attestation(table, expected)
            del missing["generator_sha256"]
            evidence.write_text(json.dumps(missing), encoding="utf-8")
            self.assertFalse(isolated_builder.attestation_matches(evidence, table, expected))

    def test_attention_lookup_preserves_schema_v1_and_v2_contracts(self) -> None:
        metadata = {
            "units": "us",
            "ramulator_commit": "b30320bc9385b708e86b67ebb9f48858cc66d798",
            "extension_commit": "test",
            "hardware_config_sha256": "a" * 64,
            "trace_generator_sha256": "b" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            v1 = Path(directory) / "v1.json"
            v1.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "metadata": metadata,
                        "attention": [{"batch_size": 2, "context_length": 10, "duration_us": 1.0}],
                        "expert_gemv": [{"token_count": 1, "gwrite_us": 0.0, "compute_us": 1.0, "read_us": 0.0}],
                    }
                ),
                encoding="utf-8",
            )
            table_v1 = RamulatorTimingTable.load(v1)
            self.assertEqual(table_v1.attention_us(2, 10), 1.0)
            with self.assertRaises(ValueError):
                table_v1.attention_us((10, 10))

            v2 = Path(directory) / "v2.json"
            v2.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "metadata": metadata,
                        "attention": [{"context_lengths": [10, 10], "duration_us": 2.0}],
                        "expert_gemv": [{"token_count": 1, "gwrite_us": 0.0, "compute_us": 1.0, "read_us": 0.0}],
                    }
                ),
                encoding="utf-8",
            )
            table_v2 = RamulatorTimingTable.load(v2)
            self.assertEqual(table_v2.attention_us((10, 10)), 2.0)
            with self.assertRaises(ValueError):
                table_v2.attention_us(2, 10)

    def test_replay_rejects_an_active_fill_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            lock_path = cache / ".stage.lock"
            with lock_path.open("w") as held:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with mock.patch.object(replay, "CACHE", cache):
                    with self.assertRaises(BlockingIOError):
                        replay.main()


if __name__ == "__main__":
    unittest.main()
