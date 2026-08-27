from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sieve_replay.capture import load_prompt_records


class RouterCaptureInputTest(unittest.TestCase):
    def test_prompt_loader_preserves_request_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prompts.jsonl"
            path.write_text(
                '{"request_id":"a","prompt":"first"}\n'
                '{"request_id":2,"prompt":"second"}\n',
                encoding="utf-8",
            )
            records = load_prompt_records(path)
        self.assertEqual(tuple(record.request_id for record in records), ("a", 2))
        self.assertEqual(tuple(record.prompt for record in records), ("first", "second"))

    def test_prompt_loader_rejects_duplicate_request_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prompts.jsonl"
            path.write_text(
                '{"request_id":"a","prompt":"first"}\n'
                '{"request_id":"a","prompt":"second"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate request_id"):
                load_prompt_records(path)

    def test_prompt_loader_rejects_nonpositive_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_prompts must be positive"):
            load_prompt_records("unused.jsonl", max_prompts=0)


if __name__ == "__main__":
    unittest.main()
