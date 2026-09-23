from __future__ import annotations

import unittest

from sieve_replay.model.partial_attention import (
    quantize_scalar,
    run_partial_attention_sweep,
    run_partial_attention_trial,
)


class PartialAttentionTest(unittest.TestCase):
    def test_bfloat16_quantization_is_distinct_from_float32(self) -> None:
        value = 1.0001
        self.assertNotEqual(quantize_scalar(value, "fp32"), value)
        self.assertEqual(quantize_scalar(value, "bf16"), 1.0)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            quantize_scalar(value, "int8")

    def test_trial_is_deterministic_and_has_all_formats(self) -> None:
        arguments = {
            "context_length": 64,
            "total_pseudo_channels": 4,
            "logit_std": 1.0,
            "seed": 7,
            "head_dim": 16,
        }
        first = run_partial_attention_trial(**arguments)
        second = run_partial_attention_trial(**arguments)
        self.assertEqual(first, second)
        self.assertEqual(
            {row["scalar_format"] for row in first}, {"fp32", "bf16", "fp16"}
        )
        fp32 = next(row for row in first if row["scalar_format"] == "fp32")
        self.assertEqual(fp32["vs_fp32_partial_max_abs_error"], 0.0)
        self.assertGreaterEqual(fp32["cosine_similarity"], 0.999)

    def test_sweep_builds_complete_factorial_matrix(self) -> None:
        rows = run_partial_attention_sweep(
            context_lengths=(32, 64),
            total_pseudo_channels=(2, 4),
            logit_stds=(1.0, 2.0),
            seeds=(1, 2),
            head_dim=8,
        )
        self.assertEqual(len(rows), 2 * 2 * 2 * 2 * 3)


if __name__ == "__main__":
    unittest.main()
