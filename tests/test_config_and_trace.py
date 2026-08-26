from __future__ import annotations

import unittest
from pathlib import Path

from sieve_replay.config import load_configuration
from sieve_replay.trace import load_trace


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "configs/experiments/single_layer_smoke.json"


class ConfigAndTraceTest(unittest.TestCase):
    def test_official_qwen3_shape_is_loaded(self) -> None:
        loaded = load_configuration(EXPERIMENT)
        model = loaded.model
        self.assertEqual(model.num_hidden_layers, 48)
        self.assertEqual(model.num_attention_heads, 32)
        self.assertEqual(model.num_key_value_heads, 4)
        self.assertEqual(model.num_experts, 128)
        self.assertEqual(model.num_experts_per_tok, 8)

    def test_hotspot_trace_counts_assignments(self) -> None:
        loaded = load_configuration(EXPERIMENT)
        trace = load_trace(loaded.experiment.trace_path, loaded.model, layer=0, step=0)
        counts = {load.expert_id: load.token_count for load in trace.expert_loads}
        self.assertEqual(trace.batch_size, 8)
        self.assertEqual(trace.total_expert_assignments, 64)
        self.assertEqual(counts[0], 8)
        self.assertEqual(counts[1], 5)
        self.assertEqual(counts[2], 3)


if __name__ == "__main__":
    unittest.main()
