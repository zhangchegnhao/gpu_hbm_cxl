from __future__ import annotations

import unittest
from pathlib import Path

from sieve_replay.config import load_configuration
from sieve_replay.ramulator import SieveCycleConfig, candidate_token_counts


ROOT = Path(__file__).resolve().parents[1]
ANALYTIC_EXPERIMENT = ROOT / "configs/experiments/single_layer_smoke.json"
CYCLE_CONFIG = ROOT / "configs/ramulator/sieve_hbm3e_cycle_v0.json"


class RamulatorCycleConfigurationTest(unittest.TestCase):
    def test_sieve_topology_is_explicit(self) -> None:
        cycle = SieveCycleConfig.load(CYCLE_CONFIG)
        self.assertEqual(cycle.total_channels, 128)
        self.assertEqual(cycle.total_pseudo_channels, 256)
        self.assertEqual(cycle.banks_per_pseudo_channel, 24)
        self.assertEqual(cycle.organization_count, [1, 2, 1, 6, 4, 16384, 256])

    def test_candidate_counts_cover_every_sieve_prefix(self) -> None:
        configuration = load_configuration(ANALYTIC_EXPERIMENT)
        counts = candidate_token_counts(configuration)
        self.assertEqual(len(counts), 50)
        self.assertEqual(counts[0], 1)
        self.assertEqual(counts[-1], 64)
        self.assertIn(17, counts)
        self.assertIn(51, counts)


if __name__ == "__main__":
    unittest.main()
