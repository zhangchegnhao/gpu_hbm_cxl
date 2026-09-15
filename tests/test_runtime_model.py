from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from sieve_replay.config import load_configuration
from sieve_replay.policy import create_policy
from sieve_replay.timing import (
    AnalyticTimingModel,
    RuntimeExpertTimingModel,
    TimingAnchor,
)
from sieve_replay.trace import load_trace


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "configs/experiments/single_layer_smoke.json"


class OracleGuardTiming(AnalyticTimingModel):
    def placement_candidates(self, *_args: object) -> None:
        raise AssertionError("runtime policy must not query exact placement candidates")


class RuntimeTimingModelTest(unittest.TestCase):
    @staticmethod
    def _calibration() -> RuntimeExpertTimingModel:
        return RuntimeExpertTimingModel(
            metadata={
                "method": "piecewise-linear-isolated-v1",
                "source": "test",
                "scope": "test",
            },
            gpu_anchors=(
                TimingAnchor(0, 0.0),
                TimingAnchor(4, 8.0),
                TimingAnchor(8, 20.0),
            ),
            pim_anchors=(
                TimingAnchor(0, 0.0),
                TimingAnchor(32, 24.0),
                TimingAnchor(64, 56.0),
            ),
        )

    def test_piecewise_interpolation_and_extrapolation(self) -> None:
        calibration = self._calibration()
        self.assertEqual(calibration.gpu_expert_read_us(2), 4.0)
        self.assertEqual(calibration.gpu_expert_read_us(6), 14.0)
        self.assertEqual(calibration.gpu_expert_read_us(10), 26.0)
        self.assertEqual(calibration.pim_expert_pipeline_us(16), 12.0)
        self.assertEqual(calibration.pim_expert_pipeline_us(48), 40.0)
        self.assertFalse(calibration.gpu_prediction_is_extrapolated(8))
        self.assertTrue(calibration.gpu_prediction_is_extrapolated(9))
        self.assertFalse(calibration.pim_prediction_is_extrapolated(64))
        self.assertTrue(calibration.pim_prediction_is_extrapolated(65))

    @staticmethod
    def _payload() -> dict[str, object]:
        return {
            "schema_version": 1,
            "metadata": {
                "units": "us",
                "method": "piecewise-linear-isolated-v1",
                "source": "test",
                "scope": "test",
                "model_sha256": "1" * 64,
                "hardware_sha256": "2" * 64,
                "cycle_config_sha256": "3" * 64,
                "source_contention_table_sha256": "4" * 64,
                "generator_sha256": "5" * 64,
            },
            "gpu_expert_count_anchors": [
                {"expert_count": 0, "duration_us": 0.0},
                {"expert_count": 1, "duration_us": 1.0},
            ],
            "pim_token_count_anchors": [
                {"token_count": 0, "duration_us": 0.0},
                {"token_count": 1, "duration_us": 1.0},
            ],
        }

    def _load_payload(self, payload: dict[str, object]) -> RuntimeExpertTimingModel:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return RuntimeExpertTimingModel.load(path)

    def test_loader_rejects_non_hex_digest(self) -> None:
        payload = self._payload()
        metadata = payload["metadata"]
        assert isinstance(metadata, dict)
        metadata["model_sha256"] = "z" * 64
        with self.assertRaisesRegex(ValueError, "model_sha256"):
            self._load_payload(payload)

    def test_loader_rejects_non_finite_duration(self) -> None:
        payload = self._payload()
        anchors = payload["gpu_expert_count_anchors"]
        assert isinstance(anchors, list)
        anchors[1]["duration_us"] = math.nan
        with self.assertRaisesRegex(ValueError, "duration_us"):
            self._load_payload(payload)

    def test_loader_rejects_extra_fields(self) -> None:
        payload = self._payload()
        payload["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "extra"):
            self._load_payload(payload)

    def test_generated_calibration_matches_runtime_inputs(self) -> None:
        experiment = ROOT / "configs/experiments/full_decode_real_pilot_runtime_v1.json"
        loaded = load_configuration(experiment)
        calibration_path = loaded.experiment.runtime_calibration_path
        cycle_path = loaded.experiment.ramulator_cycle_config_path
        assert calibration_path is not None
        assert cycle_path is not None
        calibration = RuntimeExpertTimingModel.load(calibration_path)
        calibration.validate_inputs(
            loaded.experiment.model_path,
            loaded.experiment.hardware_path,
            cycle_path,
        )

    def test_runtime_policy_does_not_query_oracle_candidates(self) -> None:
        loaded = load_configuration(EXPERIMENT)
        trace = load_trace(
            loaded.experiment.trace_path,
            loaded.model,
            loaded.experiment.layer,
            loaded.experiment.step,
        )
        timing = OracleGuardTiming(
            loaded.model,
            loaded.hardware,
            self._calibration(),
        )
        decision = create_policy("sieve-runtime-v1", timing).place(trace)
        self.assertEqual(
            set(decision.gpu_experts) | set(decision.pim_experts),
            {load.expert_id for load in trace.expert_loads},
        )
        assert decision.search_report is not None
        self.assertFalse(decision.search_report["oracle_access_during_decision"])
        self.assertEqual(
            decision.search_report["candidate_count"],
            len(trace.expert_loads) + 1,
        )

    def test_runtime_policy_requires_calibration(self) -> None:
        loaded = load_configuration(EXPERIMENT)
        trace = load_trace(
            loaded.experiment.trace_path,
            loaded.model,
            loaded.experiment.layer,
            loaded.experiment.step,
        )
        timing = AnalyticTimingModel(loaded.model, loaded.hardware)
        with self.assertRaisesRegex(ValueError, "runtime_calibration"):
            create_policy("sieve-runtime-v1", timing).place(trace)


if __name__ == "__main__":
    unittest.main()
