from __future__ import annotations

import json
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from hearing_assist.cape import (
    CAPEEffectModel,
    CAPEPolicy,
    ListenerProfile,
    PolicySettings,
    TREATMENT_NAMES,
    load_examples,
    save_examples,
)
from hearing_assist.config import HearingAssistConfig
from hearing_assist.dsp import HearingAssistProcessor
from hearing_assist.synthetic import generate_synthetic_examples


class CAPEModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.examples = generate_synthetic_examples(
            listener_count=4,
            music_segments_per_listener=4,
            seed=90210,
        )
        cls.model = CAPEEffectModel(
            ensemble_size=3,
            hidden_features=24,
            ridge=0.1,
            seed=90210,
        )
        cls.metrics = cls.model.fit(cls.examples)

    def test_jsonl_schema_round_trip(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "cape.jsonl"
            count = save_examples(path, self.examples[:16])
            loaded = load_examples(path)
        self.assertEqual(count, 16)
        self.assertEqual(len(loaded), 16)
        self.assertEqual(loaded[0].listener.listener_id, self.examples[0].listener.listener_id)

    def test_model_learns_and_round_trips_without_prediction_drift(self) -> None:
        self.assertLess(self.metrics["mae"], 0.08)
        item = self.examples[1]
        expected = self.model.raw_effect(item.listener, item.music, item.treatment)
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "cape.npz"
            self.model.save(path)
            restored = CAPEEffectModel.load(path)
            actual = restored.raw_effect(item.listener, item.music, item.treatment)
        for left, right in zip(expected[:3], actual[:3]):
            np.testing.assert_allclose(left, right, atol=1e-7)
        self.assertAlmostEqual(expected[3], actual[3], places=7)

    def test_policy_returns_explainable_bounded_decision(self) -> None:
        item = self.examples[0]
        policy = CAPEPolicy(
            self.model,
            item.listener,
            PolicySettings(
                min_effect=-1.0,
                max_uncertainty=1.0,
                max_distortion=1.0,
                max_naturalness_drop=1.0,
            ),
        )
        decision = policy.select(item.music)
        self.assertIn(decision.treatment.name, TREATMENT_NAMES)
        self.assertNotEqual(decision.treatment.name, "reference")
        self.assertEqual(len(decision.delta), 5)
        self.assertTrue(np.isfinite(decision.score))

    def test_high_required_effect_causes_reference_abstention(self) -> None:
        item = self.examples[0]
        policy = CAPEPolicy(
            self.model,
            item.listener,
            PolicySettings(
                min_effect=1.0,
                max_uncertainty=1.0,
                max_distortion=1.0,
                max_naturalness_drop=1.0,
            ),
        )
        decision = policy.select(item.music)
        self.assertEqual(decision.treatment.name, "reference")
        self.assertEqual(decision.reason, "predicted_effect_below_threshold")

    def test_cape_model_controls_streaming_dsp_and_remains_limited(self) -> None:
        listener = self.examples[0].listener
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_path = root / "cape.npz"
            listener_path = root / "listener.json"
            self.model.save(model_path)
            listener_path.write_text(
                json.dumps(listener.to_dict(), ensure_ascii=False), encoding="utf-8"
            )
            config = HearingAssistConfig()
            config.audiogram.hearing_loss_db_hl = list(listener.audiogram_db_hl)
            config.music.enabled = True
            config.cape.enabled = True
            config.cape.model_path = str(model_path)
            config.cape.listener_profile_path = str(listener_path)
            config.cape.decision_interval_blocks = 1
            config.cape.min_hold_ms = 0.0
            config.cape.crossfade_ms = 0.0
            config.cape.min_effect = -1.0
            config.cape.max_uncertainty = 1.0
            config.cape.max_distortion = 1.0
            config.cape.max_naturalness_drop = 1.0
            config.limiter.startup_ramp_ms = 0.0
            config.limiter.ceiling_dbfs = -9.0
            config.validate()
            processor = HearingAssistProcessor(config)
            time = np.arange(config.audio.block_size, dtype=np.float32) / config.audio.sample_rate
            block = (0.1 * np.sin(2 * np.pi * 440.0 * time)).astype(np.float32)
            outputs = []
            reports = []
            for _ in range(20):
                output, report = processor.process_block(block)
                outputs.append(output)
                reports.append(report)
        self.assertTrue(np.all(np.isfinite(np.concatenate(outputs))))
        self.assertTrue(any(report.cape_action != "reference" for report in reports))
        ceiling = 10 ** (config.limiter.ceiling_dbfs / 20.0)
        self.assertLessEqual(float(np.max(np.abs(np.concatenate(outputs)))), ceiling + 1e-6)

    def test_listener_and_active_audiogram_mismatch_is_rejected(self) -> None:
        listener = self.examples[0].listener
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_path = root / "cape.npz"
            listener_path = root / "listener.json"
            self.model.save(model_path)
            listener_path.write_text(
                json.dumps(listener.to_dict(), ensure_ascii=False), encoding="utf-8"
            )
            config = HearingAssistConfig()
            config.cape.enabled = True
            config.cape.model_path = str(model_path)
            config.cape.listener_profile_path = str(listener_path)
            config.validate()
            with self.assertRaisesRegex(ValueError, "audiogram must exactly match"):
                HearingAssistProcessor(config)


if __name__ == "__main__":
    unittest.main()
