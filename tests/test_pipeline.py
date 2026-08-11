from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from music_ci.analysis.pitch import CausalPitchTracker
from music_ci.config import AppConfig
from music_ci.core.models import AnalysisFrame
from music_ci.core.pipeline import ResearchPipeline
from music_ci.io.audio import generate_demo_melody
from music_ci.io.results import save_results
from music_ci.strategies.ace import AceTopKPolicy


class PitchTrackerTests(unittest.TestCase):
    def test_tracks_220_hz_sine(self) -> None:
        sample_rate = 16_000
        tracker = CausalPitchTracker(sample_rate=sample_rate)
        time = np.arange(int(0.25 * sample_rate), dtype=np.float32) / sample_rate
        signal = np.sin(2 * np.pi * 220.0 * time).astype(np.float32)
        estimates = []
        for start in range(0, signal.size, 128):
            f0, confidence, _ = tracker.update(signal[start : start + 128])
            if confidence > 0.35:
                estimates.append(f0)
        self.assertTrue(estimates)
        self.assertLess(abs(float(np.median(estimates)) - 220.0), 5.0)


class SelectionTests(unittest.TestCase):
    def test_ace_selects_exact_top_k(self) -> None:
        energy = np.linspace(0.0, 1.0, 22, dtype=np.float32)
        zeros = np.zeros(22, dtype=np.float32)
        analysis = AnalysisFrame(
            envelopes=energy,
            normalized_energy=energy,
            f0_hz=0.0,
            f0_confidence=0.0,
            harmonicity=0.0,
            harmonic_match=zeros,
            redundancy=np.zeros((22, 22), dtype=np.float32),
        )
        selected = AceTopKPolicy(active_channels=8).select(analysis)
        np.testing.assert_array_equal(selected.selected_indices, np.arange(14, 22))
        self.assertEqual(int(selected.selected_mask.sum()), 8)


class PipelineTests(unittest.TestCase):
    def test_end_to_end_shape_selection_and_determinism(self) -> None:
        config = AppConfig()
        samples = generate_demo_melody(config.audio.sample_rate)[:4_096]
        first = ResearchPipeline(config).process(samples)
        second = ResearchPipeline(config).process(samples)
        expected_frames = int(np.ceil(samples.size / config.audio.frame_samples))
        self.assertEqual(first.electrodogram.shape, (22, expected_frames))
        self.assertTrue(
            all(frame.selection.selected_mask.sum() == 8 for frame in first.frame_results)
        )
        np.testing.assert_allclose(first.electrodogram, second.electrodogram, atol=1e-7)

    def test_result_contract_is_written(self) -> None:
        config = AppConfig()
        samples = generate_demo_melody(config.audio.sample_rate)[:1_024]
        pipeline = ResearchPipeline(config)
        result = pipeline.process(samples)
        expected = {
            "electrodogram.npy",
            "selected_channels.csv",
            "f0_trace.csv",
            "channel_scores.csv",
            "runtime.csv",
            "vocoder.wav",
            "metrics.json",
            "config.json",
        }
        with TemporaryDirectory() as temporary:
            destination = save_results(temporary, result, config, pipeline.filterbank)
            self.assertTrue(expected.issubset({path.name for path in Path(destination).iterdir()}))
            stored = np.load(Path(destination) / "electrodogram.npy")
            np.testing.assert_array_equal(stored, result.electrodogram)


if __name__ == "__main__":
    unittest.main()
