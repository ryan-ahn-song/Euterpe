from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from hearing_assist.config import HearingAssistConfig, load_hearing_profile
from hearing_assist.dsp import HearingAssistProcessor
from hearing_assist.engine import process_array, process_wav
from hearing_assist.live import LiveHearingAssist
from music_ci.io.audio import write_wav_mono


ROOT = Path(__file__).resolve().parents[1]


def _test_config() -> HearingAssistConfig:
    config = HearingAssistConfig()
    config.music.enabled = False
    config.music.harmonic_boost_db = 0.0
    config.limiter.startup_ramp_ms = 0.0
    config.limiter.ceiling_dbfs = -1.0
    config.validate()
    return config


class HearingAssistConfigTests(unittest.TestCase):
    def test_all_packaged_profiles_validate(self) -> None:
        paths = sorted((ROOT / "configs").glob("hearing_assist_*.yaml"))
        self.assertGreaterEqual(len(paths), 3)
        for path in paths:
            with self.subTest(path=path.name):
                load_hearing_profile(path)

    def test_unsafe_max_gain_is_rejected(self) -> None:
        config = HearingAssistConfig()
        config.compression.max_gain_db = 30.0
        with self.assertRaises(ValueError):
            config.validate()


class HearingAssistDSPTests(unittest.TestCase):
    def test_zero_gain_path_is_sample_aligned_after_declared_delay(self) -> None:
        config = _test_config()
        rng = np.random.default_rng(20_260_811)
        source = (0.04 * rng.standard_normal(12_345)).astype(np.float32)
        output, _ = process_array(source, config)
        self.assertEqual(source.shape, output.shape)
        self.assertLess(float(np.max(np.abs(source - output))), 1e-6)

    def test_limiter_never_exceeds_digital_ceiling(self) -> None:
        config = _test_config()
        config.audiogram.manual_gain_db = [12.0] * 8
        config.compression.max_gain_db = 12.0
        config.limiter.ceiling_dbfs = -12.0
        config.validate()
        time = np.arange(config.audio.sample_rate, dtype=np.float32) / config.audio.sample_rate
        source = (0.95 * np.sin(2 * np.pi * 1_000 * time)).astype(np.float32)
        output, _ = process_array(source, config)
        ceiling = 10 ** (config.limiter.ceiling_dbfs / 20)
        self.assertLessEqual(float(np.max(np.abs(output))), ceiling + 1e-6)

    def test_compression_applies_more_gain_to_quiet_signal(self) -> None:
        def measured_gain(amplitude: float) -> float:
            config = _test_config()
            config.audiogram.manual_gain_db = [12.0] * 8
            config.compression.max_gain_db = 12.0
            config.validate()
            time = np.arange(config.audio.sample_rate, dtype=np.float32) / config.audio.sample_rate
            source = (amplitude * np.sin(2 * np.pi * 1_000 * time)).astype(np.float32)
            output, _ = process_array(source, config)
            region = slice(source.size // 2, None)
            return float(
                20
                * np.log10(
                    np.sqrt(np.mean(output[region] ** 2))
                    / np.sqrt(np.mean(source[region] ** 2))
                )
            )

        self.assertGreater(measured_gain(0.001), measured_gain(0.1) + 3.0)

    def test_nonfinite_input_triggers_muted_fail_safe(self) -> None:
        config = _test_config()
        processor = HearingAssistProcessor(config)
        broken = np.zeros(config.audio.block_size, dtype=np.float32)
        broken[10] = np.nan
        output, diagnostics = processor.process_block(broken)
        self.assertTrue(diagnostics.fail_safe_triggered)
        self.assertTrue(np.all(output == 0.0))
        self.assertEqual(processor.fail_safe_count, 1)

    def test_end_to_end_wav_writes_auditable_report(self) -> None:
        config = _test_config()
        time = np.arange(4_800, dtype=np.float32) / config.audio.sample_rate
        source = (0.1 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "input.wav"
            output_path = root / "output.wav"
            report = root / "report"
            write_wav_mono(input_path, source, config.audio.sample_rate)
            metrics = process_wav(input_path, output_path, config, report)
            self.assertTrue(output_path.exists())
            self.assertTrue((report / "metrics.json").exists())
            self.assertTrue((report / "profile_snapshot.json").exists())
            self.assertTrue((report / "block_diagnostics.csv").exists())
            loaded = json.loads((report / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded["fail_safe_blocks"], 0)
            self.assertEqual(metrics["profile"], config.name)

    def test_live_callback_processes_mono_to_stereo_without_device(self) -> None:
        config = _test_config()
        session = LiveHearingAssist(config)
        indata = np.zeros((config.audio.block_size, 1), dtype=np.float32)
        outdata = np.full((config.audio.block_size, 2), np.nan, dtype=np.float32)
        session._callback(indata, outdata, config.audio.block_size, None, None)
        self.assertTrue(np.all(np.isfinite(outdata)))
        self.assertTrue(np.all(outdata[:, 0] == outdata[:, 1]))
        self.assertEqual(session.summary().blocks, 1)
