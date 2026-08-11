from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


STANDARD_FREQUENCIES_HZ = (250.0, 500.0, 1_000.0, 2_000.0, 3_000.0, 4_000.0, 6_000.0, 8_000.0)


@dataclass
class AudioConfig:
    sample_rate: int = 48_000
    block_size: int = 256
    input_channels: int = 1
    output_channels: int = 2


@dataclass
class AudiogramConfig:
    frequencies_hz: list[float] = field(default_factory=lambda: list(STANDARD_FREQUENCIES_HZ))
    hearing_loss_db_hl: list[float] = field(default_factory=lambda: [0.0] * 8)
    manual_gain_db: list[float] | None = None


@dataclass
class CompressionConfig:
    knee_dbfs: float = -45.0
    ratio: float = 2.0
    attack_ms: float = 8.0
    release_ms: float = 120.0
    low_level_gate_dbfs: float = -78.0
    max_gain_db: float = 12.0


@dataclass
class MusicConfig:
    enabled: bool = True
    harmonic_boost_db: float = 1.5
    timbre_preserve_mix: float = 0.45
    melody_relief_db: float = 1.25
    transient_attack_scale: float = 4.0
    pitch_min_hz: float = 80.0
    pitch_max_hz: float = 1_000.0
    confidence_threshold: float = 0.55


@dataclass
class CAPEConfig:
    enabled: bool = False
    model_path: str = ""
    listener_profile_path: str = ""
    decision_interval_blocks: int = 16
    min_hold_ms: float = 300.0
    crossfade_ms: float = 200.0
    strengths: list[float] = field(default_factory=lambda: [0.35, 0.65, 1.0])
    min_effect: float = 0.015
    max_uncertainty: float = 0.20
    max_distortion: float = 0.18
    max_naturalness_drop: float = 0.08
    distortion_penalty: float = 0.35
    uncertainty_penalty: float = 0.25


@dataclass
class LimiterConfig:
    ceiling_dbfs: float = -6.0
    release_ms: float = 80.0
    startup_ramp_ms: float = 750.0


@dataclass
class CalibrationConfig:
    output_calibrated: bool = False
    calibrated_device: str = ""
    verified_by: str = ""
    notes: str = ""


@dataclass
class HearingAssistConfig:
    name: str = "safe-passthrough"
    description: str = "Zero-gain starting profile"
    audio: AudioConfig = field(default_factory=AudioConfig)
    audiogram: AudiogramConfig = field(default_factory=AudiogramConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    music: MusicConfig = field(default_factory=MusicConfig)
    cape: CAPEConfig = field(default_factory=CAPEConfig)
    limiter: LimiterConfig = field(default_factory=LimiterConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    def validate(self) -> None:
        audio = self.audio
        if audio.sample_rate not in {16_000, 32_000, 44_100, 48_000}:
            raise ValueError("sample_rate must be one of 16000, 32000, 44100, or 48000")
        if audio.block_size < 64 or audio.block_size > 2_048:
            raise ValueError("block_size must be between 64 and 2048 samples")
        if audio.input_channels != 1:
            raise ValueError("the current prototype accepts one microphone channel")
        if audio.output_channels not in {1, 2}:
            raise ValueError("output_channels must be 1 or 2")

        frequencies = np.asarray(self.audiogram.frequencies_hz, dtype=float)
        losses = np.asarray(self.audiogram.hearing_loss_db_hl, dtype=float)
        if frequencies.ndim != 1 or frequencies.size < 2:
            raise ValueError("audiogram requires at least two frequencies")
        if frequencies.size != losses.size:
            raise ValueError("audiogram frequencies and hearing-loss values must have equal length")
        if np.any(~np.isfinite(frequencies)) or np.any(np.diff(frequencies) <= 0):
            raise ValueError("audiogram frequencies must be finite and strictly increasing")
        if frequencies[0] < 100 or frequencies[-1] >= audio.sample_rate / 2:
            raise ValueError("audiogram frequencies must lie between 100 Hz and Nyquist")
        if np.any(~np.isfinite(losses)) or np.any((losses < -10) | (losses > 120)):
            raise ValueError("hearing-loss values must lie between -10 and 120 dB HL")
        if self.audiogram.manual_gain_db is not None:
            manual = np.asarray(self.audiogram.manual_gain_db, dtype=float)
            if manual.size != frequencies.size or np.any(~np.isfinite(manual)):
                raise ValueError("manual_gain_db must match the audiogram frequency count")
            if np.any((manual < 0) | (manual > self.compression.max_gain_db)):
                raise ValueError("manual gains must be between 0 and max_gain_db")

        compression = self.compression
        if not -80.0 <= compression.knee_dbfs <= -15.0:
            raise ValueError("compression knee must be between -80 and -15 dBFS")
        if not 1.0 <= compression.ratio <= 5.0:
            raise ValueError("compression ratio must be between 1 and 5")
        if not 1.0 <= compression.attack_ms <= 100.0:
            raise ValueError("attack_ms must be between 1 and 100")
        if not 10.0 <= compression.release_ms <= 2_000.0:
            raise ValueError("release_ms must be between 10 and 2000")
        if not 0.0 <= compression.max_gain_db <= 24.0:
            raise ValueError("max_gain_db must be between 0 and 24 dB")
        if compression.low_level_gate_dbfs >= compression.knee_dbfs:
            raise ValueError("low-level gate must be below the compression knee")

        music = self.music
        if not 0.0 <= music.harmonic_boost_db <= 3.0:
            raise ValueError("harmonic boost must be between 0 and 3 dB")
        if not 0.0 <= music.timbre_preserve_mix <= 1.0:
            raise ValueError("timbre_preserve_mix must lie between 0 and 1")
        if not 0.0 <= music.melody_relief_db <= 3.0:
            raise ValueError("melody_relief_db must be between 0 and 3 dB")
        if not 1.0 <= music.transient_attack_scale <= 10.0:
            raise ValueError("transient_attack_scale must be between 1 and 10")
        if not 40.0 <= music.pitch_min_hz < music.pitch_max_hz <= 1_500.0:
            raise ValueError("invalid music pitch range")
        if not 0.0 <= music.confidence_threshold <= 1.0:
            raise ValueError("pitch confidence threshold must lie between 0 and 1")

        cape = self.cape
        if cape.enabled and (not cape.model_path or not cape.listener_profile_path):
            raise ValueError("enabled CAPE requires model_path and listener_profile_path")
        if not 1 <= cape.decision_interval_blocks <= 1_024:
            raise ValueError("CAPE decision_interval_blocks must be between 1 and 1024")
        if not 0.0 <= cape.min_hold_ms <= 10_000.0:
            raise ValueError("CAPE min_hold_ms must be between 0 and 10000")
        if not 0.0 <= cape.crossfade_ms <= 2_000.0:
            raise ValueError("CAPE crossfade_ms must be between 0 and 2000")
        strengths = np.asarray(cape.strengths, dtype=float)
        if strengths.ndim != 1 or strengths.size == 0 or np.any(~np.isfinite(strengths)):
            raise ValueError("CAPE strengths must be a non-empty finite list")
        if np.any((strengths <= 0.0) | (strengths > 1.0)):
            raise ValueError("CAPE strengths must lie in (0, 1]")
        if np.any(np.diff(strengths) <= 0.0):
            raise ValueError("CAPE strengths must be strictly increasing")
        if not -1.0 <= cape.min_effect <= 1.0:
            raise ValueError("CAPE min_effect must lie between -1 and 1")
        for name in ("max_uncertainty", "max_distortion", "max_naturalness_drop"):
            value = getattr(cape, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"CAPE {name} must lie between 0 and 1")
        for name in ("distortion_penalty", "uncertainty_penalty"):
            value = getattr(cape, name)
            if not 0.0 <= value <= 10.0:
                raise ValueError(f"CAPE {name} must lie between 0 and 10")

        limiter = self.limiter
        if not -18.0 <= limiter.ceiling_dbfs <= -1.0:
            raise ValueError("digital limiter ceiling must be between -18 and -1 dBFS")
        if not 10.0 <= limiter.release_ms <= 2_000.0:
            raise ValueError("limiter release must be between 10 and 2000 ms")
        if not 0.0 <= limiter.startup_ramp_ms <= 5_000.0:
            raise ValueError("startup ramp must be between 0 and 5000 ms")

    def prescribed_gain_db(self) -> np.ndarray:
        """Return conservative research gains, not a clinical prescription formula."""
        if self.audiogram.manual_gain_db is not None:
            gains = np.asarray(self.audiogram.manual_gain_db, dtype=np.float32)
        else:
            losses = np.asarray(self.audiogram.hearing_loss_db_hl, dtype=np.float32)
            gains = 0.35 * np.maximum(losses - 20.0, 0.0)
        return np.clip(gains, 0.0, self.compression.max_gain_db).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SECTIONS: dict[str, type[Any]] = {
    "audio": AudioConfig,
    "audiogram": AudiogramConfig,
    "compression": CompressionConfig,
    "music": MusicConfig,
    "cape": CAPEConfig,
    "limiter": LimiterConfig,
    "calibration": CalibrationConfig,
}


def _section(section_type: type[Any], values: dict[str, Any]) -> Any:
    valid = set(section_type.__dataclass_fields__)
    unknown = set(values) - valid
    if unknown:
        raise ValueError(f"unknown keys in {section_type.__name__}: {sorted(unknown)}")
    return section_type(**values)


def load_hearing_profile(path: str | Path | None = None) -> HearingAssistConfig:
    config = HearingAssistConfig()
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        allowed = {"name", "description", *_SECTIONS}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown profile sections: {sorted(unknown)}")
        if "name" in raw:
            config.name = str(raw["name"])
        if "description" in raw:
            config.description = str(raw["description"])
        for name, section_type in _SECTIONS.items():
            if name in raw:
                if not isinstance(raw[name], dict):
                    raise ValueError(f"{name} must be a mapping")
                setattr(config, name, _section(section_type, raw[name]))
        profile_root = Path(path).resolve().parent
        if config.cape.model_path:
            model_path = Path(config.cape.model_path)
            if not model_path.is_absolute():
                config.cape.model_path = str((profile_root / model_path).resolve())
        if config.cape.listener_profile_path:
            listener_path = Path(config.cape.listener_profile_path)
            if not listener_path.is_absolute():
                config.cape.listener_profile_path = str((profile_root / listener_path).resolve())
    config.validate()
    return config
