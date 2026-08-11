from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AudioConfig:
    sample_rate: int = 16_000
    frame_samples: int = 128
    context_ms: int = 48


@dataclass
class FilterBankConfig:
    num_channels: int = 22
    min_hz: float = 150.0
    max_hz: float = 7_500.0
    envelope_cutoff_hz: float = 400.0


@dataclass
class SelectionConfig:
    strategy: str = "msa"
    active_channels: int = 8
    energy_weight: float = 1.0
    harmonic_weight: float = 0.6
    continuity_weight: float = 0.2
    redundancy_weight: float = 0.3


@dataclass
class PitchConfig:
    min_hz: float = 80.0
    max_hz: float = 1_000.0
    confidence_threshold: float = 0.35


@dataclass
class RuntimeConfig:
    frame_deadline_ms: float = 8.0
    random_seed: int = 20_260_811


@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    filterbank: FilterBankConfig = field(default_factory=FilterBankConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    pitch: PitchConfig = field(default_factory=PitchConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self) -> None:
        if self.audio.sample_rate != 16_000:
            raise ValueError("The v0.1 research profile currently requires 16 kHz audio")
        if self.audio.frame_samples <= 0:
            raise ValueError("frame_samples must be positive")
        if self.filterbank.num_channels != 22:
            raise ValueError("The current research profile requires 22 analysis channels")
        if not 0 < self.selection.active_channels <= self.filterbank.num_channels:
            raise ValueError("active_channels must be between 1 and num_channels")
        if self.selection.strategy not in {"ace", "msa"}:
            raise ValueError("strategy must be 'ace' or 'msa'")
        nyquist = self.audio.sample_rate / 2
        if self.filterbank.max_hz >= nyquist:
            raise ValueError("filterbank.max_hz must be below Nyquist")
        if not 0 < self.pitch.min_hz < self.pitch.max_hz:
            raise ValueError("invalid pitch range")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    valid = set(instance.__dataclass_fields__)
    unknown = set(values) - valid
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    return type(instance)(**{**asdict(instance), **values})


def load_config(path: str | Path | None = None) -> AppConfig:
    config = AppConfig()
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        allowed = {"audio", "filterbank", "selection", "pitch", "runtime"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
        for name in allowed:
            if name in raw:
                setattr(config, name, _merge_dataclass(getattr(config, name), raw[name]))
    config.validate()
    return config

