from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample_poly


def _decode_pcm(raw: bytes, sample_width: int) -> NDArray[np.float32]:
    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        return (data - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32_768.0
    if sample_width == 3:
        bytes_ = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values = (
            bytes_[:, 0].astype(np.int32)
            | (bytes_[:, 1].astype(np.int32) << 8)
            | (bytes_[:, 2].astype(np.int32) << 16)
        )
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return values.astype(np.float32) / 8_388_608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2_147_483_648.0
    raise ValueError(f"Unsupported PCM sample width: {sample_width}")


def read_wav_mono(path: str | Path, target_sample_rate: int = 16_000) -> NDArray[np.float32]:
    with wave.open(str(path), "rb") as wav:
        if wav.getcomptype() != "NONE":
            raise ValueError("Only uncompressed PCM WAV files are supported")
        channels = wav.getnchannels()
        source_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        raw = wav.readframes(wav.getnframes())
    data = _decode_pcm(raw, sample_width)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    if source_rate != target_sample_rate:
        divisor = int(np.gcd(source_rate, target_sample_rate))
        data = resample_poly(
            data,
            target_sample_rate // divisor,
            source_rate // divisor,
        ).astype(np.float32)
    return np.clip(data, -1.0, 1.0).astype(np.float32)


def write_wav_mono(
    path: str | Path, samples: NDArray[np.float32], sample_rate: int = 16_000
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples, -1.0, 1.0) * 32_767.0).astype("<i2")
    with wave.open(str(destination), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def generate_demo_melody(sample_rate: int = 16_000) -> NDArray[np.float32]:
    notes = [220.0, 246.94, 261.63, 293.66, 329.63, 293.66, 261.63, 220.0]
    note_seconds = 0.24
    gap_seconds = 0.02
    chunks: list[NDArray[np.float32]] = []
    for frequency in notes:
        count = int(round(note_seconds * sample_rate))
        time = np.arange(count, dtype=np.float32) / sample_rate
        attack = min(int(0.02 * sample_rate), count // 4)
        envelope = np.ones(count, dtype=np.float32)
        envelope[:attack] = np.linspace(0.0, 1.0, attack, dtype=np.float32)
        envelope[-attack:] = np.linspace(1.0, 0.0, attack, dtype=np.float32)
        tone = (
            np.sin(2.0 * np.pi * frequency * time)
            + 0.35 * np.sin(2.0 * np.pi * 2.0 * frequency * time)
            + 0.15 * np.sin(2.0 * np.pi * 3.0 * frequency * time)
        )
        chunks.append((0.35 * tone * envelope).astype(np.float32))
        chunks.append(np.zeros(int(round(gap_seconds * sample_rate)), dtype=np.float32))
    return np.concatenate(chunks)

