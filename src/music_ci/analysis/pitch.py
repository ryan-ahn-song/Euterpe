from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class CausalPitchTracker:
    """Small deterministic autocorrelation tracker using past-only context."""

    def __init__(
        self,
        sample_rate: int,
        context_ms: int = 48,
        min_hz: float = 80.0,
        max_hz: float = 1_000.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.context_samples = int(round(sample_rate * context_ms / 1_000))
        self.min_hz = min_hz
        self.max_hz = max_hz
        self._buffer = np.zeros(0, dtype=np.float32)

    def reset(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)

    def update(self, samples: NDArray[np.float32]) -> tuple[float, float, float]:
        self._buffer = np.concatenate((self._buffer, samples))[-self.context_samples :]
        minimum_needed = max(int(2 * self.sample_rate / self.min_hz), samples.size)
        if self._buffer.size < minimum_needed:
            return 0.0, 0.0, 0.0

        signal = self._buffer.astype(np.float64)
        signal -= signal.mean()
        rms = float(np.sqrt(np.mean(signal * signal)))
        if rms < 1e-5:
            return 0.0, 0.0, 0.0

        signal *= np.hanning(signal.size)
        correlation = np.correlate(signal, signal, mode="full")[signal.size - 1 :]
        if correlation[0] <= 1e-12:
            return 0.0, 0.0, 0.0
        correlation /= correlation[0]

        min_lag = max(1, int(self.sample_rate / self.max_hz))
        max_lag = min(correlation.size - 1, int(self.sample_rate / self.min_hz))
        search = correlation[min_lag : max_lag + 1]
        lag = int(np.argmax(search)) + min_lag
        confidence = float(np.clip(correlation[lag], 0.0, 1.0))

        refined_lag = float(lag)
        if 1 <= lag < correlation.size - 1:
            left, middle, right = correlation[lag - 1 : lag + 2]
            denominator = left - 2.0 * middle + right
            if abs(denominator) > 1e-12:
                refined_lag += float(0.5 * (left - right) / denominator)

        f0_hz = float(self.sample_rate / refined_lag)
        harmonicity = confidence
        return f0_hz, confidence, harmonicity


def harmonic_match(
    center_frequencies: NDArray[np.float32], f0_hz: float, confidence: float
) -> NDArray[np.float32]:
    if f0_hz <= 0.0 or confidence <= 0.0:
        return np.zeros_like(center_frequencies, dtype=np.float32)
    harmonic_number = np.maximum(1.0, np.rint(center_frequencies / f0_hz))
    nearest = harmonic_number * f0_hz
    cents = 1_200.0 * np.log2(np.maximum(center_frequencies, 1.0) / np.maximum(nearest, 1.0))
    match = np.exp(-0.5 * (cents / 75.0) ** 2) * confidence
    return match.astype(np.float32)


def spectral_redundancy(center_frequencies: NDArray[np.float32]) -> NDArray[np.float32]:
    log_frequency = np.log2(center_frequencies.astype(np.float64))
    distance = np.abs(log_frequency[:, None] - log_frequency[None, :])
    redundancy = np.exp(-distance / 0.35)
    np.fill_diagonal(redundancy, 0.0)
    return redundancy.astype(np.float32)

