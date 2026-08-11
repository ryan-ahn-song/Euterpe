from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfilt


class FilterBank22:
    """Stateful causal analysis filterbank for the research prototype.

    The band layout is logarithmic and deliberately isolated behind this class so
    Nucleus Toolbox-matched coefficients can replace it without changing the rest
    of the pipeline.
    """

    def __init__(
        self,
        sample_rate: int,
        num_channels: int = 22,
        min_hz: float = 150.0,
        max_hz: float = 7_500.0,
        envelope_cutoff_hz: float = 400.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        edges = np.geomspace(min_hz, max_hz, num_channels + 1)
        self.low_edges = edges[:-1]
        self.high_edges = edges[1:]
        self.center_frequencies = np.sqrt(self.low_edges * self.high_edges).astype(np.float32)
        self._sos = [
            butter(2, [lo, hi], btype="bandpass", fs=sample_rate, output="sos")
            for lo, hi in zip(self.low_edges, self.high_edges)
        ]
        self._zi = [np.zeros((sos.shape[0], 2), dtype=np.float64) for sos in self._sos]
        self._envelope_state = np.zeros(num_channels, dtype=np.float64)
        self._envelope_alpha = float(np.exp(-2.0 * np.pi * envelope_cutoff_hz / sample_rate))

    def reset(self) -> None:
        self._zi = [np.zeros_like(zi) for zi in self._zi]
        self._envelope_state.fill(0.0)

    def process_frame(
        self, samples: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if samples.ndim != 1:
            raise ValueError("FilterBank22 expects mono audio")
        channel_audio = np.empty((self.num_channels, samples.size), dtype=np.float32)
        envelopes = np.empty(self.num_channels, dtype=np.float32)
        alpha = self._envelope_alpha

        for channel, sos in enumerate(self._sos):
            filtered, self._zi[channel] = sosfilt(sos, samples, zi=self._zi[channel])
            channel_audio[channel] = filtered.astype(np.float32)
            state = self._envelope_state[channel]
            for value in np.abs(filtered):
                state = alpha * state + (1.0 - alpha) * value
            self._envelope_state[channel] = state
            envelopes[channel] = state

        return channel_audio, envelopes

