from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfilt

from music_ci.analysis.filterbank import FilterBank22
from music_ci.core.models import PipelineResult


def synthesize_noise_band_vocoder(
    result: PipelineResult,
    filterbank: FilterBank22,
    frame_samples: int,
    random_seed: int,
) -> NDArray[np.float32]:
    """Create deterministic research listening output from frame envelopes."""
    sample_count = result.input_samples.size
    rng = np.random.default_rng(random_seed)
    noise = rng.standard_normal(sample_count).astype(np.float64)
    output = np.zeros(sample_count, dtype=np.float64)

    for channel, (low, high) in enumerate(zip(filterbank.low_edges, filterbank.high_edges)):
        sos = butter(2, [low, high], btype="bandpass", fs=result.sample_rate, output="sos")
        carrier = sosfilt(sos, noise)
        frame_envelope = result.electrodogram[channel]
        expanded = np.repeat(frame_envelope, frame_samples)[:sample_count]
        output += carrier * expanded

    peak = float(np.max(np.abs(output))) if output.size else 0.0
    if peak > 1e-12:
        output = 0.9 * output / peak
    return output.astype(np.float32)

