from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class AnalysisFrame:
    envelopes: FloatArray
    normalized_energy: FloatArray
    f0_hz: float
    f0_confidence: float
    harmonicity: float
    harmonic_match: FloatArray
    redundancy: FloatArray


@dataclass(frozen=True)
class SelectionFrame:
    selected_mask: BoolArray
    selected_indices: IntArray
    scores: FloatArray
    energy_scores: FloatArray
    harmonic_scores: FloatArray
    continuity_scores: FloatArray
    redundancy_penalties: FloatArray


@dataclass(frozen=True)
class FrameResult:
    frame_index: int
    sample_index: int
    analysis: AnalysisFrame
    selection: SelectionFrame
    electrodogram: FloatArray
    runtime_ms: float
    deadline_miss: bool


@dataclass(frozen=True)
class PipelineResult:
    sample_rate: int
    input_samples: FloatArray
    frame_results: list[FrameResult]
    electrodogram: FloatArray

