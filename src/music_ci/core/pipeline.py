from __future__ import annotations

from time import perf_counter_ns

import numpy as np
from numpy.typing import NDArray

from music_ci.analysis.filterbank import FilterBank22
from music_ci.analysis.pitch import CausalPitchTracker, harmonic_match, spectral_redundancy
from music_ci.config import AppConfig
from music_ci.core.models import AnalysisFrame, FrameResult, PipelineResult
from music_ci.strategies.ace import AceTopKPolicy
from music_ci.strategies.msa_select import MusicStructureAwarePolicy


class ResearchPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        fb = config.filterbank
        audio = config.audio
        pitch = config.pitch
        self.filterbank = FilterBank22(
            sample_rate=audio.sample_rate,
            num_channels=fb.num_channels,
            min_hz=fb.min_hz,
            max_hz=fb.max_hz,
            envelope_cutoff_hz=fb.envelope_cutoff_hz,
        )
        self.pitch_tracker = CausalPitchTracker(
            sample_rate=audio.sample_rate,
            context_ms=audio.context_ms,
            min_hz=pitch.min_hz,
            max_hz=pitch.max_hz,
        )
        selection = config.selection
        if selection.strategy == "ace":
            self.policy = AceTopKPolicy(selection.active_channels, fb.num_channels)
        else:
            self.policy = MusicStructureAwarePolicy(
                active_channels=selection.active_channels,
                num_channels=fb.num_channels,
                energy_weight=selection.energy_weight,
                harmonic_weight=selection.harmonic_weight,
                continuity_weight=selection.continuity_weight,
                redundancy_weight=selection.redundancy_weight,
                confidence_threshold=pitch.confidence_threshold,
            )
        self.redundancy = spectral_redundancy(self.filterbank.center_frequencies)

    def reset(self) -> None:
        self.filterbank.reset()
        self.pitch_tracker.reset()
        self.policy.reset()

    def process(self, samples: NDArray[np.float32]) -> PipelineResult:
        if samples.ndim != 1:
            raise ValueError("ResearchPipeline expects mono audio")
        self.reset()
        frame_size = self.config.audio.frame_samples
        num_frames = int(np.ceil(samples.size / frame_size))
        padded = np.pad(samples, (0, num_frames * frame_size - samples.size))
        frame_results: list[FrameResult] = []
        electrodogram = np.zeros(
            (self.config.filterbank.num_channels, num_frames), dtype=np.float32
        )

        for frame_index in range(num_frames):
            start = frame_index * frame_size
            frame = padded[start : start + frame_size].astype(np.float32, copy=False)
            started = perf_counter_ns()
            _, envelopes = self.filterbank.process_frame(frame)
            f0_hz, confidence, harmonicity = self.pitch_tracker.update(frame)
            log_energy = np.log1p(1_000.0 * np.maximum(envelopes, 0.0))
            peak = float(np.max(log_energy))
            normalized = (
                (log_energy / peak).astype(np.float32)
                if peak > 1e-12
                else np.zeros_like(envelopes)
            )
            analysis = AnalysisFrame(
                envelopes=envelopes,
                normalized_energy=normalized,
                f0_hz=f0_hz,
                f0_confidence=confidence,
                harmonicity=harmonicity,
                harmonic_match=harmonic_match(
                    self.filterbank.center_frequencies, f0_hz, confidence
                ),
                redundancy=self.redundancy,
            )
            selection = self.policy.select(analysis)
            electrode_frame = normalized * selection.selected_mask
            elapsed_ms = (perf_counter_ns() - started) / 1_000_000.0
            deadline_miss = elapsed_ms > self.config.runtime.frame_deadline_ms
            electrodogram[:, frame_index] = electrode_frame
            frame_results.append(
                FrameResult(
                    frame_index=frame_index,
                    sample_index=start,
                    analysis=analysis,
                    selection=selection,
                    electrodogram=electrode_frame,
                    runtime_ms=elapsed_ms,
                    deadline_miss=deadline_miss,
                )
            )

        return PipelineResult(
            sample_rate=self.config.audio.sample_rate,
            input_samples=samples.copy(),
            frame_results=frame_results,
            electrodogram=electrodogram,
        )

