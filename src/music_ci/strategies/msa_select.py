from __future__ import annotations

import numpy as np

from music_ci.core.models import AnalysisFrame, SelectionFrame
from music_ci.strategies.base import ChannelSelectionPolicy


class MusicStructureAwarePolicy(ChannelSelectionPolicy):
    def __init__(
        self,
        active_channels: int,
        num_channels: int = 22,
        energy_weight: float = 1.0,
        harmonic_weight: float = 0.6,
        continuity_weight: float = 0.2,
        redundancy_weight: float = 0.3,
        confidence_threshold: float = 0.35,
    ) -> None:
        super().__init__(active_channels, num_channels)
        self.energy_weight = energy_weight
        self.harmonic_weight = harmonic_weight
        self.continuity_weight = continuity_weight
        self.redundancy_weight = redundancy_weight
        self.confidence_threshold = confidence_threshold

    def select(self, analysis: AnalysisFrame) -> SelectionFrame:
        confidence_gate = float(
            np.clip(
                (analysis.f0_confidence - self.confidence_threshold) /
                max(1e-6, 1.0 - self.confidence_threshold),
                0.0,
                1.0,
            )
        )
        energy = self.energy_weight * analysis.normalized_energy
        harmonic = self.harmonic_weight * confidence_gate * analysis.harmonic_match
        continuity = self.continuity_weight * self.previous_mask.astype(np.float32)
        base_scores = energy + harmonic + continuity

        selected: list[int] = []
        penalties = np.zeros(self.num_channels, dtype=np.float32)
        working_scores = base_scores.copy()
        for _ in range(self.active_channels):
            if selected:
                penalties = np.max(analysis.redundancy[:, selected], axis=1)
                working_scores = base_scores - self.redundancy_weight * penalties
            working_scores[selected] = -np.inf
            selected.append(int(np.argmax(working_scores)))

        indices = np.array(sorted(selected), dtype=np.int64)
        mask = self._make_mask(indices)
        final_scores = base_scores - self.redundancy_weight * penalties
        return SelectionFrame(
            selected_mask=mask,
            selected_indices=indices,
            scores=final_scores.astype(np.float32),
            energy_scores=energy.astype(np.float32),
            harmonic_scores=harmonic.astype(np.float32),
            continuity_scores=continuity.astype(np.float32),
            redundancy_penalties=(self.redundancy_weight * penalties).astype(np.float32),
        )

