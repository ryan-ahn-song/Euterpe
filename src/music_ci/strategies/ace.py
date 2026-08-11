from __future__ import annotations

import numpy as np

from music_ci.core.models import AnalysisFrame, SelectionFrame
from music_ci.strategies.base import ChannelSelectionPolicy


class AceTopKPolicy(ChannelSelectionPolicy):
    """Energy-only n-of-m baseline.

    This is an ACE-shaped research baseline, not yet a certified reproduction of
    a manufacturer's clinical strategy.
    """

    def select(self, analysis: AnalysisFrame) -> SelectionFrame:
        indices = np.argsort(analysis.normalized_energy, kind="stable")[-self.active_channels :]
        indices = np.sort(indices).astype(np.int64)
        mask = self._make_mask(indices)
        zeros = np.zeros(self.num_channels, dtype=np.float32)
        return SelectionFrame(
            selected_mask=mask,
            selected_indices=indices,
            scores=analysis.normalized_energy.copy(),
            energy_scores=analysis.normalized_energy.copy(),
            harmonic_scores=zeros.copy(),
            continuity_scores=zeros.copy(),
            redundancy_penalties=zeros.copy(),
        )

