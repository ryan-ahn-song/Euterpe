from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from music_ci.core.models import AnalysisFrame, SelectionFrame


class ChannelSelectionPolicy(ABC):
    def __init__(self, active_channels: int, num_channels: int = 22) -> None:
        self.active_channels = active_channels
        self.num_channels = num_channels
        self.previous_mask = np.zeros(num_channels, dtype=np.bool_)

    def reset(self) -> None:
        self.previous_mask.fill(False)

    @abstractmethod
    def select(self, analysis: AnalysisFrame) -> SelectionFrame:
        raise NotImplementedError

    def _make_mask(self, indices: NDArray[np.int64]) -> NDArray[np.bool_]:
        mask = np.zeros(self.num_channels, dtype=np.bool_)
        mask[indices] = True
        self.previous_mask = mask.copy()
        return mask

