"""Non-clinical, real-time music-aware hearing-assist prototype."""

from .cape import CAPEEffectModel, CAPEPolicy, ListenerProfile
from .config import HearingAssistConfig, load_hearing_profile
from .dsp import HearingAssistProcessor

__all__ = [
    "CAPEEffectModel",
    "CAPEPolicy",
    "HearingAssistConfig",
    "HearingAssistProcessor",
    "ListenerProfile",
    "load_hearing_profile",
]
__version__ = "0.3.0"
