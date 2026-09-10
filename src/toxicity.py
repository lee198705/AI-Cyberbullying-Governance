"""Backward-compatible toxicity API.

New code should import from `src.toxicity_predictor` for runtime model loading
or from `src.toxicity_rules` for the deterministic fallback.
"""

from .toxicity_predictor import get_predictor, predict_toxicity
from .toxicity_rules import (
    TOXIC_TERMS,
    ToxicityResult,
    heuristic_predict_toxicity,
    moderation_action,
    risk_level,
)


__all__ = [
    "TOXIC_TERMS",
    "ToxicityResult",
    "get_predictor",
    "heuristic_predict_toxicity",
    "moderation_action",
    "predict_toxicity",
    "risk_level",
]
