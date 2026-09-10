"""Backward-compatible sentiment API.

New code can import the runtime predictor from ``sentiment_predictor`` or the
deterministic fallback from ``sentiment_rules`` directly.
"""

from .sentiment_predictor import SentimentPredictor, get_sentiment_predictor, predict_sentiment
from .sentiment_rules import SentimentResult, heuristic_predict_sentiment

__all__ = [
    "SentimentPredictor",
    "SentimentResult",
    "get_sentiment_predictor",
    "heuristic_predict_sentiment",
    "predict_sentiment",
]
