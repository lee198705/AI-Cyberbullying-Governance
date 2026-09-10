import math
from dataclasses import dataclass

from .preprocess import tokenize
from .rationality import RATIONAL_TERMS


TOXIC_TERMS = {
    "有病": 0.34,
    "垃圾": 0.32,
    "闭嘴": 0.28,
    "丢人": 0.25,
    "脑子": 0.20,
    "污染": 0.20,
    "滚": 0.35,
    "傻": 0.30,
    "死": 0.26,
}


@dataclass
class ToxicityResult:
    probability: float
    level: str
    action: str
    matched_terms: list[str]
    model_name: str = "Demo Rule-based Fallback"


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def risk_level(probability: float) -> str:
    if probability >= 0.80:
        return "高风险"
    if probability >= 0.60:
        return "中高风险"
    if probability >= 0.35:
        return "需提示"
    return "正常"


def moderation_action(probability: float) -> str:
    if probability >= 0.80:
        return "阻断并转人工审核"
    if probability >= 0.60:
        return "降权 / 限流"
    if probability >= 0.35:
        return "发布前风险提示"
    return "正常展示"


def heuristic_predict_toxicity(text: str) -> ToxicityResult:
    compact = "".join(tokenize(text))
    toxic_hits = [term for term in TOXIC_TERMS if term in text or term in compact]
    rational_hits = [term for term in RATIONAL_TERMS if term in text or term in compact]
    score = -1.9
    score += sum(TOXIC_TERMS[t] for t in toxic_hits) * 6.0
    score -= sum(RATIONAL_TERMS[t] for t in rational_hits) * 2.3
    score += min(text.count("!") + text.count("！"), 3) * 0.18
    probability = round(_sigmoid(score), 4)
    return ToxicityResult(
        probability=probability,
        level=risk_level(probability),
        action=moderation_action(probability),
        matched_terms=toxic_hits,
    )
