from dataclasses import dataclass


POSITIVE_TERMS = {"支持", "赞同", "重要", "清楚", "合理", "希望", "完整", "透明", "保护"}
NEGATIVE_TERMS = {"不同意", "担心", "风险", "攻击", "辱骂", "造谣", "有病", "垃圾", "闭嘴", "丢人"}


@dataclass
class SentimentResult:
    label: str
    score: float
    model_name: str = "Demo Rule-based Fallback"


def heuristic_predict_sentiment(text: str) -> SentimentResult:
    pos = sum(1 for term in POSITIVE_TERMS if term in text)
    neg = sum(1 for term in NEGATIVE_TERMS if term in text)
    raw = pos - neg
    if raw >= 1:
        return SentimentResult("positive", min(1.0, 0.55 + raw * 0.15))
    if raw <= -1:
        return SentimentResult("negative", max(-1.0, -0.55 + raw * 0.15))
    return SentimentResult("neutral", 0.0)
