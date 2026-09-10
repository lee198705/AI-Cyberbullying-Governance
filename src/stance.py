from dataclasses import dataclass


SUPPORT_TERMS = {
    "支持": 0.36,
    "赞同": 0.34,
    "认可": 0.30,
    "有必要": 0.28,
    "应该": 0.22,
    "同意": 0.26,
    "需要": 0.20,
}

OPPOSE_TERMS = {
    "反对": 0.36,
    "不同意": 0.34,
    "不赞成": 0.32,
    "没必要": 0.28,
    "不应该": 0.26,
}

NEGATED_SUPPORT_TERMS = {
    "并不支持": 0.36,
    "不支持": 0.36,
    "并不赞同": 0.34,
    "不赞同": 0.34,
    "并不认可": 0.30,
    "不认可": 0.30,
}

NEGATED_OPPOSE_TERMS = {
    "并不反对": 0.30,
    "不反对": 0.30,
}

# These phrases contain stance keywords but do not express support or opposition
# by themselves. Reserve their spans before matching shorter terms.
NEUTRAL_MASK_TERMS = {"不同意见"}


@dataclass
class StanceResult:
    score: float
    label: str
    confidence: float
    matched_terms: list[str]


def stance_label(score: float) -> str:
    if score <= -0.25:
        return "A1"
    if score >= 0.25:
        return "A2"
    return "B"


def _match_terms(text: str) -> list[tuple[int, str, int, float]]:
    candidates = [
        (term, 0, 0.0) for term in NEUTRAL_MASK_TERMS
    ] + [
        (term, 1, weight) for term, weight in SUPPORT_TERMS.items()
    ] + [
        (term, -1, weight) for term, weight in OPPOSE_TERMS.items()
    ] + [
        (term, -1, weight) for term, weight in NEGATED_SUPPORT_TERMS.items()
    ] + [
        (term, 1, weight) for term, weight in NEGATED_OPPOSE_TERMS.items()
    ]
    candidates.sort(key=lambda item: len(item[0]), reverse=True)
    occupied = [False] * len(text)
    matches = []
    for term, direction, weight in candidates:
        start = 0
        while (index := text.find(term, start)) != -1:
            end = index + len(term)
            if not any(occupied[index:end]):
                occupied[index:end] = [True] * len(term)
                if direction:
                    matches.append((index, term, direction, weight))
            start = index + 1
    return sorted(matches, key=lambda item: item[0])


def predict_stance(text: str, topic: str | None = None) -> StanceResult:
    matches = _match_terms(text)
    raw = sum(direction * weight for _, _, direction, weight in matches)
    score = max(-1.0, min(1.0, raw))
    evidence = min(1.0, abs(raw) * 1.8)
    topic_bonus = 0.08 if topic and topic in text else 0.0
    confidence = max(0.0, min(1.0, evidence + topic_bonus))
    if confidence < 0.18:
        score = 0.0
    return StanceResult(
        score=round(score, 4),
        label=stance_label(score),
        confidence=round(confidence, 4),
        matched_terms=[term for _, term, _, _ in matches],
    )


def aggregate_stances(results: list[StanceResult]) -> dict | None:
    expressive = [result for result in results if result.confidence >= 0.18]
    if not expressive:
        return None
    score = sum(result.score * result.confidence for result in expressive) / sum(
        result.confidence for result in expressive
    )
    directions = [1 if result.score > 0 else -1 if result.score < 0 else 0 for result in expressive]
    major = max(directions.count(1), directions.count(-1), directions.count(0))
    consistency = major / len(directions)
    sample_factor = min(1.0, len(expressive) / 5)
    evidence = sum(result.confidence for result in expressive) / len(expressive)
    confidence = min(
        1.0,
        0.15
        + 0.40 * sample_factor
        + 0.25 * consistency * sample_factor
        + 0.20 * evidence,
    )
    return {
        "score": round(score, 4),
        "label": stance_label(score),
        "confidence": round(confidence, 4),
        "samples": len(expressive),
    }
