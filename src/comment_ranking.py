import math

from .sentiment_predictor import predict_sentiment
from .toxicity_predictor import predict_toxicity
from .rationality import rationality_score


def quality_score(text: str, replies: int = 0) -> float:
    length_score = min(len(text) / 90, 1.0)
    reply_score = min(math.log1p(replies) / 3, 1.0)
    return round(0.65 * length_score + 0.35 * reply_score, 4)


def rank_comment(comment: dict, profile: dict | None = None) -> dict:
    likes = int(comment.get("likes", 0))
    replies = int(comment.get("replies", 0))
    text = comment["text"]
    toxicity = predict_toxicity(text)
    sentiment = predict_sentiment(text)
    rationality = rationality_score(text)
    quality = quality_score(text, replies)
    neutrality = 1 - min(abs(sentiment.score), 1)
    like_score = min(math.log1p(likes) / math.log(120), 1.0)
    profile_bonus = 0.05 if profile and profile.get("global_user_type") == "理性活跃用户" else 0.0
    final = (
        0.35 * like_score
        + 0.30 * rationality
        + 0.20 * neutrality
        + 0.15 * quality
        - 0.50 * toxicity.probability
        + profile_bonus
    )
    return {
        **comment,
        "toxicity": toxicity.probability,
        "risk_level": toxicity.level,
        "moderation_action": toxicity.action,
        "sentiment": sentiment.label,
        "rationality_score": round(rationality, 4),
        "quality_score": quality,
        "profile_bonus": round(profile_bonus, 4),
        "rank_score": round(max(final, 0.0), 4),
    }


def rank_comments(comments: list[dict], profiles: dict[str, dict] | None = None) -> list[dict]:
    profiles = profiles or {}
    ranked = [rank_comment(c, profiles.get(c["user_id"])) for c in comments]
    return sorted(ranked, key=lambda row: row["rank_score"], reverse=True)
