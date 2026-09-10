from .comment_ranking import rank_comments
from .toxicity_predictor import predict_toxicity


def monitor_comments(comments: list[dict], profiles: dict[str, dict] | None = None) -> dict:
    assessed = []
    for comment in comments:
        result = predict_toxicity(comment["text"])
        assessed.append(
            {
                **comment,
                "toxicity": result.probability,
                "risk_level": result.level,
                "moderation_action": result.action,
                "matched_terms": "、".join(result.matched_terms) if result.matched_terms else "-",
            }
        )
    high_risk = [row for row in assessed if row["toxicity"] >= 0.60]
    blocked = [row for row in assessed if row["toxicity"] >= 0.80]
    return {
        "comments": assessed,
        "ranked_comments": rank_comments(comments, profiles),
        "total": len(comments),
        "risk_count": len(high_risk),
        "blocked_count": len(blocked),
        "avg_toxicity": round(sum(row["toxicity"] for row in assessed) / max(len(assessed), 1), 4),
    }
