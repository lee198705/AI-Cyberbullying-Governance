from .preprocess import cosine, tokenize, weighted_user_profiles
from .toxicity_predictor import predict_toxicity


def _article_vector(article: dict) -> dict:
    text = " ".join([article["category"], article["tags"], article["tags"], article["title"], article["content"]])
    counts = {}
    for token in tokenize(text):
        counts[token] = counts.get(token, 0.0) + 1.0
    return counts


def _diversity_score(article: dict, seen_categories: set[str]) -> float:
    return 0.35 if article["category"] in seen_categories else 1.0


def recommend_articles(
    user_id: str,
    articles: list[dict],
    interactions: list[dict],
    comments: list[dict] | None = None,
    top_n: int = 5,
) -> list[dict]:
    comments = comments or []
    profiles = weighted_user_profiles(articles, interactions)
    user_profile = profiles.get(user_id, {})
    read_ids = {row["article_id"] for row in interactions if row["user_id"] == user_id}
    seen_categories = {row["category"] for row in articles if row["article_id"] in read_ids}
    article_vectors = {row["article_id"]: _article_vector(row) for row in articles}
    discussion_risks = {row["article_id"]: [] for row in articles}
    for comment in comments:
        if comment["article_id"] in discussion_risks:
            discussion_risks[comment["article_id"]].append(predict_toxicity(comment["text"]).probability)

    results = []
    for article in articles:
        if article["article_id"] in read_ids:
            continue
        interest = cosine(user_profile, article_vectors[article["article_id"]]) if user_profile else 0.35
        quality = float(article.get("quality_score", 0.75))
        diversity = _diversity_score(article, seen_categories)
        content_text = " ".join([article["title"], article["content"]])
        content_risk = predict_toxicity(content_text).probability
        discussion_values = discussion_risks.get(article["article_id"], [])
        discussion_risk = sum(discussion_values) / len(discussion_values) if discussion_values else 0.05
        final = 0.5 * interest + 0.3 * quality + 0.2 * diversity - 0.4 * content_risk
        results.append(
            {
                **article,
                "interest_score": round(interest, 4),
                "diversity_score": round(diversity, 4),
                "content_risk": round(content_risk, 4),
                "discussion_risk": round(discussion_risk, 4),
                "final_score": round(max(final, 0.0), 4),
                "reason": _reason(interest, diversity, content_risk, discussion_risk),
            }
        )
    return sorted(results, key=lambda row: row["final_score"], reverse=True)[:top_n]


def _reason(interest: float, diversity: float, content_risk: float, discussion_risk: float) -> str:
    parts = []
    parts.append("兴趣匹配" if interest >= 0.45 else "冷启动探索")
    if diversity >= 0.9:
        parts.append("观点多样性")
    if content_risk <= 0.2:
        parts.append("低风险内容")
    if discussion_risk >= 0.5:
        parts.append("评论区需加强审核")
    return " + ".join(parts)
