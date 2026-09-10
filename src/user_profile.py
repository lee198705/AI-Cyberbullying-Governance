from collections import defaultdict

from .rationality import rationality_score
from .sentiment_predictor import predict_sentiment
from .stance import aggregate_stances, predict_stance
from .toxicity_predictor import predict_toxicity


def _interest_terms(article: dict) -> list[str]:
    terms = [article.get("category", "").strip()]
    terms.extend(tag.strip() for tag in article.get("tags", "").split(",") if tag.strip())
    return [term for term in terms if term]


def _global_user_type(toxicity: float, rationality: float, activity: float) -> str:
    if toxicity >= 0.55:
        return "高风险/高情绪用户"
    if rationality >= 0.70 and activity >= 0.35:
        return "理性活跃用户"
    return "中立稳定用户"


def build_user_profiles(
    users: list[dict], comments: list[dict], interactions: list[dict], articles: list[dict]
) -> dict[str, dict]:
    article_index = {row["article_id"]: row for row in articles}
    user_comments = defaultdict(list)
    for row in comments:
        user_comments[row["user_id"]].append(row)

    action_count = defaultdict(int)
    topic_counts = defaultdict(lambda: defaultdict(float))
    stance_values = defaultdict(lambda: defaultdict(list))

    for row in interactions:
        user_id = row["user_id"]
        action_count[user_id] += 1
        article = article_index.get(row["article_id"])
        if not article:
            continue
        weight = {"view": 1.0, "like": 2.0, "share": 3.0}.get(row["action"], 1.0)
        for topic in _interest_terms(article):
            topic_counts[user_id][topic] += weight

    for row in comments:
        article = article_index.get(row["article_id"])
        if not article:
            continue
        for topic in _interest_terms(article):
            topic_counts[row["user_id"]][topic] += 1.2
        main_topic = article.get("main_topic", "").strip() or article.get("category", "").strip()
        stance = predict_stance(row["text"], main_topic)
        if main_topic and stance.confidence >= 0.18:
            stance_values[row["user_id"]][main_topic].append(stance)

    profiles = {}
    for user in users:
        user_id = user["user_id"]
        rows = user_comments[user_id]
        toxicity_scores = [predict_toxicity(row["text"]).probability for row in rows]
        sentiment_scores = [predict_sentiment(row["text"]).score for row in rows]
        rationality_scores = [rationality_score(row["text"]) for row in rows]
        toxicity = sum(toxicity_scores) / len(toxicity_scores) if toxicity_scores else 0.0
        sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
        rationality = sum(rationality_scores) / len(rationality_scores) if rationality_scores else 0.35
        activity = min((len(rows) + action_count[user_id]) / 8, 1.0)
        ranked_topics = sorted(topic_counts[user_id].items(), key=lambda item: item[1], reverse=True)
        topic_stances = {}
        for topic, values in stance_values[user_id].items():
            aggregated = aggregate_stances(values)
            if aggregated:
                topic_stances[topic] = aggregated
        topic_preferences = [topic for topic, _ in ranked_topics[:3]]
        fallback_topics = [item.strip() for item in user.get("initial_topics", "").split(",") if item.strip()]
        profiles[user_id] = {
            "user_id": user_id,
            "name": user.get("name", user_id),
            "toxicity_score": round(toxicity, 4),
            "sentiment_score": round(sentiment, 4),
            "activity_score": round(activity, 4),
            "rationality_score": round(rationality, 4),
            "topic_preferences": topic_preferences or fallback_topics,
            "topic_stances": topic_stances,
            "global_user_type": _global_user_type(toxicity, rationality, activity),
        }
    return profiles
