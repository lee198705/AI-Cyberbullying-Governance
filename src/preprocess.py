import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


CHINESE_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_csv(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = []
    for chunk in CHINESE_RE.findall(text):
        if re.fullmatch(r"[\u4e00-\u9fff]+", chunk):
            if len(chunk) == 1:
                tokens.append(chunk)
            else:
                tokens.extend(chunk[i : i + 2] for i in range(len(chunk) - 1))
        else:
            tokens.append(chunk)
    return [token for token in tokens if token.strip()]


def normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def tfidf_vectors(rows: list[dict], text_field: str) -> tuple[list[Counter], list[str]]:
    docs = [tokenize(row[text_field]) for row in rows]
    df = Counter()
    for doc in docs:
        df.update(set(doc))
    vocab = sorted(df)
    total = max(len(docs), 1)
    vectors = []
    for doc in docs:
        counts = Counter(doc)
        length = max(sum(counts.values()), 1)
        vector = Counter()
        for token, count in counts.items():
            idf = math.log((1 + total) / (1 + df[token])) + 1
            vector[token] = (count / length) * idf
        vectors.append(vector)
    return vectors, vocab


def cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    shared = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in shared)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def weighted_user_profiles(
    articles: list[dict], interactions: list[dict], action_weights: dict[str, float] | None = None
) -> dict[str, Counter]:
    action_weights = action_weights or {"view": 1.0, "like": 2.0, "share": 3.0}
    article_texts = []
    for row in articles:
        article_texts.append(
            {
                **row,
                "feature_text": " ".join(
                    [row["category"], row["tags"], row["tags"], row["title"], row["content"]]
                ),
            }
        )
    vectors, _ = tfidf_vectors(article_texts, "feature_text")
    article_vectors = {row["article_id"]: vectors[idx] for idx, row in enumerate(article_texts)}
    profiles = defaultdict(Counter)
    totals = defaultdict(float)
    for item in interactions:
        vec = article_vectors.get(item["article_id"])
        if not vec:
            continue
        weight = action_weights.get(item["action"], 1.0)
        totals[item["user_id"]] += weight
        for token, value in vec.items():
            profiles[item["user_id"]][token] += weight * value
    for user_id, total in totals.items():
        if total:
            for token in list(profiles[user_id]):
                profiles[user_id][token] /= total
    return dict(profiles)
