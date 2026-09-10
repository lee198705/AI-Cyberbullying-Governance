RATIONAL_TERMS = {
    "数据": 0.12,
    "来源": 0.10,
    "建议": 0.12,
    "讨论": 0.12,
    "机制": 0.10,
    "观点": 0.08,
    "不同意": 0.08,
    "报告": 0.10,
    "透明": 0.12,
    "校准": 0.10,
    "样本": 0.10,
    "证据": 0.12,
    "事实": 0.10,
}


def rationality_score(text: str) -> float:
    hits = sum(1 for term in RATIONAL_TERMS if term in text)
    punctuation_penalty = min(text.count("!") + text.count("！"), 3) * 0.04
    return round(max(0.0, min(1.0, 0.35 + hits * 0.11 - punctuation_penalty)), 4)
