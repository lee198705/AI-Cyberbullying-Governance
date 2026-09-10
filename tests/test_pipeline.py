from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.moderation import monitor_comments
from src.preprocess import read_csv
from src.recommender import recommend_articles
from src.user_profile import build_user_profiles
from src.preprocess import tokenize
from src.stance import predict_stance
from src.sentiment import predict_sentiment
from src.sentiment_predictor import SentimentPredictor


class PipelineTest(unittest.TestCase):
    def test_moderation_pipeline_flags_risky_comments(self):
        comments = read_csv(ROOT / "data" / "comments.csv")
        users = read_csv(ROOT / "data" / "users.csv")
        articles = read_csv(ROOT / "data" / "articles.csv")
        interactions = read_csv(ROOT / "data" / "interactions.csv")
        profiles = build_user_profiles(users, comments, interactions, articles)
        report = monitor_comments(comments, profiles)
        self.assertEqual(report["total"], len(comments))
        self.assertGreaterEqual(report["risk_count"], 1)
        self.assertGreaterEqual(report["blocked_count"], 1)

    def test_profiles_and_recommendations_are_generated(self):
        users = read_csv(ROOT / "data" / "users.csv")
        comments = read_csv(ROOT / "data" / "comments.csv")
        articles = read_csv(ROOT / "data" / "articles.csv")
        interactions = read_csv(ROOT / "data" / "interactions.csv")
        profiles = build_user_profiles(users, comments, interactions, articles)
        recs = recommend_articles("U001", articles, interactions, comments, top_n=3)
        self.assertTrue(profiles["U001"]["global_user_type"])
        self.assertIn("topic_stances", profiles["U001"])
        self.assertIn("samples", next(iter(profiles["U004"]["topic_stances"].values())))
        self.assertEqual(set(profiles["U004"]["topic_stances"]), {"公共讨论治理"})
        self.assertLess(profiles["U004"]["topic_stances"]["公共讨论治理"]["confidence"], 0.5)
        self.assertEqual(len(recs), 3)
        self.assertGreaterEqual(recs[0]["final_score"], recs[-1]["final_score"])
        self.assertIn("content_risk", recs[0])
        self.assertIn("discussion_risk", recs[0])

    def test_single_chinese_character_tokenizes_once(self):
        self.assertEqual(tokenize("好"), ["好"])

    def test_stance_is_not_sentiment(self):
        text = "支持加强平台审核，辱骂和造谣本来就不该被推荐。"
        self.assertLess(predict_sentiment(text).score, 0)
        stance = predict_stance(text, "平台治理")
        self.assertEqual(stance.label, "A2")
        self.assertGreater(stance.confidence, 0)

    def test_negative_phrase_not_match_positive_substring(self):
        result = predict_stance("我不同意这个观点")
        self.assertEqual(result.label, "A1")
        self.assertEqual(result.matched_terms, ["不同意"])

        result = predict_stance("这个方案不应该上线")
        self.assertEqual(result.label, "A1")
        self.assertEqual(result.matched_terms, ["不应该"])

        result = predict_stance("我们需要保护不同意见")
        self.assertNotIn("同意", result.matched_terms)

    def test_negated_stance_phrases_reverse_direction(self):
        result = predict_stance("我不支持这个方案")
        self.assertEqual(result.label, "A1")
        self.assertEqual(result.matched_terms, ["不支持"])

        result = predict_stance("我并不赞同这个观点")
        self.assertEqual(result.label, "A1")
        self.assertEqual(result.matched_terms, ["并不赞同"])

        result = predict_stance("我不反对这个方案")
        self.assertEqual(result.label, "A2")
        self.assertEqual(result.matched_terms, ["不反对"])

    def test_sentiment_predictor_falls_back_without_checkpoint(self):
        predictor = SentimentPredictor(ROOT / "models" / "checkpoints" / "missing.pt")
        result = predictor.predict("这篇报告很重要，数据也很清楚")
        self.assertEqual(predictor.model_name, "Demo Rule-based Fallback")
        self.assertEqual(result.label, "positive")


if __name__ == "__main__":
    unittest.main()
