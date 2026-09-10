from functools import lru_cache
from pathlib import Path

from .preprocess import tokenize
from .sentiment_rules import SentimentResult, heuristic_predict_sentiment


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "models" / "checkpoints" / "sentiment_bilstm.pt"
DEFAULT_LABELS = ["negative", "neutral", "positive"]


class SentimentPredictor:
    def __init__(self, checkpoint_path: str | Path = DEFAULT_CHECKPOINT):
        self.checkpoint_path = Path(checkpoint_path)
        self.model = None
        self.vocab = None
        self.labels = DEFAULT_LABELS
        self.model_name = "Demo Rule-based Fallback"
        self.load_error = ""
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        if not self.checkpoint_path.exists():
            return
        try:
            import torch

            from models.sentiment_lstm import SentimentBiLSTM

            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            vocab = checkpoint["vocab"]
            config = checkpoint.get("config", {})
            labels = checkpoint.get("labels", DEFAULT_LABELS)
            model = SentimentBiLSTM(
                len(vocab),
                embed_dim=config.get("embed_dim", 128),
                hidden_dim=config.get("hidden_dim", 128),
                num_classes=config.get("num_classes", len(labels)),
            )
            model.load_state_dict(checkpoint["model"])
            model.eval()
        except Exception as exc:
            self.load_error = str(exc)
            return
        self.torch = torch
        self.vocab = vocab
        self.labels = labels
        self.model = model
        self.model_name = "Sentiment BiLSTM checkpoint"
        self.max_len = checkpoint.get("config", {}).get("max_len", 80)

    def predict(self, text: str) -> SentimentResult:
        if self.model is None or self.vocab is None:
            return heuristic_predict_sentiment(text)

        with self.torch.no_grad():
            probabilities = self.torch.softmax(self.model(self._encode(text)), dim=1)[0]
        label_index = int(probabilities.argmax().item())
        label = self.labels[label_index]
        negative = float(probabilities[self.labels.index("negative")])
        positive = float(probabilities[self.labels.index("positive")])
        return SentimentResult(
            label=label,
            score=round(positive - negative, 4),
            model_name=self.model_name,
        )

    def _encode(self, text: str):
        max_len = getattr(self, "max_len", 80)
        ids = [self.vocab.get(token, 1) for token in tokenize(text)[:max_len]]
        ids += [0] * (max_len - len(ids))
        return self.torch.tensor([ids], dtype=self.torch.long)


@lru_cache(maxsize=1)
def get_sentiment_predictor() -> SentimentPredictor:
    return SentimentPredictor()


def predict_sentiment(text: str) -> SentimentResult:
    return get_sentiment_predictor().predict(text)
