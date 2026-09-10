from functools import lru_cache
from pathlib import Path

from .preprocess import tokenize
from .toxicity_rules import ToxicityResult, heuristic_predict_toxicity, moderation_action, risk_level


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "models" / "checkpoints" / "toxicity_bilstm.pt"


class ToxicityPredictor:
    def __init__(self, checkpoint_path: str | Path = DEFAULT_CHECKPOINT):
        self.checkpoint_path = Path(checkpoint_path)
        self.model = None
        self.vocab = None
        self.model_name = "Demo Rule-based Fallback"
        self.load_error = ""
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        if not self.checkpoint_path.exists():
            return
        try:
            import torch

            from models.toxicity_lstm import ToxicityBiLSTM

            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            vocab = checkpoint["vocab"]
            config = checkpoint.get("config", {})
            model = ToxicityBiLSTM(
                len(vocab),
                embed_dim=config.get("embed_dim", 128),
                hidden_dim=config.get("hidden_dim", 128),
                num_classes=config.get("num_classes", 2),
            )
            model.load_state_dict(checkpoint["model"])
            model.eval()
        except Exception as exc:
            self.load_error = str(exc)
            return
        self.torch = torch
        self.vocab = vocab
        self.model = model
        self.model_name = "BiLSTM checkpoint"
        self.max_len = checkpoint.get("config", {}).get("max_len", 80)

    def predict(self, text: str) -> ToxicityResult:
        if self.model is None or self.vocab is None:
            return heuristic_predict_toxicity(text)

        input_ids = self._encode(text)
        with self.torch.no_grad():
            logits = self.model(input_ids)
            probability = self.torch.softmax(logits, dim=1)[0, 1].item()
        probability = round(float(probability), 4)
        return ToxicityResult(
            probability=probability,
            level=risk_level(probability),
            action=moderation_action(probability),
            matched_terms=[],
            model_name=self.model_name,
        )

    def _encode(self, text: str):
        max_len = getattr(self, "max_len", 80)
        ids = [self.vocab.get(token, 1) for token in tokenize(text)[:max_len]]
        ids += [0] * (max_len - len(ids))
        return self.torch.tensor([ids], dtype=self.torch.long)


@lru_cache(maxsize=1)
def get_predictor() -> ToxicityPredictor:
    return ToxicityPredictor()


def predict_toxicity(text: str) -> ToxicityResult:
    return get_predictor().predict(text)
