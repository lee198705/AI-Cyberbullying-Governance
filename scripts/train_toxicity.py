"""Train a BiLSTM toxicity classifier on COLDataset-style CSV files.

Expected columns: TEXT,label where label 0=safe and 1=offensive.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.toxicity_lstm import ToxicityBiLSTM
from src.preprocess import tokenize


MODEL_CONFIG = {
    "embed_dim": 128,
    "hidden_dim": 128,
    "num_classes": 2,
    "max_len": 80,
}


def load_rows(path: Path) -> list[tuple[str, int]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return [(row["TEXT"], int(row["label"])) for row in csv.DictReader(f)]


def build_vocab(texts: list[str], max_size: int = 12000) -> dict[str, int]:
    counter = Counter(token for text in texts for token in tokenize(text))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, _ in counter.most_common(max_size - 2):
        vocab[token] = len(vocab)
    return vocab


def encode(text: str, vocab: dict[str, int], max_len: int = MODEL_CONFIG["max_len"]) -> list[int]:
    ids = [vocab.get(token, 1) for token in tokenize(text)[:max_len]]
    return ids + [0] * (max_len - len(ids))


def make_loader(rows, vocab, batch_size, shuffle: bool = True):
    x = torch.tensor([encode(text, vocab) for text, _ in rows], dtype=torch.long)
    y = torch.tensor([label for _, label in rows], dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def binary_metrics(preds: list[int], labels: list[int]) -> dict[str, float]:
    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
    tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / max(len(labels), 1)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate(model, rows, vocab, batch_size) -> dict[str, float]:
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for x, y in make_loader(rows, vocab, batch_size, shuffle=False):
            batch_preds = model(x).argmax(dim=1)
            preds.extend(batch_preds.tolist())
            labels.extend(y.tolist())
    return binary_metrics(preds, labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="../COLDataset-main/COLDataset/train.csv")
    parser.add_argument("--dev", default="../COLDataset-main/COLDataset/dev.csv")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", default="models/checkpoints/toxicity_bilstm.pt")
    args = parser.parse_args()

    train_rows = load_rows(Path(args.train))
    dev_rows = load_rows(Path(args.dev))
    vocab = build_vocab([text for text, _ in train_rows])
    model = ToxicityBiLSTM(
        len(vocab),
        embed_dim=MODEL_CONFIG["embed_dim"],
        hidden_dim=MODEL_CONFIG["hidden_dim"],
        num_classes=MODEL_CONFIG["num_classes"],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    criterion = nn.CrossEntropyLoss()
    best_f1 = -1.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in make_loader(train_rows, vocab, args.batch_size):
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        metrics = evaluate(model, dev_rows, vocab, args.batch_size)
        print(
            f"epoch={epoch} "
            f"loss={sum(losses)/len(losses):.4f} "
            f"accuracy={metrics['accuracy']:.4f} "
            f"precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} "
            f"f1={metrics['f1']:.4f}"
        )
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "vocab": vocab,
                    "config": MODEL_CONFIG,
                    "dev_metrics": metrics,
                },
                out_path,
            )
            print(f"saved best checkpoint to {out_path} with f1={best_f1:.4f}")


if __name__ == "__main__":
    main()
