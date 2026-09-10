import torch
from torch import nn


class ToxicityBiLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.dropout = nn.Dropout(0.35)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        output, _ = self.encoder(embedded)
        mask = (input_ids != 0).unsqueeze(-1)
        masked_output = output * mask
        lengths = mask.sum(dim=1).clamp(min=1)
        pooled = masked_output.sum(dim=1) / lengths
        return self.classifier(self.dropout(pooled))
