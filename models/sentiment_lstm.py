import torch
from torch import nn


class SentimentBiLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128, num_classes: int = 3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids):
        hidden, _ = self.encoder(self.embedding(input_ids))
        mask = input_ids != 0
        attention_scores = self.attention(hidden).squeeze(-1)
        attention_scores = attention_scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(attention_scores, dim=1).unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=1)
        return self.classifier(pooled)
