from __future__ import annotations

import torch
from torch import nn


class TextHMMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        output_dim: int,
        *,
        embedding_dim: int,
        num_hidden_states: int,
        dropout: float,
        padding_idx: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx,
        )
        self.emission = nn.Linear(embedding_dim, num_hidden_states)
        self.token_transition = nn.Parameter(
            torch.empty(num_hidden_states, num_hidden_states)
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_hidden_states, output_dim)
        nn.init.xavier_uniform_(self.token_transition)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        emissions = torch.softmax(self.emission(self.embedding(inputs.long())), dim=-1)
        state = emissions[:, 0, :]
        token_transition = torch.softmax(self.token_transition, dim=0)
        for step in range(1, emissions.size(1)):
            state = torch.matmul(state, token_transition) * emissions[:, step, :]
            state = state / state.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return self.classifier(self.dropout(state))
