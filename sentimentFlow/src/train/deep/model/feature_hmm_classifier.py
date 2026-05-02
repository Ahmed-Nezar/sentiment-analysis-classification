from __future__ import annotations

import torch
from torch import nn


class FeatureHMMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        num_hidden_states: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.emission = nn.Linear(input_dim, num_hidden_states)
        self.transition = nn.Parameter(
            torch.empty(num_hidden_states, num_hidden_states)
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_hidden_states, output_dim)
        nn.init.xavier_uniform_(self.transition)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        state = torch.softmax(self.emission(inputs.float()), dim=-1)
        transition = torch.softmax(self.transition, dim=0)
        state = torch.matmul(state, transition)
        return self.classifier(self.dropout(state))
