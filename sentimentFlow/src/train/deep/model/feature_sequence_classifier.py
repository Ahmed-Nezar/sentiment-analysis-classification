from __future__ import annotations

import torch
from torch import nn

from .utils import SequenceType, recurrent_class


class FeatureSequenceClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        model_type: SequenceType,
        hidden_dims: list[int],
        dropout: float,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        recurrent_cls = recurrent_class(model_type)
        self.recurrent_layers = nn.ModuleList()
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            self.recurrent_layers.append(
                recurrent_cls(
                    current_dim,
                    hidden_dim,
                    num_layers=1,
                    bidirectional=bidirectional,
                    batch_first=True,
                )
            )
            current_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(current_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence = inputs.float().unsqueeze(1)
        for recurrent_layer in self.recurrent_layers:
            sequence, _ = recurrent_layer(sequence)
            sequence = self.dropout(sequence)
        return self.classifier(sequence[:, -1, :])
