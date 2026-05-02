from __future__ import annotations

import torch
from torch import nn

from .utils import activation_layer


class FeatureMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dims: list[int],
        activation_functions: list[str],
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for layer_index, hidden_dim in enumerate(hidden_dims):
            activation_name = activation_functions[min(layer_index, len(activation_functions) - 1)]
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    activation_layer(activation_name),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs.float())
