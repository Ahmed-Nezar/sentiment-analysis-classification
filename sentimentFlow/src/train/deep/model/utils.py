from __future__ import annotations

from typing import Literal

from torch import nn


SequenceType = Literal["rnn", "lstm", "gru"]


def recurrent_class(model_type: SequenceType) -> type[nn.RNNBase]:
    if model_type == "rnn":
        return nn.RNN
    if model_type == "lstm":
        return nn.LSTM
    return nn.GRU


def activation_layer(name: str) -> nn.Module:
    normalized_name = name.strip().lower()
    if normalized_name == "relu":
        return nn.ReLU()
    if normalized_name == "gelu":
        return nn.GELU()
    if normalized_name == "tanh":
        return nn.Tanh()
    if normalized_name == "sigmoid":
        return nn.Sigmoid()
    if normalized_name == "leaky_relu":
        return nn.LeakyReLU()
    if normalized_name == "elu":
        return nn.ELU()
    if normalized_name == "selu":
        return nn.SELU()
    if normalized_name in {"identity", "linear", "none"}:
        return nn.Identity()
    raise ValueError(f"Unsupported activation function: {name}")
