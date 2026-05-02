from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from tqdm import tqdm


TOKEN_PATTERN = re.compile(r"\b\w+\b", re.IGNORECASE)
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def _dense_row(features: Any, index: int) -> np.ndarray:
    if sparse.issparse(features):
        return features[index].toarray().ravel().astype(np.float32)
    return np.asarray(features[index], dtype=np.float32).ravel()


class FeatureDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, features: Any, labels: np.ndarray) -> None:
        self.features = features
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.from_numpy(_dense_row(self.features, index))
        label = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return features, label


class TextDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        texts: list[str],
        labels: np.ndarray,
        *,
        vocabulary: dict[str, int],
        max_length: int,
    ) -> None:
        self.texts = texts
        self.labels = labels.astype(np.int64)
        self.vocabulary = vocabulary
        self.max_length = max(1, max_length)
        self.pad_index = vocabulary[PAD_TOKEN]
        self.unk_index = vocabulary[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids = [
            self.vocabulary.get(token, self.unk_index)
            for token in tokenize(self.texts[index])[: self.max_length]
        ]
        if len(token_ids) < self.max_length:
            token_ids.extend([self.pad_index] * (self.max_length - len(token_ids)))
        features = torch.tensor(token_ids, dtype=torch.long)
        label = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return features, label


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(str(text))]


def build_vocabulary(texts: list[str], max_vocab_size: int) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for text in tqdm(texts, desc="Building DL vocabulary", unit="text"):
        counter.update(tokenize(text))

    vocabulary = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    limit = max(2, max_vocab_size) - len(vocabulary)
    for token, _ in counter.most_common(limit):
        vocabulary[token] = len(vocabulary)
    return vocabulary


def load_feature_arrays(
    embedding_run: dict[str, Any],
) -> tuple[Any, Any, np.ndarray, np.ndarray, LabelEncoder]:
    x_train = joblib.load(Path(str(embedding_run["x_train_path"])))
    x_test = joblib.load(Path(str(embedding_run["x_test_path"])))
    y_train_raw = joblib.load(Path(str(embedding_run["y_train_path"])))
    y_test_raw = joblib.load(Path(str(embedding_run["y_test_path"])))

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw)
    y_test = encoder.transform(y_test_raw)
    return x_train, x_test, y_train, y_test, encoder


def load_text_splits(
    embedding_run: dict[str, Any],
    *,
    text_column: str,
    label_column: str,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, LabelEncoder]:
    train_frame = pd.read_csv(Path(str(embedding_run["train_split_path"])))
    test_frame = pd.read_csv(Path(str(embedding_run["test_split_path"])))

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_frame[label_column])
    y_test = encoder.transform(test_frame[label_column])
    return (
        train_frame[text_column].astype(str).tolist(),
        test_frame[text_column].astype(str).tolist(),
        y_train,
        y_test,
        encoder,
    )


def load_dataset_text_split(
    dataset_path: Path,
    *,
    text_column: str,
    label_column: str,
    test_size: float,
    random_state: int,
    use_stratify: bool,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, LabelEncoder]:
    dataset_df = pd.read_csv(dataset_path)
    if text_column not in dataset_df.columns:
        raise KeyError(f"Missing text column '{text_column}' in dataset")
    if label_column not in dataset_df.columns:
        raise KeyError(f"Missing label column '{label_column}' in dataset")

    x_values = dataset_df[text_column].astype(str)
    y_values = dataset_df[label_column]
    stratify_values = y_values if use_stratify else None

    try:
        split = train_test_split(
            x_values,
            y_values,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_values,
        )
    except ValueError:
        split = train_test_split(
            x_values,
            y_values,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )

    x_train, x_test, y_train_raw, y_test_raw = split
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw)
    y_test = encoder.transform(y_test_raw)
    return (
        x_train.reset_index(drop=True).tolist(),
        x_test.reset_index(drop=True).tolist(),
        y_train,
        y_test,
        encoder,
    )


def infer_feature_dim(features: Any) -> int:
    shape = getattr(features, "shape", None)
    if shape is not None and len(shape) > 1:
        return int(shape[1])
    first_row = _dense_row(features, 0)
    return int(first_row.shape[0])
