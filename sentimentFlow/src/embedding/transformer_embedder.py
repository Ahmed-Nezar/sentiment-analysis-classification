from __future__ import annotations

import gc
import importlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .base_embedder import BaseEmbedder
from .utils import to_text_list


SUPPORTED_TRANSFORMER_MODELS: set[str] = {
    "Qwen/Qwen3-Embedding-0.6B",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-m3",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
    "thenlper/gte-base",
    "thenlper/gte-small",
    "thenlper/gte-large",
    "bert-base-uncased",
    "bert-large-uncased",
}

BERT_TRANSFORMER_MODELS: set[str] = {
    "bert-base-uncased",
    "bert-large-uncased",
}


class TransformerEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 32,
        max_length: int = 128,
        device: str | None = None,
    ) -> None:
        normalized_model_name = str(model_name).strip()
        if normalized_model_name not in SUPPORTED_TRANSFORMER_MODELS:
            raise ValueError(f"Unsupported transformer embedding model: {model_name}")

        try:
            torch_module = importlib.import_module("torch")
        except Exception as exc:
            raise ImportError(
                "torch is required for transformer-based embeddings. "
                "Install dependencies with: uv sync"
            ) from exc

        self.model_name = normalized_model_name
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.torch = torch_module
        self.device = device or (
            "cuda" if self.torch.cuda.is_available() else "cpu"
        )
        self.uses_sentence_transformer = self.model_name not in BERT_TRANSFORMER_MODELS

        if self.uses_sentence_transformer:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:
                raise ImportError(
                    "sentence-transformers is required for transformer embedding model "
                    f"'{self.model_name}'. Install dependencies with: uv sync"
                ) from exc

            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.tokenizer = None
        else:
            try:
                transformers_module = importlib.import_module("transformers")
            except Exception as exc:
                raise ImportError(
                    "transformers is required for BERT-based embeddings. "
                    "Install dependencies with: uv sync"
                ) from exc

            auto_tokenizer = getattr(transformers_module, "AutoTokenizer")
            auto_model = getattr(transformers_module, "AutoModel")
            self.tokenizer = auto_tokenizer.from_pretrained(self.model_name)
            self.model = auto_model.from_pretrained(self.model_name).to(self.device)
            self.model.eval()

    def _encode_with_transformers(self, texts: list[str]) -> np.ndarray:
        all_embeddings: list[np.ndarray] = []

        for start_index in range(0, len(texts), self.batch_size):
            batch = texts[start_index : start_index + self.batch_size]
            encoded_inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            )
            encoded_inputs = {
                key: value.to(self.device)
                for key, value in encoded_inputs.items()
            }

            with self.torch.no_grad():
                outputs = self.model(**encoded_inputs)

            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def _encode_with_sentence_transformer(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings)

    def _encode(self, texts: Iterable[str]) -> np.ndarray:
        normalized_texts = to_text_list(texts)
        if not normalized_texts:
            return np.empty((0, 0), dtype=np.float32)

        if self.uses_sentence_transformer:
            return self._encode_with_sentence_transformer(normalized_texts)

        return self._encode_with_transformers(normalized_texts)

    def fit_transform(self, texts: Iterable[str]) -> Any:
        return self._encode(texts)

    def transform(self, texts: Iterable[str]) -> Any:
        return self._encode(texts)

    def save(self, output_dir: Path) -> Path:
        output_path = output_dir / "embedding_object.json"
        output_path.write_text(
            json.dumps(self.get_params(), indent=2),
            encoding="utf-8",
        )
        return output_path

    def get_params(self) -> dict[str, Any]:
        return {
            "kind": "transformer",
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "device": self.device,
            "uses_sentence_transformer": self.uses_sentence_transformer,
            "persists_model_weights": False,
        }

    def cleanup(self) -> None:
        if hasattr(self, "model"):
            del self.model

        if hasattr(self, "tokenizer"):
            del self.tokenizer

        gc.collect()
        if self.device.startswith("cuda") and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

