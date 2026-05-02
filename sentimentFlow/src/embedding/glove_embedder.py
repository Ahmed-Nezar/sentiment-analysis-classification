import numpy as np
import joblib
from pathlib import Path
from typing import Any, Iterable
from tqdm import tqdm

from .base_embedder import BaseEmbedder
from .utils import to_text_list, tokenize_text

class GloveAveragingEmbedder(BaseEmbedder):
    def __init__(self, glove_file_path: Path, vector_size: int = 300) -> None:
        self.glove_file_path = glove_file_path
        self.vector_size = vector_size
        self.embeddings: dict[str, np.ndarray] = {}

    def _load_filtered_embeddings(self, vocabulary: set[str]) -> dict[str, np.ndarray]:
        if not self.glove_file_path.exists():
            raise FileNotFoundError(f"GloVe file not found at {self.glove_file_path}")

        filtered: dict[str, np.ndarray] = {}
        with self.glove_file_path.open("r", encoding="utf-8") as glove_file:
            for line in tqdm(
                glove_file,
                desc="Loading GloVe vectors",
                unit="line",
            ):
                values = line.split()
                if len(values) < 2:
                    continue

                word = values[0]
                if word not in vocabulary:
                    continue

                vector = np.asarray(values[1:], dtype=np.float32)
                if vector.shape[0] != self.vector_size:
                    continue

                filtered[word] = vector
                if len(filtered) == len(vocabulary):
                    break

        return filtered

    def _sentence_to_vector(self, tokens: list[str]) -> np.ndarray:
        vectors = [self.embeddings[token] for token in tokens if token in self.embeddings]
        if not vectors:
            return np.zeros(self.vector_size, dtype=np.float32)
        return np.mean(np.asarray(vectors, dtype=np.float32), axis=0)

    def _texts_to_matrix(self, tokenized_texts: list[list[str]]) -> np.ndarray:
        if not tokenized_texts:
            return np.zeros((0, self.vector_size), dtype=np.float32)
        matrix = np.vstack(
            [
                self._sentence_to_vector(tokens)
                for tokens in tqdm(
                    tokenized_texts,
                    desc="Vectorizing GloVe sentences",
                    unit="sentence",
                )
            ]
        )
        return matrix.astype(np.float32)

    def fit_transform(self, texts: Iterable[str]) -> Any:
        text_list = to_text_list(texts)
        tokenized_texts = [
            tokenize_text(text)
            for text in tqdm(
                text_list,
                desc="Tokenizing GloVe train text",
                unit="text",
            )
        ]
        vocabulary = {token for tokens in tokenized_texts for token in tokens}
        self.embeddings = self._load_filtered_embeddings(vocabulary)
        return self._texts_to_matrix(tokenized_texts)

    def transform(self, texts: Iterable[str]) -> Any:
        text_list = to_text_list(texts)
        tokenized_texts = [
            tokenize_text(text)
            for text in tqdm(
                text_list,
                desc="Tokenizing GloVe test text",
                unit="text",
            )
        ]
        return self._texts_to_matrix(tokenized_texts)

    def save(self, output_dir: Path) -> Path:
        output_path = output_dir / "embedding_object.joblib"
        joblib.dump(
            {
                "vector_size": self.vector_size,
                "embeddings": self.embeddings,
                "glove_file_path": str(self.glove_file_path),
            },
            output_path,
        )
        return output_path

    def get_params(self) -> dict[str, Any]:
        return {
            "kind": "glove",
            "vector_dimension": self.vector_size,
            "vector_size": self.vector_size,
            "glove_file_path": str(self.glove_file_path),
            "loaded_vocabulary_size": len(self.embeddings),
        }
