import numpy as np
from gensim.models import Word2Vec, FastText
from typing import Any, Iterable
from pathlib import Path
from tqdm import tqdm

from .base_embedder import BaseEmbedder
from .utils import to_text_list, tokenize_text


class WordModelAveragingEmbedder(BaseEmbedder):
    def __init__(
        self,
        kind: str,
        vector_size: int = 300,
        window: int = 5,
        min_count: int = 1,
        workers: int = 4,
        sg: int = 1,
        epochs: int = 5,
    ) -> None:
        if kind not in {"word2vec", "fasttext"}:
            raise ValueError(f"Unsupported word model kind: {kind}")

        self.kind = kind
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.epochs = epochs
        self.model: Word2Vec | FastText | None = None

    def _build_model(self, tokenized_texts: list[list[str]]) -> None:
        model_cls = Word2Vec if self.kind == "word2vec" else FastText
        self.model = model_cls(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,
            epochs=self.epochs,
        )

    def _sentence_to_vector(self, tokens: list[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Word model has not been trained yet")

        vectors = [self.model.wv[token] for token in tokens if token in self.model.wv]
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
                    desc=f"Vectorizing {self.kind} sentences",
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
                desc=f"Tokenizing {self.kind} train text",
                unit="text",
            )
        ]
        self._build_model(tokenized_texts)
        return self._texts_to_matrix(tokenized_texts)

    def transform(self, texts: Iterable[str]) -> Any:
        text_list = to_text_list(texts)
        tokenized_texts = [
            tokenize_text(text)
            for text in tqdm(
                text_list,
                desc=f"Tokenizing {self.kind} test text",
                unit="text",
            )
        ]
        return self._texts_to_matrix(tokenized_texts)

    def save(self, output_dir: Path) -> Path:
        if self.model is None:
            raise RuntimeError("Word model has not been trained yet")

        output_path = output_dir / "embedding_object.model"
        self.model.save(str(output_path))
        return output_path

    def get_params(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "vector_dimension": self.vector_size,
            "vector_size": self.vector_size,
            "window": self.window,
            "min_count": self.min_count,
            "workers": self.workers,
            "sg": self.sg,
            "epochs": self.epochs,
        }


