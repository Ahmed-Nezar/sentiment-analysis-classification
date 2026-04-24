import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from typing import Any, Iterable
from pathlib import Path

from .base_embedder import BaseEmbedder
from .utils import to_text_list



class SklearnEmbedder(BaseEmbedder):
    def __init__(
        self,
        kind: str,
        ngram_range: tuple[int, int] = (1, 1),
        max_features: int | None = None,
        min_df: int | float = 1,
        max_df: int | float = 1.0,
    ) -> None:
        self.kind = kind
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df

        if kind == "bow":
            self.vectorizer = CountVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                min_df=min_df,
                max_df=max_df,
            )
        elif kind == "tfidf":
            self.vectorizer = TfidfVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                min_df=min_df,
                max_df=max_df,
            )
        else:
            raise ValueError(f"Unsupported vectorizer kind: {kind}")

    def fit_transform(self, texts: Iterable[str]) -> Any:
        return self.vectorizer.fit_transform(to_text_list(texts))

    def transform(self, texts: Iterable[str]) -> Any:
        return self.vectorizer.transform(to_text_list(texts))

    def save(self, output_dir: Path) -> Path:
        output_path = output_dir / "embedding_object.joblib"
        joblib.dump(self.vectorizer, output_path)
        return output_path

    def get_params(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "ngram_range": [int(self.ngram_range[0]), int(self.ngram_range[1])],
            "max_features": self.max_features,
            "min_df": self.min_df,
            "max_df": self.max_df,
        }
