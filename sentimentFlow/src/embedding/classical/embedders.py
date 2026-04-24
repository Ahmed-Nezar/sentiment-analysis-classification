from __future__ import annotations

from pathlib import Path
from typing import Any

from .base_embedder import BaseEmbedder
from .glove_embedder import GloveAveragingEmbedder
from .sklearn_embedder import SklearnEmbedder
from .word2vec_embedder import WordModelAveragingEmbedder
from .utils import to_float_or_int, to_int, to_ngram_range, to_optional_int




def _resolve_path(value: Any, project_root: Path) -> Path:
    if isinstance(value, Path):
        return value

    as_path = Path(str(value))
    if as_path.is_absolute():
        return as_path
    return (project_root / as_path).resolve()


def build_embedder(run_config: dict[str, Any], project_root: Path) -> BaseEmbedder:
    embedder_type = str(run_config.get("type", "")).strip().lower()

    if embedder_type in {"bow", "tfidf"}:
        return SklearnEmbedder(
            kind=embedder_type,
            ngram_range=to_ngram_range(run_config.get("ngram_range"), (1, 1)),
            max_features=to_optional_int(run_config.get("max_features"), None),
            min_df=to_float_or_int(run_config.get("min_df"), 1),
            max_df=to_float_or_int(run_config.get("max_df"), 1.0),
        )

    if embedder_type in {"word2vec", "fasttext"}:
        return WordModelAveragingEmbedder(
            kind=embedder_type,
            vector_size=to_int(run_config.get("vector_size"), 300),
            window=to_int(run_config.get("window"), 5),
            min_count=to_int(run_config.get("min_count"), 1),
            workers=to_int(run_config.get("workers"), 4),
            sg=to_int(run_config.get("sg"), 1),
            epochs=to_int(run_config.get("epochs"), 5),
        )

    if embedder_type == "glove":
        glove_file_path = run_config.get("glove_file_path")
        if glove_file_path is None:
            raise ValueError(
                "GloVe embedder requires 'glove_file_path' in the embedding config"
            )

        return GloveAveragingEmbedder(
            glove_file_path=_resolve_path(glove_file_path, project_root),
            vector_size=to_int(run_config.get("vector_size"), 300),
        )

    raise ValueError(f"Unsupported embedding type: {embedder_type}")
