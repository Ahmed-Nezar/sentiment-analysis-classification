from typing import Any, Iterable
from pathlib import Path

class BaseEmbedder:
    def fit_transform(self, texts: Iterable[str]) -> Any:
        raise NotImplementedError

    def transform(self, texts: Iterable[str]) -> Any:
        raise NotImplementedError

    def save(self, output_dir: Path) -> Path:
        raise NotImplementedError

    def get_params(self) -> dict[str, Any]:
        raise NotImplementedError