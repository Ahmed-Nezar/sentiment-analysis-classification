from typing import Any, Iterable

import numpy as np

def to_text_list(texts: Iterable[str]) -> list[str]:
    return [str(text) for text in texts]


def tokenize_text(text: str) -> list[str]:
    return str(text).split()


def to_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        return int(float(stripped))
    return default


def to_optional_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return default
    return to_int(value, default or 0)


def to_float_or_int(value: Any, default: float | int) -> float | int:
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        if "." in stripped:
            return float(stripped)
        return int(stripped)
    return default


def to_ngram_range(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (to_int(value[0], default[0]), to_int(value[1], default[1]))

    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) == 2:
            return (to_int(parts[0], default[0]), to_int(parts[1], default[1]))

    return default


def adjust_vector_dimension(
    features: np.ndarray,
    vector_dimension: int | None,
) -> np.ndarray:
    if vector_dimension is None:
        return features

    target_dimension = int(vector_dimension)
    if target_dimension <= 0:
        return features

    current_dimension = int(features.shape[1])
    if current_dimension == target_dimension:
        return features

    if current_dimension > target_dimension:
        return features[:, :target_dimension]

    padding = np.zeros(
        (features.shape[0], target_dimension - current_dimension),
        dtype=features.dtype,
    )
    return np.hstack([features, padding])
