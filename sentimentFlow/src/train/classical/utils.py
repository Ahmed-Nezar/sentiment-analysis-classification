from pathlib import Path
from typing import Any
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ...utils import PROJECT_ROOT


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


def to_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return default


def to_name_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(item).strip().lower() for item in value if str(item).strip()]

    as_text = str(value).strip()
    if not as_text:
        return []
    return [item.strip().lower() for item in as_text.split(",") if item.strip()]


def to_bool_int_pair(
    value: Any,
    *,
    default_flag: bool,
    default_count: int,
) -> tuple[bool, int]:
    if isinstance(value, (list, tuple)):
        raw_items = list(value)
    elif value is None:
        raw_items = []
    else:
        raw_items = [value]

    flag = to_bool(raw_items[0], default_flag) if raw_items else default_flag
    count = to_int(raw_items[1], default_count) if len(raw_items) > 1 else default_count
    return flag, max(1, count)


def resolve_path(value: Any, fallback_relative: str) -> Path:
    if value is None:
        return (PROJECT_ROOT / fallback_relative).resolve()

    if isinstance(value, Path):
        return value

    as_path = Path(str(value))
    if as_path.is_absolute():
        return as_path
    return (PROJECT_ROOT / as_path).resolve()


def shape_of(features: Any) -> list[int]:
    shape = getattr(features, "shape", None)
    if shape is not None:
        return [int(value) for value in shape]

    if isinstance(features, list):
        return [len(features)]

    return []


def json_compatible(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): json_compatible(item)
            for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))
        }

    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]

    if isinstance(value, set):
        return [json_compatible(item) for item in sorted(value, key=lambda x: str(x))]

    return value


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }
