from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Final

from dotenv import load_dotenv


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
ENV_FILE_PATH: Final[Path] = PROJECT_ROOT / ".env"
CONFIG_FILE_PATH: Final[Path] = PROJECT_ROOT / ".config"
EMBEDDING_CONFIG_FILE_PATH: Final[Path] = PROJECT_ROOT / ".embedding_config"
ML_CONFIG_FILE_PATH: Final[Path] = PROJECT_ROOT / ".ml_config"
CONFIG_FILE_PATHS: Final[tuple[Path, ...]] = (
    CONFIG_FILE_PATH,
    EMBEDDING_CONFIG_FILE_PATH,
    ML_CONFIG_FILE_PATH,
)
DATASET_PATH_KEYS: Final[set[str]] = {"DATASET_PATH", "CLEANED_DATASET_PATH"}
CONFIG: dict[str, Any] = {}


def _strip_inline_comment(raw_value: str) -> str:
    in_single_quote = False
    in_double_quote = False

    for index, character in enumerate(raw_value):
        if character == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            continue

        if character == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            continue

        if character == "#" and not in_single_quote and not in_double_quote:
            return raw_value[:index].rstrip()

    return raw_value.rstrip()


def _strip_quotes(raw_value: str) -> str:
    return raw_value.strip().strip('"').strip("'")


def _is_path_key(key: str) -> bool:
    return key.endswith("_PATH") or key.endswith("_DIR")


def _parse_path_value(raw_value: str) -> Path:
    path = Path(_strip_quotes(_strip_inline_comment(raw_value)))
    if path.is_absolute():
        return path
    return (CONFIG_FILE_PATH.parent / path).resolve()


def _parse_scalar_value(raw_value: str) -> Any:
    value = _strip_quotes(_strip_inline_comment(raw_value))
    if value == "":
        return ""

    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None

    if "," in value:
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) > 1:
            return [_parse_scalar_value(part) for part in parts]

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return value


def _parse_config_file(config_path: Path) -> dict[str, Any]:
    parsed_config: dict[str, Any] = {}
    for line in config_path.read_text(encoding="utf-8").splitlines():
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith("#") or "=" not in stripped_line:
            continue

        key, raw_value = stripped_line.split("=", 1)
        normalized_key = key.strip()
        if _is_path_key(normalized_key):
            parsed_config[normalized_key] = _parse_path_value(raw_value)
        else:
            parsed_config[normalized_key] = _parse_scalar_value(raw_value)

    return parsed_config


def _to_name_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(item).strip().lower() for item in value if str(item).strip()]

    as_text = str(value).strip()
    if not as_text:
        return []
    return [item.strip().lower() for item in as_text.split(",") if item.strip()]


def _apply_prefixed_runs(
    config: dict[str, Any],
    *,
    prefix: str,
    types_key: str,
    runs_key: str,
    infer_type: Callable[[str], str],
) -> None:
    prefixed_configs: dict[str, dict[str, Any]] = {}
    prefix_marker = f"{prefix}__"

    for key, value in config.items():
        if not key.startswith(prefix_marker):
            continue

        parts = key.split("__", 2)
        if len(parts) != 3:
            continue

        _, run_name, setting_name = parts
        normalized_name = run_name.strip().lower()
        if not normalized_name:
            continue

        run_config = prefixed_configs.setdefault(normalized_name, {})
        run_config[setting_name.strip().lower()] = value

    configured_runs = _to_name_list(config.get(types_key))
    if not configured_runs and prefixed_configs:
        configured_runs = list(prefixed_configs.keys())

    runs: list[dict[str, Any]] = []
    for run_name in configured_runs:
        run_config = dict(prefixed_configs.get(run_name, {}))
        run_config["name"] = run_name
        run_config.setdefault("type", infer_type(run_name))
        runs.append(run_config)

    config[runs_key] = runs


def _apply_embedding_runs(config: dict[str, Any]) -> None:
    _apply_prefixed_runs(
        config,
        prefix="EMBEDDING",
        types_key="EMBEDDING_TYPES",
        runs_key="EMBEDDING_RUNS",
        infer_type=lambda run_name: run_name.split("_", 1)[0],
    )


def _apply_ml_runs(config: dict[str, Any]) -> None:
    _apply_prefixed_runs(
        config,
        prefix="ML",
        types_key="ML_TYPES",
        runs_key="ML_RUNS",
        infer_type=lambda run_name: run_name,
    )


def load_config() -> dict[str, Any]:
    load_dotenv(ENV_FILE_PATH, override=False)

    parsed_config: dict[str, Any] = {}
    existing_config_paths = [
        config_path
        for config_path in CONFIG_FILE_PATHS
        if config_path.exists()
    ]
    if not existing_config_paths:
        expected_paths = ", ".join(str(config_path) for config_path in CONFIG_FILE_PATHS)
        raise FileNotFoundError(
            "No configuration files were found. "
            f"Expected at least one of: {expected_paths}"
        )

    for config_path in existing_config_paths:
        parsed_config.update(_parse_config_file(config_path))

    _apply_embedding_runs(parsed_config)
    _apply_ml_runs(parsed_config)
    CONFIG.clear()
    CONFIG.update(parsed_config)
    return CONFIG


def load_config_paths() -> dict[str, Path]:
    loaded_config = load_config()
    config_paths: dict[str, Path] = {}
    for key in DATASET_PATH_KEYS:
        value = loaded_config.get(key)
        if isinstance(value, Path):
            config_paths[key] = value

    return config_paths
