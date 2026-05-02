from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Final

import yaml
from dotenv import load_dotenv


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
ENV_FILE_PATH: Final[Path] = PROJECT_ROOT / ".env"
CONFIG_FILE_PATH: Final[Path] = PROJECT_ROOT / "config.yaml"
EMBEDDING_CONFIG_FILE_PATH: Final[Path] = PROJECT_ROOT / "embedding_config.yaml"
ML_CONFIG_FILE_PATH: Final[Path] = PROJECT_ROOT / "ml_config.yaml"
DL_CONFIG_FILE_PATH: Final[Path] = PROJECT_ROOT / "dl_config.yaml"
CONFIG_FILE_PATHS: Final[tuple[Path, ...]] = (
    CONFIG_FILE_PATH,
    EMBEDDING_CONFIG_FILE_PATH,
    ML_CONFIG_FILE_PATH,
    DL_CONFIG_FILE_PATH,
)
DATASET_PATH_KEYS: Final[set[str]] = {"DATASET_PATH", "CLEANED_DATASET_PATH"}
CONFIG: dict[str, Any] = {}


def _is_path_key(key: str) -> bool:
    return key.endswith("_PATH") or key.endswith("_DIR")


def _normalize_key(raw_key: str) -> str:
    return raw_key.strip().upper()


def _normalize_setting_key(raw_key: str) -> str:
    return raw_key.strip().lower()


def _resolve_path_value(value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _normalize_value(key: str, value: Any) -> Any:
    if _is_path_key(key) and value is not None:
        return _resolve_path_value(value)
    return value


def _normalize_flat_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for raw_key, value in mapping.items():
        key = _normalize_key(str(raw_key))
        normalized[key] = _normalize_value(key, value)
    return normalized


def _section_to_flat_keys(
    section: dict[str, Any],
    *,
    prefix: str,
    aliases: dict[str, str],
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}

    for raw_key, value in section.items():
        key = str(raw_key).strip()
        normalized_key = key.upper()
        if isinstance(value, dict):
            continue

        config_key = aliases.get(key, aliases.get(normalized_key, f"{prefix}_{normalized_key}"))
        flattened[config_key] = _normalize_value(config_key, value)

    return flattened


def _section_runs_to_flat_keys(
    runs: dict[str, Any] | None,
    *,
    prefix: str,
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if not runs:
        return flattened

    for raw_run_name, raw_run_config in runs.items():
        if not isinstance(raw_run_config, dict):
            continue

        run_name = str(raw_run_name).strip().upper()
        for raw_setting_name, value in raw_run_config.items():
            setting_name = str(raw_setting_name).strip().upper()
            key = f"{prefix}__{run_name}__{setting_name}"
            flattened[key] = _normalize_value(key, value)

    return flattened


def _section_dl_models_to_flat_keys(models: dict[str, Any] | None) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if not models:
        return flattened

    for raw_model_name, raw_model_config in models.items():
        if not isinstance(raw_model_config, dict):
            continue

        model_name = str(raw_model_name).strip()
        if not model_name:
            continue

        base_settings = {
            str(setting_name).strip(): value
            for setting_name, value in raw_model_config.items()
            if not isinstance(value, dict)
        }
        nested_configs = {
            str(setting_name).strip(): value
            for setting_name, value in raw_model_config.items()
            if isinstance(value, dict)
        }

        if not nested_configs:
            run_name = model_name.upper()
            for raw_setting_name, value in base_settings.items():
                setting_name = raw_setting_name.upper()
                key = f"DL__{run_name}__{setting_name}"
                flattened[key] = _normalize_value(key, value)
            continue

        for config_name, config_values in nested_configs.items():
            if not isinstance(config_values, dict):
                continue

            run_name = f"{model_name}_{config_name}".upper()
            merged_settings = dict(base_settings)
            merged_settings.update(config_values)
            merged_settings.setdefault("type", model_name)
            merged_settings.setdefault("config_name", config_name)

            for raw_setting_name, value in merged_settings.items():
                setting_name = str(raw_setting_name).strip().upper()
                key = f"DL__{run_name}__{setting_name}"
                flattened[key] = _normalize_value(key, value)

    return flattened


def _normalize_embedding_config(raw_config: dict[str, Any]) -> dict[str, Any]:
    section = raw_config.get("embedding", raw_config.get("EMBEDDING"))
    if not isinstance(section, dict):
        return _normalize_flat_mapping(raw_config)

    aliases = {
        "output_dir": "EMBEDDINGS_OUTPUT_DIR",
        "OUTPUT_DIR": "EMBEDDINGS_OUTPUT_DIR",
        "types": "EMBEDDING_TYPES",
        "TYPES": "EMBEDDING_TYPES",
        "runs_to_execute": "EMBEDDING_TYPES",
        "RUNS_TO_EXECUTE": "EMBEDDING_TYPES",
    }
    flattened = _section_to_flat_keys(section, prefix="EMBEDDING", aliases=aliases)
    flattened.update(_section_runs_to_flat_keys(section.get("runs"), prefix="EMBEDDING"))
    return flattened


def _normalize_ml_config(raw_config: dict[str, Any]) -> dict[str, Any]:
    section = raw_config.get("ml", raw_config.get("ML"))
    if not isinstance(section, dict):
        return _normalize_flat_mapping(raw_config)

    aliases = {
        "types": "ML_TYPES",
        "TYPES": "ML_TYPES",
        "classifiers_to_execute": "ML_TYPES",
        "CLASSIFIERS_TO_EXECUTE": "ML_TYPES",
    }
    flattened = _section_to_flat_keys(section, prefix="ML", aliases=aliases)
    classifiers = section.get("classifiers")
    if not isinstance(classifiers, dict):
        classifiers = section.get("runs")
    flattened.update(_section_runs_to_flat_keys(classifiers, prefix="ML"))
    return flattened


def _normalize_dl_config(raw_config: dict[str, Any]) -> dict[str, Any]:
    section = raw_config.get("dl", raw_config.get("DL"))
    if not isinstance(section, dict):
        return _normalize_flat_mapping(raw_config)

    aliases = {
        "model_type": "DL_MODEL_TYPE",
        "MODEL_TYPE": "DL_MODEL_TYPE",
        "model_types": "DL_MODEL_TYPES",
        "MODEL_TYPES": "DL_MODEL_TYPES",
    }
    flattened = _section_to_flat_keys(section, prefix="DL", aliases=aliases)
    defaults = section.get("defaults")
    if isinstance(defaults, dict):
        flattened.update(
            _section_to_flat_keys(defaults, prefix="DL", aliases={})
        )
    models = section.get("models")
    if not isinstance(models, dict):
        models = section.get("runs")
    flattened.update(_section_dl_models_to_flat_keys(models))
    return flattened


def _normalize_yaml_config(raw_config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    if config_path == EMBEDDING_CONFIG_FILE_PATH:
        return _normalize_embedding_config(raw_config)
    if config_path == ML_CONFIG_FILE_PATH:
        return _normalize_ml_config(raw_config)
    if config_path == DL_CONFIG_FILE_PATH:
        return _normalize_dl_config(raw_config)
    return _normalize_flat_mapping(raw_config)


def _parse_config_file(config_path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    return _normalize_yaml_config(loaded, config_path)


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


def _apply_dl_runs(config: dict[str, Any]) -> None:
    configured_model_types = config.get("DL_MODEL_TYPES")
    if configured_model_types is None:
        configured_model_types = config.get("DL_MODEL_TYPE")
    if configured_model_types is not None:
        config["DL_MODEL_TYPES"] = configured_model_types

    prefixed_configs: dict[str, dict[str, Any]] = {}
    prefix_marker = "DL__"

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

    configured_runs = _to_name_list(config.get("DL_MODEL_TYPES"))
    if not configured_runs and prefixed_configs:
        configured_runs = list(prefixed_configs.keys())

    runs: list[dict[str, Any]] = []
    for run_name, prefixed_config in prefixed_configs.items():
        run_config = dict(prefixed_config)
        inferred_type = str(run_config.get("type", run_name)).strip().lower()
        run_config["name"] = run_name
        run_config.setdefault("type", inferred_type)
        if configured_runs and run_name not in configured_runs and inferred_type not in configured_runs:
            continue
        runs.append(run_config)

    config["DL_RUNS"] = runs


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
    _apply_dl_runs(parsed_config)
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
