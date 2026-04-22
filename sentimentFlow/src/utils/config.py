from __future__ import annotations

from pathlib import Path
from typing import Final


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
CONFIG_FILE_PATH: Final[Path] = PROJECT_ROOT / ".config"
CONFIG = {}

def _parse_config_value(raw_value: str) -> Path:
	path = Path(raw_value.strip().strip('"').strip("'"))
	if path.is_absolute():
		return path
	return (CONFIG_FILE_PATH.parent / path).resolve()


def load_config_paths() -> dict[str, Path]:
    if not CONFIG_FILE_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found at {CONFIG_FILE_PATH}")

    config_paths: dict[str, Path] = {}
    for line in CONFIG_FILE_PATH.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in {"DATASET_PATH", "CLEANED_DATASET_PATH"}:
            config_paths[key] = _parse_config_value(value)

    CONFIG.update(config_paths)
