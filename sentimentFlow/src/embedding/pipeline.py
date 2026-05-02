from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ..utils import PROJECT_ROOT, load_config
from .embedders import build_embedder


@dataclass
class SplitData:
    x_train_texts: pd.Series
    x_test_texts: pd.Series
    y_train: pd.Series
    y_test: pd.Series


def _to_int(value: Any, default: int) -> int:
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


def _to_float(value: Any, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        return float(stripped)
    return default


def _to_bool(value: Any, default: bool) -> bool:
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


def _resolve_path(value: Any, fallback_relative: str) -> Path:
    if value is None:
        return (PROJECT_ROOT / fallback_relative).resolve()

    if isinstance(value, Path):
        return value

    as_path = Path(str(value))
    if as_path.is_absolute():
        return as_path
    return (PROJECT_ROOT / as_path).resolve()


def _shape_of(features: Any) -> list[int]:
    shape = getattr(features, "shape", None)
    if shape is not None:
        return [int(value) for value in shape]

    if isinstance(features, list):
        return [len(features)]

    return []


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): _json_compatible(item)
            for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))
        }

    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]

    if isinstance(value, set):
        return [_json_compatible(item) for item in sorted(value, key=lambda x: str(x))]

    return value


def _hash_payload(payload: dict[str, Any], digest_len: int = 16) -> str:
    canonical = json.dumps(
        _json_compatible(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:digest_len]


class EmbeddingPipeline:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config if config is not None else load_config()
        self.text_column = str(self.config.get("EMBEDDING_TEXT_COLUMN", "text"))
        self.label_column = str(self.config.get("EMBEDDING_LABEL_COLUMN", "label"))
        self.test_size = _to_float(self.config.get("EMBEDDING_TEST_SIZE"), 0.2)
        self.random_state = _to_int(self.config.get("EMBEDDING_RANDOM_STATE"), 42)
        self.use_stratify = _to_bool(self.config.get("EMBEDDING_STRATIFY"), True)
        self.force_rerun = _to_bool(self.config.get("EMBEDDING_FORCE_RERUN"), False)

        self.dataset_path = _resolve_path(
            self.config.get("CLEANED_DATASET_PATH") or self.config.get("DATASET_PATH"),
            "datasets/cleaned_dataset.csv",
        )
        self.output_root = _resolve_path(
            self.config.get("EMBEDDINGS_OUTPUT_DIR"), "models/embeddings"
        )

        raw_runs = self.config.get("EMBEDDING_RUNS", [])
        self.embedding_runs = raw_runs if isinstance(raw_runs, list) else []
        self.dataset_signature = self._build_dataset_signature()
        self.configuration_hash_payload = self._build_configuration_hash_payload()
        self.configuration_hash = _hash_payload(self.configuration_hash_payload)
        self.configuration_output_dir = self.output_root / self.configuration_hash

    def _build_dataset_signature(self) -> dict[str, Any]:
        signature: dict[str, Any] = {"dataset_path": str(self.dataset_path)}
        if self.dataset_path.exists():
            dataset_stat = self.dataset_path.stat()
            signature["dataset_size_bytes"] = dataset_stat.st_size
            signature["dataset_mtime_ns"] = dataset_stat.st_mtime_ns
        return signature

    def _normalized_runs_for_hash(self) -> list[dict[str, Any]]:
        normalized_runs: list[dict[str, Any]] = []
        for run_config in self.embedding_runs:
            if not isinstance(run_config, dict):
                continue
            normalized_runs.append(_json_compatible(run_config))

        normalized_runs.sort(key=lambda config: str(config.get("name", "")))
        return normalized_runs

    def _build_configuration_hash_payload(self) -> dict[str, Any]:
        return {
            "embedding_runs": self._normalized_runs_for_hash(),
            "split_config": {
                "text_column": self.text_column,
                "label_column": self.label_column,
                "test_size": self.test_size,
                "random_state": self.random_state,
                "use_stratify": self.use_stratify,
            },
            "dataset": self.dataset_signature,
        }

    def _is_cached_run_usable(
        self,
        metadata_path: Path,
        expected_configuration_hash: str,
    ) -> dict[str, Any] | None:
        if self.force_rerun or not metadata_path.exists():
            return None

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

        metadata_hash = str(
            metadata.get("configuration_hash") or metadata.get("run_hash") or ""
        )
        if metadata_hash != expected_configuration_hash:
            return None

        required_path_fields = [
            "embedding_object_path",
            "x_train_path",
            "x_test_path",
            "y_train_path",
            "y_test_path",
            "train_split_path",
            "test_split_path",
            "metadata_path",
        ]
        for path_key in required_path_fields:
            file_path_value = metadata.get(path_key)
            if not file_path_value:
                return None
            if not Path(str(file_path_value)).exists():
                return None

        metadata["status"] = "reused"
        metadata["configuration_hash"] = expected_configuration_hash
        metadata["run_hash"] = expected_configuration_hash
        return metadata

    def _validate_dataset(self, dataset_df: pd.DataFrame) -> None:
        if self.text_column not in dataset_df.columns:
            raise KeyError(f"Missing text column '{self.text_column}' in dataset")
        if self.label_column not in dataset_df.columns:
            raise KeyError(f"Missing label column '{self.label_column}' in dataset")

    def _split_dataset(self, dataset_df: pd.DataFrame) -> SplitData:
        self._validate_dataset(dataset_df)

        x_values = dataset_df[self.text_column].astype(str)
        y_values = dataset_df[self.label_column]
        stratify_values = y_values if self.use_stratify else None

        try:
            split = train_test_split(
                x_values,
                y_values,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=stratify_values,
            )
        except ValueError:
            split = train_test_split(
                x_values,
                y_values,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=None,
            )

        x_train_texts, x_test_texts, y_train, y_test = split
        return SplitData(
            x_train_texts.reset_index(drop=True),
            x_test_texts.reset_index(drop=True),
            y_train.reset_index(drop=True),
            y_test.reset_index(drop=True),
        )

    def _save_splits(self, run_dir: Path, split_data: SplitData) -> tuple[Path, Path]:
        train_split_path = run_dir / "train_split.csv"
        test_split_path = run_dir / "test_split.csv"

        train_frame = pd.DataFrame(
            {
                self.text_column: split_data.x_train_texts,
                self.label_column: split_data.y_train,
            }
        )
        test_frame = pd.DataFrame(
            {
                self.text_column: split_data.x_test_texts,
                self.label_column: split_data.y_test,
            }
        )

        train_frame.to_csv(train_split_path, index=False)
        test_frame.to_csv(test_split_path, index=False)

        return train_split_path, test_split_path

    def _run_single_embedding(
        self,
        run_config: dict[str, Any],
        split_data: SplitData,
    ) -> dict[str, Any]:
        run_name = str(run_config.get("name", "")).strip().lower()
        if not run_name:
            raise ValueError("Each embedding config must include a non-empty run name")

        run_dir = self.configuration_output_dir / run_name
        metadata_path = run_dir / "metadata.json"

        cached = self._is_cached_run_usable(metadata_path, self.configuration_hash)
        if cached is not None:
            return cached

        run_dir.mkdir(parents=True, exist_ok=True)

        embedder = build_embedder(run_config, PROJECT_ROOT)
        try:
            configured_parameters = _json_compatible(run_config)
            effective_parameters = _json_compatible(embedder.get_params())

            with tqdm(
                total=4,
                desc=f"Embedding {run_name} stages",
                unit="stage",
            ) as progress:
                x_train_features = embedder.fit_transform(
                    split_data.x_train_texts.tolist()
                )
                progress.update()

                x_test_features = embedder.transform(split_data.x_test_texts.tolist())
                progress.update()

                x_train_path = run_dir / "X_train.joblib"
                x_test_path = run_dir / "X_test.joblib"
                y_train_path = run_dir / "y_train.joblib"
                y_test_path = run_dir / "y_test.joblib"

                joblib.dump(x_train_features, x_train_path)
                joblib.dump(x_test_features, x_test_path)
                joblib.dump(split_data.y_train.to_numpy(), y_train_path)
                joblib.dump(split_data.y_test.to_numpy(), y_test_path)
                progress.update()

                embedding_object_path = embedder.save(run_dir)
                train_split_path, test_split_path = self._save_splits(
                    run_dir,
                    split_data,
                )
                progress.update()

            run_summary = {
                "name": run_name,
                "type": str(run_config.get("type", "")).lower(),
                "configuration_hash": self.configuration_hash,
                "run_hash": self.configuration_hash,
                "status": "generated",
                "output_dir": str(run_dir),
                "embedding_object_path": str(embedding_object_path),
                "x_train_path": str(x_train_path),
                "x_test_path": str(x_test_path),
                "y_train_path": str(y_train_path),
                "y_test_path": str(y_test_path),
                "train_split_path": str(train_split_path),
                "test_split_path": str(test_split_path),
                "parameters": effective_parameters,
                "configured_parameters": configured_parameters,
                "configuration_hash_payload": _json_compatible(
                    self.configuration_hash_payload
                ),
                "train_shape": _shape_of(x_train_features),
                "test_shape": _shape_of(x_test_features),
                "metadata_path": str(metadata_path),
            }

            metadata_path.write_text(
                json.dumps(run_summary, indent=2),
                encoding="utf-8",
            )
            return run_summary
        finally:
            embedder.cleanup()

    def run(self) -> dict[str, Any]:
        if not self.embedding_runs:
            raise ValueError(
                "No embedding runs found in config. Set embedding.types and "
                "embedding.runs in embedding_config.yaml"
            )

        self.output_root.mkdir(parents=True, exist_ok=True)
        self.configuration_output_dir.mkdir(parents=True, exist_ok=True)
        dataset_df = pd.read_csv(self.dataset_path)
        split_data = self._split_dataset(dataset_df)

        runs: list[dict[str, Any]] = []
        for run_config in self.embedding_runs:
            if not isinstance(run_config, dict):
                continue
            runs.append(self._run_single_embedding(run_config, split_data))

        summary = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "configuration_hash": self.configuration_hash,
            "configuration_output_dir": str(self.configuration_output_dir),
            "configuration_hash_payload": _json_compatible(
                self.configuration_hash_payload
            ),
            "dataset_path": str(self.dataset_path),
            "dataset_signature": _json_compatible(self.dataset_signature),
            "output_root": str(self.output_root),
            "text_column": self.text_column,
            "label_column": self.label_column,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "use_stratify": self.use_stratify,
            "force_rerun": self.force_rerun,
            "hash_algorithm": "sha256",
            "hash_length": 16,
            "runs": runs,
        }

        summary_path = self.configuration_output_dir / "embedding_summary.json"
        summary["summary_path"] = str(summary_path)

        latest_summary_path = self.output_root / "embedding_summary.json"
        summary["latest_summary_path"] = str(latest_summary_path)

        summary_content = json.dumps(summary, indent=2)
        summary_path.write_text(summary_content, encoding="utf-8")
        latest_summary_path.write_text(summary_content, encoding="utf-8")
        return summary


def run_embedding_pipeline(config: dict[str, Any] | None = None) -> dict[str, Any]:
    pipeline = EmbeddingPipeline(config=config)
    return pipeline.run()
