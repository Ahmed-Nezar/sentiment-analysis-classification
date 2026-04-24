from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from ...utils import PROJECT_ROOT, load_config
from .classifiers import build_classifier


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


def _to_name_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(item).strip().lower() for item in value if str(item).strip()]

    as_text = str(value).strip()
    if not as_text:
        return []
    return [item.strip().lower() for item in as_text.split(",") if item.strip()]


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

class MLPipeline:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config if config is not None else load_config()
        self.output_root = _resolve_path(self.config.get("ML_OUTPUT_DIR"), "models/ml_models")
        self.embedding_run_path  = _resolve_path(
            self.config.get("ML_EMBEDDING_RUN_PATH"),
            "models/embeddings",
        )
        self.random_state = _to_int(self.config.get("ML_RANDOM_STATE"), 42)
        self.force_rerun = _to_bool(self.config.get("ML_FORCE_RERUN"), False)
        self.embedding_filter = _to_name_list(self.config.get("ML_EMBEDDING_TYPES"))

        raw_runs = self.config.get("ML_RUNS", [])
        self.ml_runs = raw_runs if isinstance(raw_runs, list) else []

        self.embedding_runs = self._discover_embedding_runs()
        self.ml_run_id = self.embedding_run_path.name

        self.configuration_output_dir = self.output_root / self.ml_run_id

    def _discover_embedding_runs(self) -> list[dict[str, Any]]:
        if not self.embedding_run_path.exists():
            raise FileNotFoundError(
                f"Embedding run path not found: {self.embedding_run_path}"
            )

        runs = []
        for metadata_path in self.embedding_run_path.glob("*/metadata.json"):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            runs.append(metadata)

        if not runs:
            raise ValueError(
                f"No embedding metadata files found inside: {self.embedding_run_path}"
            )

        return runs

    def _selected_embedding_runs(self) -> list[dict[str, Any]]:
        if self.embedding_filter:
            return [
                run for run in self.embedding_runs
                if str(run.get("name", "")).strip().lower() in self.embedding_filter
            ]

        return self.embedding_runs

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
            metadata.get("ml_configuration_hash") or metadata.get("run_hash") or ""
        )
        if metadata_hash != expected_configuration_hash:
            return None

        required_path_fields = [
            "model_path",
            "predictions_path",
            "metadata_path",
        ]
        for path_key in required_path_fields:
            file_path_value = metadata.get(path_key)
            if not file_path_value:
                return None
            if not Path(str(file_path_value)).exists():
                return None

        metadata["status"] = "reused"
        metadata["ml_configuration_hash"] = expected_configuration_hash
        return metadata

    def _load_embedding_features(
        self, embedding_run: dict[str, Any]
    ) -> tuple[Any, Any, Any, Any]:
        x_train_path = Path(str(embedding_run["x_train_path"]))
        x_test_path = Path(str(embedding_run["x_test_path"]))
        y_train_path = Path(str(embedding_run["y_train_path"]))
        y_test_path = Path(str(embedding_run["y_test_path"]))

        x_train = joblib.load(x_train_path)
        x_test = joblib.load(x_test_path)
        y_train = joblib.load(y_train_path)
        y_test = joblib.load(y_test_path)
        return x_train, x_test, y_train, y_test

    def _run_single_model(
        self,
        embedding_run: dict[str, Any],
        model_run_config: dict[str, Any],
    ) -> dict[str, Any]:
        embedding_name = str(embedding_run.get("name", "")).strip().lower()
        model_name = str(model_run_config.get("name", "")).strip().lower()

        if not embedding_name:
            raise ValueError("Embedding run must contain a non-empty 'name'")
        if not model_name:
            raise ValueError("ML run must contain a non-empty 'name'")

        run_dir = self.configuration_output_dir / embedding_name / model_name
        metadata_path = run_dir / "metadata.json"

        cached = self._is_cached_run_usable(metadata_path, self.ml_run_id)
        if cached is not None:
            return cached

        run_dir.mkdir(parents=True, exist_ok=True)

        x_train, x_test, y_train, y_test = self._load_embedding_features(embedding_run)

        classifier, effective_parameters = build_classifier(
            model_run_config,
            random_state=self.random_state,
        )
        configured_parameters = {
            key: value
            for key, value in model_run_config.items()
            if key not in {"name", "type"}
        }

        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)

        model_path = run_dir / "model.joblib"
        predictions_path = run_dir / "y_pred.joblib"

        joblib.dump(classifier, model_path)
        joblib.dump(y_pred, predictions_path)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision_macro": float(
                precision_score(y_test, y_pred, average="macro", zero_division=0)
            ),
            "recall_macro": float(
                recall_score(y_test, y_pred, average="macro", zero_division=0)
            ),
            "f1_macro": float(
                f1_score(y_test, y_pred, average="macro", zero_division=0)
            ),
        }

        run_summary = {
            "embedding_name": embedding_name,
            "embedding_type": str(embedding_run.get("type", "")).lower(),
            "model_name": model_name,
            "model_type": str(model_run_config.get("type", "")).lower(),
            "embedding_run_id": self.ml_run_id,
            "status": "generated",
            "output_dir": str(run_dir),
            "model_path": str(model_path),
            "predictions_path": str(predictions_path),
            "metrics": _json_compatible(metrics),
            "parameters": _json_compatible(effective_parameters),
            "configured_parameters": _json_compatible(configured_parameters),
            "train_shape": _shape_of(x_train),
            "test_shape": _shape_of(x_test),
            "metadata_path": str(metadata_path),
        }

        metadata_path.write_text(
            json.dumps(run_summary, indent=2),
            encoding="utf-8",
        )
        return run_summary

    def run(self) -> dict[str, Any]:
        if not self.ml_runs:
            raise ValueError(
                "No ML runs found in config. Set ML_TYPES and ML__<NAME>__* keys in .config"
            )

        selected_embedding_runs = self._selected_embedding_runs()
        if not selected_embedding_runs:
            raise ValueError(
                "No embedding runs available for training. "
                "Check the embedding summary file or ML_EMBEDDING_TYPES filter."
            )

        self.output_root.mkdir(parents=True, exist_ok=True)
        self.configuration_output_dir.mkdir(parents=True, exist_ok=True)

        runs: list[dict[str, Any]] = []
        for embedding_run in tqdm(
            selected_embedding_runs,
            desc="Training Models by Embedding",
        ):
            for model_run in self.ml_runs:
                if not isinstance(model_run, dict):
                    continue
                runs.append(self._run_single_model(embedding_run, model_run))

        summary = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "embedding_run_path": str(self.embedding_run_path),
            "embedding_run_id": self.ml_run_id,
            "output_root": str(self.output_root),
            "random_state": self.random_state,
            "force_rerun": self.force_rerun,
            "embedding_filter": self.embedding_filter,
            "runs": runs,
        }

        summary_path = self.configuration_output_dir / "ml_summary.json"
        summary["summary_path"] = str(summary_path)

        latest_summary_path = self.output_root / "ml_summary.json"
        summary["latest_summary_path"] = str(latest_summary_path)

        summary_content = json.dumps(summary, indent=2)
        summary_path.write_text(summary_content, encoding="utf-8")
        latest_summary_path.write_text(summary_content, encoding="utf-8")
        return summary


def run_ml_pipeline(config: dict[str, Any] | None = None) -> dict[str, Any]:
    pipeline = MLPipeline(config=config)
    return pipeline.run()
