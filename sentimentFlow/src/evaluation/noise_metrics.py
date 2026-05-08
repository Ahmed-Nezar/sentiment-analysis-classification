from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from ..extraction.noise_rows import get_noise_rows
from ..utils import PROJECT_ROOT


DEFAULT_MODEL_ROOTS = (
    PROJECT_ROOT / "models" / "ml_models",
    PROJECT_ROOT / "models" / "dl_models",
    PROJECT_ROOT / "models" / "encoder_models",
    PROJECT_ROOT / "models" / "decoder_models",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_compatible(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _as_project_path(value: Any) -> Path | None:
    if value is None:
        return None

    path = Path(str(value))
    if path.exists():
        return path

    if path.is_absolute():
        local_dataset = PROJECT_ROOT / "datasets" / path.name
        if local_dataset.exists():
            return local_dataset
        return path

    candidates = (
        PROJECT_ROOT / path,
        PROJECT_ROOT / "datasets" / path.name,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return PROJECT_ROOT / path


def _resolve_artifact_path(value: Any, model_dir: Path, fallback_name: str) -> Path | None:
    candidates: list[Path] = []
    if value is not None:
        raw = Path(str(value))
        candidates.append(raw)
        if raw.is_absolute():
            candidates.append(model_dir / raw.name)
        else:
            candidates.extend((model_dir / raw, PROJECT_ROOT / raw, model_dir / raw.name))

    candidates.append(model_dir / fallback_name)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_json_compatible(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_compatible(item) for item in value]
    return value


def _to_1d_list(values: Any) -> list[Any]:
    array = np.asarray(values)
    if array.ndim == 0:
        return [array.item()]
    return array.reshape(-1).tolist()


def _metrics(y_true: list[Any], y_pred: list[Any]) -> dict[str, Any]:
    labels = sorted(set(y_true) | set(y_pred), key=lambda item: str(item))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "support_by_label": {
            str(label): int(sum(item == label for item in y_true))
            for label in labels
        },
    }


def _dataset_is_clean(dataset_path: Path | None) -> bool:
    if dataset_path is None:
        return True
    return dataset_path.name.lower().startswith("cleaned_dataset")


class NoiseMetricsPipeline:
    def __init__(
        self,
        *,
        model_roots: Iterable[Path] | None = None,
        output_name: str = "metrics_without_noise.json",
    ) -> None:
        self.model_roots = tuple(model_roots or DEFAULT_MODEL_ROOTS)
        self.output_name = output_name
        self._dataset_cache: dict[Path, pd.DataFrame] = {}
        self._split_row_cache: dict[Path, list[int]] = {}
        self._sklearn_row_cache: dict[tuple[Path, float, int, bool, str], list[int]] = {}

    def run(self, model_dirs: Iterable[Path] | None = None) -> dict[str, Any]:
        selected_dirs = list(model_dirs) if model_dirs is not None else self._discover_model_dirs()
        results = []
        for model_dir in selected_dirs:
            try:
                results.append(self.recompute_for_model(model_dir))
            except Exception as exc:
                results.append(
                    {
                        "model_dir": str(model_dir),
                        "status": "skipped",
                        "reason": str(exc),
                    }
                )

        summaries = self._write_summaries(results)
        log_path = self._write_generation_log(results, summaries)
        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_model_dirs": len(selected_dirs),
            "generated_count": sum(item.get("status") == "generated" for item in results),
            "skipped_count": sum(item.get("status") == "skipped" for item in results),
            "results": results,
            "summary_paths": summaries,
            "generation_log_path": str(log_path),
        }

    def recompute_for_model(self, model_dir: Path) -> dict[str, Any]:
        model_dir = Path(model_dir)
        metadata_path = model_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {model_dir}")

        metadata = _read_json(metadata_path)
        y_pred_path = _resolve_artifact_path(
            metadata.get("predictions_path"),
            model_dir,
            "y_pred.joblib",
        )
        y_true_path = _resolve_artifact_path(
            metadata.get("targets_path") or metadata.get("target_path"),
            model_dir,
            "y_true.joblib",
        )
        if y_pred_path is None:
            raise FileNotFoundError("Saved predictions were not found.")

        y_pred = _to_1d_list(joblib.load(y_pred_path))
        if y_true_path is not None:
            y_true = _to_1d_list(joblib.load(y_true_path))
            row_numbers, dataset_path, dataset_kind = self._rows_from_sklearn_split(
                metadata,
                expected_size=len(y_pred),
            )
        else:
            embedding_mode = str(metadata.get("embedding_mode", "")).strip().lower()
            embedding_name = str(metadata.get("embedding_name", "")).strip().lower()
            if embedding_mode == "network" or embedding_name == "network_embedding":
                row_numbers, dataset_path, dataset_kind = self._rows_from_sklearn_split(
                    metadata,
                    expected_size=len(y_pred),
                )
                dataset_df = self._load_dataset(dataset_path)
                label_column = str(metadata.get("label_column", "label"))
                y_true = [
                    dataset_df.iloc[row_number - 1][label_column]
                    for row_number in row_numbers
                ]
            else:
                y_true, row_numbers, dataset_path, dataset_kind = self._rows_from_embedding_run(
                    metadata,
                    expected_size=len(y_pred),
                )

        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Prediction/target length mismatch: {len(y_pred)} predictions vs {len(y_true)} targets."
            )
        if len(row_numbers) != len(y_pred):
            raise ValueError(
                f"Evaluation row mapping length mismatch: {len(row_numbers)} rows vs {len(y_pred)} predictions."
            )

        clean = _dataset_is_clean(dataset_path)
        noisy_rows = get_noise_rows(clean=clean)
        keep_mask = [row_number not in noisy_rows for row_number in row_numbers]
        removed_data_rows = [
            int(row_number)
            for row_number, keep in zip(row_numbers, keep_mask, strict=True)
            if not keep
        ]
        filtered_true = [
            label
            for label, keep in zip(y_true, keep_mask, strict=True)
            if keep
        ]
        filtered_pred = [
            label
            for label, keep in zip(y_pred, keep_mask, strict=True)
            if keep
        ]

        output = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_metadata_path": metadata_path,
            "source_predictions_path": y_pred_path,
            "source_targets_path": y_true_path,
            "dataset_kind": dataset_kind,
            "dataset_path": dataset_path,
            "noise_policy": "strict_noisy_and_ambiguous_rows_removed",
            "noise_removed": True,
            "evaluation_size_original": len(y_pred),
            "noisy_rows_removed_count": len(removed_data_rows),
            "evaluation_size_without_noise": len(filtered_pred),
            "removed_data_rows": removed_data_rows,
            "original_metrics_from_metadata": (
                metadata.get("metrics")
                or metadata.get("final_metrics")
                or {}
            ),
            "recomputed_original_metrics": _metrics(y_true, y_pred),
            "metrics_without_noise": _metrics(filtered_true, filtered_pred),
            "labels": sorted(set(y_true) | set(y_pred), key=lambda item: str(item)),
            "notes": (
                "Metrics were recomputed from saved predictions after excluding "
                "strict noisy/ambiguous data rows from the evaluation split."
            ),
        }
        output.update(self._model_identity(metadata))

        output_path = model_dir / self.output_name
        _write_json(output_path, _json_compatible(output))
        return {
            "model_dir": str(model_dir),
            "status": "generated",
            "metrics_path": str(output_path),
            "dataset_kind": dataset_kind,
            "evaluation_size_original": len(y_pred),
            "noisy_rows_removed_count": len(removed_data_rows),
            "evaluation_size_without_noise": len(filtered_pred),
            "accuracy_without_noise": output["metrics_without_noise"]["accuracy"],
            "f1_macro_without_noise": output["metrics_without_noise"]["f1_macro"],
        }

    def _discover_model_dirs(self) -> list[Path]:
        dirs: list[Path] = []
        for root in self.model_roots:
            if not root.exists():
                continue
            for metadata_path in root.rglob("metadata.json"):
                if metadata_path.parent.name in {"model", "tokenizer"}:
                    continue
                dirs.append(metadata_path.parent)
        return sorted(set(dirs))

    def _rows_from_embedding_run(
        self,
        metadata: dict[str, Any],
        *,
        expected_size: int,
    ) -> tuple[list[Any], list[int], Path, str]:
        embedding_metadata_path = self._find_embedding_metadata(
            metadata,
            expected_size=expected_size,
        )
        embedding_metadata = _read_json(embedding_metadata_path)

        y_test_path = _as_project_path(embedding_metadata.get("y_test_path"))
        test_split_path = _as_project_path(embedding_metadata.get("test_split_path"))
        if y_test_path is None or not y_test_path.exists():
            raise FileNotFoundError("Embedding y_test.joblib was not found.")
        if test_split_path is None or not test_split_path.exists():
            raise FileNotFoundError("Embedding test_split.csv was not found.")

        y_true = _to_1d_list(joblib.load(y_test_path))
        if len(y_true) != expected_size:
            raise ValueError(
                f"Embedding y_test length {len(y_true)} does not match predictions length {expected_size}."
            )

        dataset_path = self._embedding_dataset_path(embedding_metadata)
        row_numbers = self._row_numbers_for_split(test_split_path, dataset_path)
        return y_true, row_numbers, dataset_path, self._dataset_kind(dataset_path)

    def _rows_from_sklearn_split(
        self,
        metadata: dict[str, Any],
        *,
        expected_size: int,
    ) -> tuple[list[int], Path, str]:
        dataset_path = _as_project_path(metadata.get("dataset_path"))
        if dataset_path is None or not dataset_path.exists():
            dataset_path = PROJECT_ROOT / "datasets" / "cleaned_dataset.csv"

        test_size = float(metadata.get("test_size", 0.2))
        random_state = int(metadata.get("random_state", 42))
        use_stratify = bool(metadata.get("use_stratify", True))
        label_column = str(metadata.get("label_column", "label"))
        cache_key = (dataset_path, test_size, random_state, use_stratify, label_column)
        if cache_key not in self._sklearn_row_cache:
            dataset_df = self._load_dataset(dataset_path)
            indices = np.arange(len(dataset_df))
            labels = dataset_df[label_column].to_numpy()
            stratify = labels if use_stratify else None
            _, test_indices = train_test_split(
                indices,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify,
            )
            self._sklearn_row_cache[cache_key] = [int(index) + 1 for index in test_indices]

        row_numbers = self._sklearn_row_cache[cache_key]
        if len(row_numbers) != expected_size:
            raise ValueError(
                f"Reconstructed test split has {len(row_numbers)} rows but predictions have {expected_size}."
            )
        return row_numbers, dataset_path, self._dataset_kind(dataset_path)

    def _row_numbers_for_split(self, split_path: Path, dataset_path: Path) -> list[int]:
        if split_path in self._split_row_cache:
            return self._split_row_cache[split_path]

        split_df = pd.read_csv(split_path)
        dataset_df = self._load_dataset(dataset_path)
        text_column = "text"
        label_column = "label"
        row_lookup: dict[tuple[str, str], deque[int]] = defaultdict(deque)
        for index, row in dataset_df.iterrows():
            key = (str(row[text_column]), str(row[label_column]))
            row_lookup[key].append(int(index) + 1)

        row_numbers: list[int] = []
        for _, row in split_df.iterrows():
            key = (str(row[text_column]), str(row[label_column]))
            matches = row_lookup.get(key)
            if not matches:
                raise ValueError(
                    f"Could not map test split row to source dataset row: {key!r}"
                )
            row_numbers.append(matches.popleft())

        self._split_row_cache[split_path] = row_numbers
        return row_numbers

    def _load_dataset(self, dataset_path: Path) -> pd.DataFrame:
        dataset_path = dataset_path.resolve()
        if dataset_path not in self._dataset_cache:
            self._dataset_cache[dataset_path] = pd.read_csv(dataset_path)
        return self._dataset_cache[dataset_path]

    def _find_embedding_metadata(
        self,
        metadata: dict[str, Any],
        *,
        expected_size: int,
    ) -> Path:
        embedding_name = str(metadata.get("embedding_name", "")).strip()
        embedding_run_id = str(
            metadata.get("embedding_run_id")
            or metadata.get("ml_run_id")
            or metadata.get("dl_run_id")
            or ""
        ).strip()
        candidates: list[Path] = []
        if embedding_run_id and embedding_name:
            candidates.append(
                PROJECT_ROOT
                / "models"
                / "embeddings_runs"
                / embedding_run_id
                / embedding_name
                / "metadata.json"
            )
        if embedding_name:
            candidates.extend(
                (PROJECT_ROOT / "models" / "embeddings_runs").glob(
                    f"*/{embedding_name}/metadata.json"
                )
            )

        existing_candidates = [candidate for candidate in candidates if candidate.exists()]
        for candidate in existing_candidates:
            candidate_metadata = _read_json(candidate)
            y_test_path = _as_project_path(candidate_metadata.get("y_test_path"))
            if y_test_path is None or not y_test_path.exists():
                continue
            try:
                y_test = _to_1d_list(joblib.load(y_test_path))
            except Exception:
                continue
            if len(y_test) == expected_size:
                return candidate
        if existing_candidates:
            return existing_candidates[0]
        raise FileNotFoundError(
            f"Embedding metadata not found for embedding_name={embedding_name!r}."
        )

    def _embedding_dataset_path(self, embedding_metadata: dict[str, Any]) -> Path:
        payload = embedding_metadata.get("configuration_hash_payload", {})
        dataset_info = payload.get("dataset", {}) if isinstance(payload, dict) else {}
        dataset_path = _as_project_path(
            dataset_info.get("dataset_path") or embedding_metadata.get("dataset_path")
        )
        if dataset_path is None or not dataset_path.exists():
            dataset_path = PROJECT_ROOT / "datasets" / "cleaned_dataset.csv"
        return dataset_path

    def _dataset_kind(self, dataset_path: Path | None) -> str:
        return "cleaned_dataset" if _dataset_is_clean(dataset_path) else "dataset"

    def _model_identity(self, metadata: dict[str, Any]) -> dict[str, Any]:
        keys = (
            "model_name",
            "model_type",
            "task",
            "embedding_name",
            "embedding_type",
            "embedding_run_id",
            "ml_run_id",
            "dl_run_id",
        )
        return {key: metadata[key] for key in keys if key in metadata}

    def _write_summaries(self, results: list[dict[str, Any]]) -> list[str]:
        paths = []
        for root in self.model_roots:
            root_results = [
                item for item in results
                if str(item.get("model_dir", "")).startswith(str(root))
            ]
            if not root_results:
                continue
            summary_path = root / "metrics_without_noise_summary.json"
            _write_json(
                summary_path,
                {
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "model_root": root,
                    "generated_count": sum(
                        item.get("status") == "generated" for item in root_results
                    ),
                    "skipped_count": sum(
                        item.get("status") == "skipped" for item in root_results
                    ),
                    "results": root_results,
                },
            )
            paths.append(str(summary_path))
        return paths

    def _write_generation_log(
        self,
        results: list[dict[str, Any]],
        summary_paths: list[str],
    ) -> Path:
        generated_results = [
            item for item in results
            if item.get("status") == "generated"
        ]
        skipped_results = [
            item for item in results
            if item.get("status") == "skipped"
        ]
        summary_counts = {}
        for root in self.model_roots:
            summary_counts[root.name] = sum(
                str(item.get("model_dir", "")).startswith(str(root))
                and item.get("status") == "generated"
                for item in results
            )

        log_path = PROJECT_ROOT / "models" / "metrics_without_noise_generation_log.json"
        _write_json(
            log_path,
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "output_file_name": self.output_name,
                "noise_policy": "strict_noisy_and_ambiguous_rows_removed",
                "written_count": len(generated_results),
                "summary_counts": summary_counts,
                "summary_paths": summary_paths,
                "skipped_count": len(skipped_results),
                "skipped": skipped_results,
            },
        )
        return log_path


def run_noise_metrics_pipeline(
    *,
    model_roots: Iterable[Path] | None = None,
    model_dirs: Iterable[Path] | None = None,
    output_name: str = "metrics_without_noise.json",
) -> dict[str, Any]:
    pipeline = NoiseMetricsPipeline(model_roots=model_roots, output_name=output_name)
    return pipeline.run(model_dirs=model_dirs)
