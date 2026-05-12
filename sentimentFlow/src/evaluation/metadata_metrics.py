from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from .noise_metrics import (
    NoiseMetricsPipeline,
    _as_project_path,
    _read_json,
    _resolve_artifact_path,
    _to_1d_list,
    _write_json,
)


def _label_sort_key(value: Any) -> tuple[int, str]:
    as_text = str(value)
    try:
        return (0, f"{int(float(as_text)):08d}")
    except ValueError:
        return (1, as_text)


def _valid_labels(y_true: list[Any]) -> list[Any]:
    return sorted(set(y_true), key=_label_sort_key)


def _class_scores(
    y_true: list[Any],
    y_pred: list[Any],
    labels: list[Any],
) -> list[dict[str, Any]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    return [
        {
            "label": str(label),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    ]


def _detailed_metrics(y_true: list[Any], y_pred: list[Any]) -> dict[str, Any]:
    labels = _valid_labels(y_true)
    base_metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "precision_weighted": float(
            precision_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall_weighted": float(
            recall_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
    }
    return {
        **base_metrics,
        "labels": [str(label) for label in labels],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "support_by_label": {
            str(label): int(sum(item == label for item in y_true))
            for label in labels
        },
        "class_scores": _class_scores(y_true, y_pred, labels),
    }


def _merge_metrics(existing: Any, generated: dict[str, Any]) -> dict[str, Any]:
    existing_metrics = existing if isinstance(existing, dict) else {}
    return {
        **existing_metrics,
        **generated,
    }


class MetadataMetricsPipeline(NoiseMetricsPipeline):
    def __init__(
        self,
        *,
        model_roots: Iterable[Path] | None = None,
        dry_run: bool = False,
    ) -> None:
        super().__init__(model_roots=model_roots)
        self.dry_run = dry_run

    def run(self, model_dirs: Iterable[Path] | None = None) -> dict[str, Any]:
        selected_dirs = list(model_dirs) if model_dirs is not None else self._discover_model_dirs()
        results = []
        for model_dir in selected_dirs:
            try:
                results.append(self.update_metadata_for_model(model_dir))
            except Exception as exc:
                results.append(
                    {
                        "model_dir": str(model_dir),
                        "status": "skipped",
                        "reason": str(exc),
                    }
                )

        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_model_dirs": len(selected_dirs),
            "updated_count": sum(item.get("status") == "updated" for item in results),
            "skipped_count": sum(item.get("status") == "skipped" for item in results),
            "dry_run": self.dry_run,
            "results": results,
        }

    def update_metadata_for_model(self, model_dir: Path) -> dict[str, Any]:
        model_dir = Path(model_dir)
        metadata_path = model_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {model_dir}")

        metadata = _read_json(metadata_path)
        y_true, y_pred, source_targets_path = self._load_targets_and_predictions(
            metadata,
            model_dir,
        )
        generated_metrics = _detailed_metrics(y_true, y_pred)
        existing_metric_key = "final_metrics" if "final_metrics" in metadata else "metrics"
        metadata[existing_metric_key] = _merge_metrics(
            metadata.get(existing_metric_key),
            generated_metrics,
        )
        metadata["evaluation_details"] = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_predictions_path": metadata.get("predictions_path"),
            "source_targets_path": source_targets_path,
            "labels": generated_metrics["labels"],
            "confusion_matrix": generated_metrics["confusion_matrix"],
            "class_scores": generated_metrics["class_scores"],
            "support_by_label": generated_metrics["support_by_label"],
        }

        if not self.dry_run:
            _write_json(metadata_path, metadata)

        return {
            "model_dir": str(model_dir),
            "metadata_path": str(metadata_path),
            "status": "updated",
            "metric_key": existing_metric_key,
            "labels": generated_metrics["labels"],
        }

    def _load_targets_and_predictions(
        self,
        metadata: dict[str, Any],
        model_dir: Path,
    ) -> tuple[list[Any], list[Any], str | None]:
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
            source_targets_path = str(y_true_path)
        else:
            embedding_mode = str(metadata.get("embedding_mode", "")).strip().lower()
            embedding_name = str(metadata.get("embedding_name", "")).strip().lower()
            if embedding_mode == "network" or embedding_name == "network_embedding":
                row_numbers, dataset_path, _dataset_kind = self._rows_from_sklearn_split(
                    metadata,
                    expected_size=len(y_pred),
                )
                dataset_df = self._load_dataset(dataset_path)
                label_column = str(metadata.get("label_column", "label"))
                y_true = [
                    dataset_df.iloc[row_number - 1][label_column]
                    for row_number in row_numbers
                ]
                source_targets_path = f"{dataset_path}::{label_column}"
            else:
                y_true, _row_numbers, _dataset_path, _dataset_kind = self._rows_from_embedding_run(
                    metadata,
                    expected_size=len(y_pred),
                )
                source_targets_path = self._find_source_targets_path(metadata, len(y_pred))

        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Prediction/target length mismatch: {len(y_pred)} predictions vs {len(y_true)} targets."
            )
        return y_true, y_pred, source_targets_path

    def _find_source_targets_path(
        self,
        metadata: dict[str, Any],
        expected_size: int,
    ) -> str | None:
        try:
            embedding_metadata_path = self._find_embedding_metadata(
                metadata,
                expected_size=expected_size,
            )
            embedding_metadata = _read_json(embedding_metadata_path)
            y_test_path = _as_project_path(embedding_metadata.get("y_test_path"))
            return str(y_test_path) if y_test_path is not None else None
        except Exception:
            return None


def run_metadata_metrics_pipeline(
    *,
    model_roots: Iterable[Path] | None = None,
    model_dirs: Iterable[Path] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    pipeline = MetadataMetricsPipeline(model_roots=model_roots, dry_run=dry_run)
    return pipeline.run(model_dirs=model_dirs)
