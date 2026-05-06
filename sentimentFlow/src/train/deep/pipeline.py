from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...utils import load_config
from ..classical.utils import (
    compute_metrics,
    json_compatible,
    resolve_path,
    shape_of,
    to_bool,
    to_int,
    to_name_list,
)
from .data import (
    FeatureDataset,
    TextDataset,
    build_vocabulary,
    infer_feature_dim,
    load_feature_arrays,
    load_dataset_text_split,
    load_text_splits,
)
from .model import (
    FeatureHMMClassifier,
    FeatureMLP,
    FeatureSequenceClassifier,
    TextHMMClassifier,
    TextSequenceClassifier,
)


SUPPORTED_MODEL_TYPES = {"nn", "rnn", "lstm", "hmm", "gru"}
SEQUENCE_MODEL_TYPES = {"rnn", "lstm", "gru"}


def _to_float(value: Any, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return float(stripped)
    return default


def _to_int_list(value: Any, fallback_dim: int, fallback_layers: int) -> list[int]:
    if isinstance(value, list):
        dims = [to_int(item, fallback_dim) for item in value]
        return [dim for dim in dims if dim > 0] or [fallback_dim]

    if isinstance(value, tuple):
        dims = [to_int(item, fallback_dim) for item in value]
        return [dim for dim in dims if dim > 0] or [fallback_dim]

    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            dims = [
                to_int(item, fallback_dim)
                for item in stripped.strip("[]").split(",")
                if item.strip()
            ]
            return [dim for dim in dims if dim > 0] or [fallback_dim]

    return [fallback_dim] * max(1, fallback_layers)


def _to_string_list(value: Any, default: list[str]) -> list[str]:
    if isinstance(value, list):
        items = [str(item).strip().lower() for item in value if str(item).strip()]
        return items or default

    if isinstance(value, tuple):
        items = [str(item).strip().lower() for item in value if str(item).strip()]
        return items or default

    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            items = [
                item.strip().lower()
                for item in stripped.strip("[]").split(",")
                if item.strip()
            ]
            return items or default

    return default


def _hash_payload(payload: dict[str, Any], digest_len: int = 16) -> str:
    canonical = json.dumps(
        json_compatible(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:digest_len]


class DLPipeline:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config if config is not None else load_config()
        self.output_root = resolve_path(self.config.get("DL_OUTPUT_DIR"), "models/dl_models")
        self.embedding_run_path = resolve_path(
            self.config.get("DL_EMBEDDING_RUN_PATH"),
            "models/embeddings",
        )
        self.dataset_path = resolve_path(
            self.config.get("CLEANED_DATASET_PATH") or self.config.get("DATASET_PATH"),
            "datasets/cleaned_dataset.csv",
        )
        self.text_column = str(self.config.get("DL_TEXT_COLUMN", "text"))
        self.label_column = str(self.config.get("DL_LABEL_COLUMN", "label"))
        self.test_size = _to_float(
            self.config.get("DL_TEST_SIZE", self.config.get("EMBEDDING_TEST_SIZE")),
            0.2,
        )
        self.random_state = to_int(self.config.get("DL_RANDOM_STATE"), 42)
        self.use_stratify = to_bool(
            self.config.get("DL_STRATIFY", self.config.get("EMBEDDING_STRATIFY")),
            True,
        )
        self.force_rerun = to_bool(self.config.get("DL_FORCE_RERUN"), False)
        self.require_cuda = to_bool(self.config.get("DL_REQUIRE_CUDA"), True)
        self.embedding_filter = to_name_list(self.config.get("DL_EMBEDDING_TYPES"))

        self.default_embedding_mode = str(
            self.config.get("DL_EMBEDDING_MODE", "network")
        ).strip().lower()
        self.default_epochs = to_int(self.config.get("DL_EPOCHS"), 10)
        self.default_batch_size = to_int(self.config.get("DL_BATCH_SIZE"), 64)
        self.default_learning_rate = _to_float(self.config.get("DL_LEARNING_RATE"), 1e-3)
        self.default_weight_decay = _to_float(self.config.get("DL_WEIGHT_DECAY"), 0.0)
        self.default_hidden_dim = to_int(self.config.get("DL_HIDDEN_DIM"), 128)
        self.default_hidden_layers = to_int(self.config.get("DL_HIDDEN_LAYERS"), 1)
        self.default_hidden_dims = _to_int_list(
            self.config.get("DL_HIDDEN_DIMS"),
            self.default_hidden_dim,
            self.default_hidden_layers,
        )
        self.default_hidden_dim = self.default_hidden_dims[0]
        self.default_hidden_layers = len(self.default_hidden_dims)
        self.default_num_hidden_states = to_int(
            self.config.get("DL_NUM_HIDDEN_STATES"),
            128,
        )
        self.default_embedding_dim = to_int(self.config.get("DL_TOKEN_EMBEDDING_DIM"), 128)
        self.default_max_sequence_length = to_int(
            self.config.get("DL_MAX_SEQUENCE_LENGTH"), 128
        )
        self.default_max_vocab_size = to_int(self.config.get("DL_MAX_VOCAB_SIZE"), 30000)
        self.default_dropout = _to_float(self.config.get("DL_DROPOUT"), 0.2)
        self.default_early_stopping_patience = to_int(
            self.config.get("DL_EARLY_STOPPING_PATIENCE"),
            3,
        )
        self.default_early_stopping_min_delta = _to_float(
            self.config.get("DL_EARLY_STOPPING_MIN_DELTA"),
            0.0,
        )
        self.default_activation_functions = _to_string_list(
            self.config.get("DL_ACTIVATION_FUNCTIONS"),
            ["relu"],
        )
        self.default_bidirectional = to_bool(self.config.get("DL_BIDIRECTIONAL"), False)

        raw_runs = self.config.get("DL_RUNS", [])
        self.dl_runs = raw_runs if isinstance(raw_runs, list) else []
        self.uses_predefined_embeddings = any(
            self._run_parameters(run_config)["embedding_mode"] == "predefined"
            for run_config in self.dl_runs
            if isinstance(run_config, dict)
        )
        self.uses_default_embedding_run_path = any(
            self._run_parameters(run_config)["embedding_mode"] == "predefined"
            and run_config.get("embedding_run_path", run_config.get("embedding_path")) is None
            for run_config in self.dl_runs
            if isinstance(run_config, dict)
        )
        self.embedding_runs = (
            self._discover_embedding_runs(self.embedding_run_path)
            if self.uses_default_embedding_run_path
            else []
        )
        self.embedding_run_id = (
            self.embedding_run_path.name
            if self.uses_default_embedding_run_path
            else "network_embedding"
        )
        self.run_id = self._allocate_run_id()
        self.configuration_output_dir = self.output_root / self.run_id
        self.device = self._resolve_device()

    def _resolve_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if self.require_cuda:
            raise RuntimeError(
                "CUDA is required for the DL pipeline. Set DL_REQUIRE_CUDA=false "
                "to allow CPU fallback."
            )
        return torch.device("cpu")

    def _allocate_run_id(self) -> str:
        payload = {
            "predefined_embedding_run_path": (
                str(self.embedding_run_path)
                if self.uses_default_embedding_run_path
                else None
            ),
            "predefined_embedding_filter": (
                self.embedding_filter
                if self.uses_default_embedding_run_path
                else []
            ),
            "network_dataset": self._dataset_signature(),
            "dl_runs": self.dl_runs,
            "defaults": {
                "embedding_mode": self.default_embedding_mode,
                "epochs": self.default_epochs,
                "batch_size": self.default_batch_size,
                "learning_rate": self.default_learning_rate,
                "weight_decay": self.default_weight_decay,
                "hidden_dim": self.default_hidden_dim,
                "hidden_layers": self.default_hidden_layers,
                "hidden_dims": self.default_hidden_dims,
                "num_hidden_states": self.default_num_hidden_states,
                "token_embedding_dim": self.default_embedding_dim,
                "max_sequence_length": self.default_max_sequence_length,
                "max_vocab_size": self.default_max_vocab_size,
                "dropout": self.default_dropout,
                "early_stopping_patience": self.default_early_stopping_patience,
                "early_stopping_min_delta": self.default_early_stopping_min_delta,
                "activation_functions": self.default_activation_functions,
                "bidirectional": self.default_bidirectional,
            },
        }
        base_run_id = _hash_payload(payload)
        candidate = base_run_id
        counter = 1
        while (self.output_root / candidate).exists():
            candidate = f"{base_run_id}_{counter}"
            counter += 1
        return candidate

    def _dataset_signature(self) -> dict[str, Any]:
        signature: dict[str, Any] = {
            "dataset_path": str(self.dataset_path),
            "text_column": self.text_column,
            "label_column": self.label_column,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "use_stratify": self.use_stratify,
        }
        if self.dataset_path.exists():
            dataset_stat = self.dataset_path.stat()
            signature["dataset_size_bytes"] = dataset_stat.st_size
            signature["dataset_mtime_ns"] = dataset_stat.st_mtime_ns
        return signature

    def _discover_embedding_runs(self, embedding_run_path: Path) -> list[dict[str, Any]]:
        if not embedding_run_path.exists():
            raise FileNotFoundError(
                f"Embedding run path not found: {embedding_run_path}"
            )

        runs: list[dict[str, Any]] = []
        for metadata_path in embedding_run_path.glob("*/metadata.json"):
            runs.append(json.loads(metadata_path.read_text(encoding="utf-8")))

        if not runs:
            raise ValueError(
                f"No embedding metadata files found inside: {embedding_run_path}"
            )
        return runs

    def _selected_embedding_runs(
        self,
        embedding_runs: list[dict[str, Any]] | None = None,
        embedding_filter: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        runs = self.embedding_runs if embedding_runs is None else embedding_runs
        selected_filter = self.embedding_filter if embedding_filter is None else embedding_filter
        if not selected_filter:
            return runs
        return [
            run
            for run in runs
            if str(run.get("name", "")).strip().lower() in selected_filter
        ]

    def _embedding_runs_for_model(self, model_run_config: dict[str, Any]) -> list[dict[str, Any]]:
        model_embedding_filter = to_name_list(
            model_run_config.get(
                "embedding_types",
                model_run_config.get("embedding_type"),
            )
        )
        embedding_filter = model_embedding_filter or self.embedding_filter
        embedding_path = model_run_config.get(
            "embedding_run_path",
            model_run_config.get("embedding_path"),
        )
        if embedding_path is None:
            return self._selected_embedding_runs(embedding_filter=embedding_filter)
        model_embedding_path = resolve_path(embedding_path, "models/embeddings")
        return self._selected_embedding_runs(
            self._discover_embedding_runs(model_embedding_path),
            embedding_filter=embedding_filter,
        )

    def _run_parameters(self, run_config: dict[str, Any]) -> dict[str, Any]:
        model_type = str(run_config.get("type", "")).strip().lower()
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(f"Unsupported DL model type: {model_type}")

        embedding_mode = str(
            run_config.get("embedding_mode", self.default_embedding_mode)
        ).strip().lower()
        if model_type == "nn":
            embedding_mode = "predefined"
        if embedding_mode not in {"network", "predefined"}:
            raise ValueError(
                "DL embedding mode must be either 'network' or 'predefined'"
            )

        hidden_dim = to_int(run_config.get("hidden_dim"), self.default_hidden_dim)
        hidden_layers = to_int(
            run_config.get("hidden_layers"), self.default_hidden_layers
        )
        hidden_dims = _to_int_list(
            run_config.get("hidden_dims"),
            hidden_dim,
            hidden_layers,
        )
        hidden_dim = hidden_dims[0]
        hidden_layers = len(hidden_dims)

        params = {
            "model_type": model_type,
            "embedding_mode": embedding_mode,
            "epochs": to_int(run_config.get("epochs"), self.default_epochs),
            "batch_size": to_int(run_config.get("batch_size"), self.default_batch_size),
            "learning_rate": _to_float(
                run_config.get("learning_rate"), self.default_learning_rate
            ),
            "weight_decay": _to_float(
                run_config.get("weight_decay"), self.default_weight_decay
            ),
            "num_hidden_states": to_int(
                run_config.get("num_hidden_states"),
                self.default_num_hidden_states,
            ),
            "token_embedding_dim": to_int(
                run_config.get("token_embedding_dim"), self.default_embedding_dim
            ),
            "max_sequence_length": to_int(
                run_config.get("max_sequence_length"),
                self.default_max_sequence_length,
            ),
            "max_vocab_size": to_int(
                run_config.get("max_vocab_size"), self.default_max_vocab_size
            ),
            "dropout": _to_float(run_config.get("dropout"), self.default_dropout),
            "early_stopping_patience": to_int(
                run_config.get("early_stopping_patience"),
                self.default_early_stopping_patience,
            ),
            "early_stopping_min_delta": _to_float(
                run_config.get("early_stopping_min_delta"),
                self.default_early_stopping_min_delta,
            ),
            "activation_functions": _to_string_list(
                run_config.get("activation_functions"),
                self.default_activation_functions,
            ),
            "bidirectional": to_bool(
                run_config.get("bidirectional"), self.default_bidirectional
            ),
        }
        if model_type != "hmm":
            params.update(
                {
                    "hidden_dim": hidden_dim,
                    "hidden_layers": hidden_layers,
                    "hidden_dims": hidden_dims,
                }
            )
        return params

    def _is_cached_run_usable(
        self,
        metadata_path: Path,
        expected_hash: str,
    ) -> dict[str, Any] | None:
        if self.force_rerun or not metadata_path.exists():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if str(metadata.get("dl_configuration_hash", "")) != expected_hash:
            return None
        for path_key in {"model_path", "predictions_path", "metadata_path"}:
            value = metadata.get(path_key)
            if not value or not Path(str(value)).exists():
                return None
        metadata["status"] = "reused"
        return metadata

    def _make_feature_loaders(
        self,
        embedding_run: dict[str, Any],
        params: dict[str, Any],
    ) -> tuple[DataLoader, DataLoader, int, list[str], dict[str, Any]]:
        x_train, x_test, y_train, y_test, encoder = load_feature_arrays(embedding_run)
        train_dataset = FeatureDataset(x_train, y_train)
        test_dataset = FeatureDataset(x_test, y_test)
        batch_size = int(params["batch_size"])
        return (
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
            infer_feature_dim(x_train),
            [str(label) for label in encoder.classes_],
            {
                "train_shape": shape_of(x_train),
                "test_shape": shape_of(x_test),
                "classes": [str(label) for label in encoder.classes_],
            },
        )

    def _make_text_loaders(
        self,
        embedding_run: dict[str, Any] | None,
        params: dict[str, Any],
    ) -> tuple[DataLoader, DataLoader, int, list[str], dict[str, Any]]:
        if embedding_run is None:
            x_train, x_test, y_train, y_test, encoder = load_dataset_text_split(
                self.dataset_path,
                text_column=self.text_column,
                label_column=self.label_column,
                test_size=self.test_size,
                random_state=self.random_state,
                use_stratify=self.use_stratify,
            )
            source = "dataset"
        else:
            x_train, x_test, y_train, y_test, encoder = load_text_splits(
                embedding_run,
                text_column=self.text_column,
                label_column=self.label_column,
            )
            source = "embedding_split"
        vocabulary = build_vocabulary(x_train, int(params["max_vocab_size"]))
        train_dataset = TextDataset(
            x_train,
            y_train,
            vocabulary=vocabulary,
            max_length=int(params["max_sequence_length"]),
        )
        test_dataset = TextDataset(
            x_test,
            y_test,
            vocabulary=vocabulary,
            max_length=int(params["max_sequence_length"]),
        )
        batch_size = int(params["batch_size"])
        return (
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
            len(vocabulary),
            [str(label) for label in encoder.classes_],
            {
                "train_shape": [len(x_train), int(params["max_sequence_length"])],
                "test_shape": [len(x_test), int(params["max_sequence_length"])],
                "classes": [str(label) for label in encoder.classes_],
                "source": source,
                "vocabulary": vocabulary,
            },
        )

    def _build_model(
        self,
        params: dict[str, Any],
        *,
        input_dim: int,
        output_dim: int,
        ) -> nn.Module:
        model_type = str(params["model_type"])
        hmm_common = {
            "num_hidden_states": int(params["num_hidden_states"]),
            "dropout": float(params["dropout"]),
        }

        if params["embedding_mode"] == "predefined":
            if model_type == "nn":
                hidden_layer_common = {
                    "hidden_dims": [int(dim) for dim in params["hidden_dims"]],
                    "dropout": float(params["dropout"]),
                }
                return FeatureMLP(
                    input_dim,
                    output_dim,
                    activation_functions=[
                        str(activation)
                        for activation in params["activation_functions"]
                    ],
                    **hidden_layer_common,
                )
            if model_type == "hmm":
                return FeatureHMMClassifier(input_dim, output_dim, **hmm_common)
            hidden_layer_common = {
                "hidden_dims": [int(dim) for dim in params["hidden_dims"]],
                "dropout": float(params["dropout"]),
            }
            return FeatureSequenceClassifier(
                input_dim,
                output_dim,
                model_type=model_type,
                bidirectional=bool(params["bidirectional"]),
                **hidden_layer_common,
            )

        if model_type == "hmm":
            return TextHMMClassifier(
                input_dim,
                output_dim,
                embedding_dim=int(params["token_embedding_dim"]),
                padding_idx=0,
                **hmm_common,
            )
        if model_type in SEQUENCE_MODEL_TYPES:
            hidden_layer_common = {
                "hidden_dims": [int(dim) for dim in params["hidden_dims"]],
                "dropout": float(params["dropout"]),
            }
            return TextSequenceClassifier(
                input_dim,
                output_dim,
                model_type=model_type,
                embedding_dim=int(params["token_embedding_dim"]),
                bidirectional=bool(params["bidirectional"]),
                padding_idx=0,
                **hidden_layer_common,
            )
        raise ValueError("NN requires predefined embeddings via DL_EMBEDDING_RUN_PATH")

    def _train_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        params: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
        model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(params["learning_rate"]),
            weight_decay=float(params["weight_decay"]),
        )
        history: list[dict[str, Any]] = []
        overfitting_streak = 0
        early_stopping_patience = max(1, int(params["early_stopping_patience"]))
        early_stopping_min_delta = max(0.0, float(params["early_stopping_min_delta"]))

        epoch_iterator = tqdm(
            range(1, int(params["epochs"]) + 1),
            desc=f"Training {params['model_type']} epochs",
            unit="epoch",
        )
        for epoch in epoch_iterator:
            model.train()
            total_loss = 0.0
            total_examples = 0
            batch_iterator = tqdm(
                train_loader,
                desc=f"{params['model_type']} epoch {epoch} batches",
                leave=False,
                unit="batch",
            )
            for features, labels in batch_iterator:
                features = features.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(features)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * labels.size(0)
                total_examples += labels.size(0)
                batch_iterator.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")

            y_true, y_pred, eval_loss = self._predict(model, test_loader, criterion)
            metrics = compute_metrics(y_true, y_pred)
            train_loss = total_loss / max(1, total_examples)
            validation_train_loss_gap = eval_loss - train_loss
            is_overfitting = validation_train_loss_gap > (
                early_stopping_min_delta + 1e-12
            )
            overfitting_streak = overfitting_streak + 1 if is_overfitting else 0
            should_stop = overfitting_streak >= early_stopping_patience
            epoch_iterator.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{eval_loss:.4f}",
                accuracy=f"{metrics['accuracy']:.4f}",
                overfit_streak=overfitting_streak,
            )
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": train_loss,
                    "validation_loss": eval_loss,
                    "validation_train_loss_gap": validation_train_loss_gap,
                    "early_stopping_min_delta": early_stopping_min_delta,
                    "validation_gap_exceeded_threshold": is_overfitting,
                    "overfitting_streak": overfitting_streak,
                    "early_stopped": should_stop,
                    "early_stopping_reason": (
                        "validation_loss_gap_exceeded_threshold"
                        if should_stop
                        else None
                    ),
                    **metrics,
                }
            )
            if should_stop:
                epoch_iterator.write(
                    "Early stopping: validation loss exceeded training loss by more "
                    f"than {early_stopping_min_delta:g} for "
                    f"{early_stopping_patience} consecutive epochs."
                )
                break

        y_true, y_pred, _ = self._predict(model, test_loader, criterion)
        return history, y_true, y_pred

    def _predict(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        criterion: nn.Module | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        model.eval()
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        total_loss = 0.0
        total_examples = 0
        with torch.no_grad():
            for features, labels in tqdm(
                data_loader,
                desc="Evaluating DL batches",
                leave=False,
                unit="batch",
            ):
                features = features.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                logits = model(features)
                if criterion is not None:
                    loss = criterion(logits, labels)
                    total_loss += float(loss.detach().cpu()) * labels.size(0)
                predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
                targets.append(labels.cpu().numpy())
                total_examples += labels.size(0)
        return (
            np.concatenate(targets),
            np.concatenate(predictions),
            total_loss / max(1, total_examples),
        )

    def _run_single_model(
        self,
        embedding_run: dict[str, Any] | None,
        model_run_config: dict[str, Any],
    ) -> dict[str, Any]:
        params = self._run_parameters(model_run_config)
        embedding_name = (
            str(embedding_run.get("name", "")).strip().lower()
            if embedding_run is not None
            else "network_embedding"
        )
        model_name = str(model_run_config.get("name", "")).strip().lower()
        run_hash = _hash_payload(
            {
                "embedding_run": embedding_run if embedding_run is not None else None,
                "network_dataset": (
                    self._dataset_signature()
                    if params["embedding_mode"] == "network"
                    else None
                ),
                "model_run_config": model_run_config,
                "params": params,
                "device": str(self.device),
            }
        )
        run_dir = self.configuration_output_dir / embedding_name / model_name
        metadata_path = run_dir / "metadata.json"

        cached = self._is_cached_run_usable(metadata_path, run_hash)
        if cached is not None:
            return cached

        run_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(self.random_state)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.random_state)

        if params["embedding_mode"] == "predefined":
            train_loader, test_loader, input_dim, labels, data_summary = (
                self._make_feature_loaders(embedding_run, params)
            )
        else:
            train_loader, test_loader, input_dim, labels, data_summary = (
                self._make_text_loaders(embedding_run, params)
            )

        model = self._build_model(params, input_dim=input_dim, output_dim=len(labels))
        history, y_true, y_pred = self._train_model(
            model,
            train_loader,
            test_loader,
            params,
        )

        model_path = run_dir / "model.pt"
        predictions_path = run_dir / "y_pred.joblib"
        vocabulary = data_summary.pop("vocabulary", None)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_type": params["model_type"],
                "embedding_mode": params["embedding_mode"],
                "input_dim": input_dim,
                "labels": labels,
                "parameters": json_compatible(params),
                "vocabulary": vocabulary,
            },
            model_path,
        )
        joblib.dump(y_pred, predictions_path)

        run_summary = {
            "embedding_name": embedding_name,
            "embedding_type": (
                str(embedding_run.get("type", "")).lower()
                if embedding_run is not None
                else "network"
            ),
            "model_name": model_name,
            "model_type": params["model_type"],
            "embedding_mode": params["embedding_mode"],
            "embedding_run_id": (
                self.embedding_run_id
                if params["embedding_mode"] == "predefined"
                and self.uses_default_embedding_run_path
                else None
            ),
            "dataset_path": (
                str(self.dataset_path)
                if params["embedding_mode"] == "network"
                else None
            ),
            "dl_run_id": self.run_id,
            "dl_configuration_hash": run_hash,
            "status": "generated",
            "output_dir": str(run_dir),
            "model_path": str(model_path),
            "predictions_path": str(predictions_path),
            "metrics": json_compatible(compute_metrics(y_true, y_pred)),
            "history": json_compatible(history),
            "parameters": json_compatible(params),
            "configured_parameters": json_compatible(model_run_config),
            "device": str(self.device),
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if self.device.type == "cuda" else None
            ),
            "train_shape": data_summary["train_shape"],
            "test_shape": data_summary["test_shape"],
            "classes": data_summary["classes"],
            "metadata_path": str(metadata_path),
        }

        metadata_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return run_summary

    def run(self) -> dict[str, Any]:
        if not self.dl_runs:
            raise ValueError(
                "No DL runs found in config. Set dl.model_type and dl.models "
                "in dl_config.yaml."
            )

        self.output_root.mkdir(parents=True, exist_ok=True)
        self.configuration_output_dir.mkdir(parents=True, exist_ok=True)

        runs: list[dict[str, Any]] = []
        for model_run in self.dl_runs:
            if not isinstance(model_run, dict):
                continue

            model_params = self._run_parameters(model_run)
            if model_params["embedding_mode"] == "network":
                runs.append(self._run_single_model(None, model_run))
                continue

            selected_embedding_runs = self._embedding_runs_for_model(model_run)
            if not selected_embedding_runs:
                raise ValueError(
                    "No embedding runs available for DL training. "
                    "This only applies when embedding_mode is predefined. "
                    "Check dl.embedding_run_path, model.embedding_path, and "
                    "dl.embedding_types."
                )

            for embedding_run in selected_embedding_runs:
                runs.append(self._run_single_model(embedding_run, model_run))

        summary = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "embedding_run_path": (
                str(self.embedding_run_path)
                if self.uses_default_embedding_run_path
                else None
            ),
            "embedding_run_id": (
                self.embedding_run_id
                if self.uses_default_embedding_run_path
                else None
            ),
            "dataset_path": str(self.dataset_path),
            "test_size": self.test_size,
            "use_stratify": self.use_stratify,
            "dl_run_id": self.run_id,
            "output_root": str(self.output_root),
            "random_state": self.random_state,
            "force_rerun": self.force_rerun,
            "embedding_filter": (
                self.embedding_filter
                if self.uses_default_embedding_run_path
                else []
            ),
            "device": str(self.device),
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if self.device.type == "cuda" else None
            ),
            "runs": runs,
        }
        summary_path = self.configuration_output_dir / "dl_summary.json"
        summary["summary_path"] = str(summary_path)
        latest_summary_path = self.output_root / "dl_summary.json"
        summary["latest_summary_path"] = str(latest_summary_path)

        summary_content = json.dumps(summary, indent=2)
        summary_path.write_text(summary_content, encoding="utf-8")
        latest_summary_path.write_text(summary_content, encoding="utf-8")
        return summary


def run_dl_pipeline(config: dict[str, Any] | None = None) -> dict[str, Any]:
    pipeline = DLPipeline(config=config)
    return pipeline.run()
