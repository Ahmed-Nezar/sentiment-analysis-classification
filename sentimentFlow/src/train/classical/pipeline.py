from __future__ import annotations

import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from .utils import to_bool, to_bool_int_pair, to_int, to_name_list, resolve_path, json_compatible, shape_of, compute_metrics
from .classifiers import build_classifier
from ...utils import load_config


class MLPipeline:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config if config is not None else load_config()
        self.output_root = resolve_path(self.config.get("ML_OUTPUT_DIR"), "models/ml_models")
        self.embedding_run_path  = resolve_path(
            self.config.get("ML_EMBEDDING_RUN_PATH"),
            "models/embeddings",
        )
        self.random_state = to_int(self.config.get("ML_RANDOM_STATE"), 42)
        self.force_rerun = to_bool(self.config.get("ML_FORCE_RERUN"), False)
        self.embedding_filter = to_name_list(self.config.get("ML_EMBEDDING_TYPES"))
        (
            self.hyperparameter_optimization_enabled,
            self.hyperparameter_optimization_trials,
        ) = to_bool_int_pair(
            self.config.get("ML_HYPERPARAMETER_OPTIMIZATION"),
            default_flag=False,
            default_count=10,
        )

        raw_runs = self.config.get("ML_RUNS", [])
        self.ml_runs = raw_runs if isinstance(raw_runs, list) else []

        self.embedding_runs = self._discover_embedding_runs()
        self.embedding_run_id = self.embedding_run_path.name
        self.ml_run_id = self._allocate_ml_run_id(
            self.embedding_run_id,
            hyperparameter_optimization_enabled=self.hyperparameter_optimization_enabled,
        )

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

    def _allocate_ml_run_id(
        self,
        base_run_id: str,
        *,
        hyperparameter_optimization_enabled: bool,
    ) -> str:
        if hyperparameter_optimization_enabled:
            counter = 1
            candidate_run_id = f"{base_run_id}_{counter}_hype"
            while (self.output_root / candidate_run_id).exists():
                counter += 1
                candidate_run_id = f"{base_run_id}_{counter}_hype"
            return candidate_run_id

        candidate_run_id = base_run_id
        counter = 1

        while (self.output_root / candidate_run_id).exists():
            candidate_run_id = f"{base_run_id}_{counter}"
            counter += 1

        return candidate_run_id

    def _load_optuna(self) -> Any:
        try:
            optuna = importlib.import_module("optuna")
        except Exception as exc:
            raise ImportError(
                "Optuna is required when ML_HYPERPARAMETER_OPTIMIZATION is enabled. "
                "Install dependencies with: uv sync"
            ) from exc

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        optuna.logging.disable_default_handler()
        return optuna

    def _suggest_hyperparameters(self, trial: Any, classifier_type: str) -> dict[str, Any]:
        if classifier_type == "logistic_regression":
            solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear", "saga"])
            params: dict[str, Any] = {
                "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
                "max_iter": trial.suggest_int("max_iter", 200, 2000, step=200),
                "solver": solver,
            }
            if solver in {"liblinear", "saga"}:
                params["penalty"] = trial.suggest_categorical("penalty", ["l1", "l2"])
            else:
                params["penalty"] = "l2"
            return params

        if classifier_type == "svm":
            kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"])
            params = {
                "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
                "kernel": kernel,
                "probability": False,
            }
            if kernel in {"rbf", "poly", "sigmoid"}:
                params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
            if kernel == "poly":
                params["degree"] = trial.suggest_int("degree", 2, 5)
            return params

        if classifier_type == "random_forest":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": trial.suggest_categorical(
                    "max_depth",
                    [None, 5, 10, 15, 20, 30, 40],
                ),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical(
                    "max_features",
                    ["sqrt", "log2"],
                ),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            }

        if classifier_type == "decision_tree":
            return {
                "criterion": trial.suggest_categorical(
                    "criterion",
                    ["gini", "entropy", "log_loss"],
                ),
                "splitter": trial.suggest_categorical("splitter", ["best", "random"]),
                "max_depth": trial.suggest_categorical(
                    "max_depth",
                    [None, 5, 10, 15, 20, 30, 40],
                ),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            }

        if classifier_type == "xgboost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    1e-3,
                    3e-1,
                    log=True,
                ),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1e1, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1e1, log=True),
            }

        raise ValueError(f"Unsupported classifier type for optimization: {classifier_type}")

    def _optimize_hyperparameters(
        self,
        embedding_run: dict[str, Any],
        model_run_config: dict[str, Any],
        *,
        x_train: Any,
        x_test: Any,
        y_train: Any,
        y_test: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        optuna = self._load_optuna()
        classifier_type = str(model_run_config.get("type", "")).strip().lower()
        model_name = str(model_run_config.get("name", "")).strip().lower()
        embedding_name = str(embedding_run.get("name", "")).strip().lower()

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial: Any) -> float:
            trial_params = self._suggest_hyperparameters(trial, classifier_type)
            candidate_run_config = {
                "name": model_name,
                "type": classifier_type,
                **trial_params,
            }
            classifier, effective_parameters = build_classifier(
                candidate_run_config,
                random_state=self.random_state,
            )
            classifier.fit(x_train, y_train)
            y_pred = classifier.predict(x_test)
            accuracy = float(accuracy_score(y_test, y_pred))
            trial.set_user_attr(
                "effective_parameters",
                json_compatible(effective_parameters),
            )
            return accuracy

        study.optimize(objective, n_trials=self.hyperparameter_optimization_trials)

        return (
            {
                "name": model_name,
                "type": classifier_type,
                **study.best_trial.user_attrs.get(
                    "effective_parameters",
                    json_compatible(study.best_trial.params),
                ),
            },
            {
                "enabled": True,
                "framework": "optuna",
                "direction": "maximize",
                "metric": "accuracy",
                "n_trials": self.hyperparameter_optimization_trials,
                "best_trial_number": int(study.best_trial.number),
                "best_accuracy": float(study.best_value),
                "best_params": json_compatible(study.best_trial.params),
                "best_effective_parameters": json_compatible(
                    study.best_trial.user_attrs.get(
                        "effective_parameters",
                        study.best_trial.params,
                    )
                ),
                "study_name": f"{embedding_name}_{model_name}",
            },
        )

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

        with tqdm(
            total=5,
            desc=f"ML {embedding_name}/{model_name} stages",
            unit="stage",
        ) as progress:
            x_train, x_test, y_train, y_test = self._load_embedding_features(
                embedding_run
            )
            progress.update()

            configured_parameters = {
                key: value
                for key, value in model_run_config.items()
                if key not in {"name", "type"}
            }
            optimization_summary: dict[str, Any] = {
                "enabled": False,
                "framework": None,
                "direction": None,
                "metric": None,
                "n_trials": 0,
            }

            if self.hyperparameter_optimization_enabled:
                classifier_run_config, optimization_summary = (
                    self._optimize_hyperparameters(
                        embedding_run,
                        model_run_config,
                        x_train=x_train,
                        x_test=x_test,
                        y_train=y_train,
                        y_test=y_test,
                    )
                )
            else:
                classifier_run_config = model_run_config
            progress.update()

            classifier, effective_parameters = build_classifier(
                classifier_run_config,
                random_state=self.random_state,
            )

            classifier.fit(x_train, y_train)
            progress.update()

            y_pred = classifier.predict(x_test)
            progress.update()

            model_path = run_dir / "model.joblib"
            predictions_path = run_dir / "y_pred.joblib"

            joblib.dump(classifier, model_path)
            joblib.dump(y_pred, predictions_path)
            progress.update()

        metrics = compute_metrics(y_test, y_pred)

        run_summary = {
            "embedding_name": embedding_name,
            "embedding_type": str(embedding_run.get("type", "")).lower(),
            "model_name": model_name,
            "model_type": str(model_run_config.get("type", "")).lower(),
            "embedding_run_id": self.embedding_run_id,
            "ml_run_id": self.ml_run_id,
            "status": "generated",
            "output_dir": str(run_dir),
            "model_path": str(model_path),
            "predictions_path": str(predictions_path),
            "metrics": json_compatible(metrics),
            "parameters": json_compatible(effective_parameters),
            "configured_parameters": json_compatible(configured_parameters),
            "parameter_source": (
                "optuna"
                if self.hyperparameter_optimization_enabled
                else "config"
            ),
            "hyperparameter_optimization": json_compatible(optimization_summary),
            "train_shape": shape_of(x_train),
            "test_shape": shape_of(x_test),
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
                "No ML runs found in config. Set ml.types and ml.classifiers "
                "in ml_config.yaml"
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
        for embedding_run in selected_embedding_runs:
            for model_run in self.ml_runs:
                if not isinstance(model_run, dict):
                    continue
                runs.append(self._run_single_model(embedding_run, model_run))

        summary = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "embedding_run_path": str(self.embedding_run_path),
            "embedding_run_id": self.embedding_run_id,
            "ml_run_id": self.ml_run_id,
            "output_root": str(self.output_root),
            "random_state": self.random_state,
            "force_rerun": self.force_rerun,
            "embedding_filter": self.embedding_filter,
            "hyperparameter_optimization": {
                "enabled": self.hyperparameter_optimization_enabled,
                "framework": "optuna" if self.hyperparameter_optimization_enabled else None,
                "n_trials": self.hyperparameter_optimization_trials,
                "metric": "accuracy" if self.hyperparameter_optimization_enabled else None,
            },
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
