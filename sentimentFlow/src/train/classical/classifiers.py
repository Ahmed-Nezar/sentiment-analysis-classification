from __future__ import annotations

import importlib
import inspect
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def _clean_model_params(run_config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in run_config.items()
        if key not in {"name", "type"}
    }


def _normalize_classifier_params(
    classifier_type: str,
    configured_params: dict[str, Any],
    random_state: int,
) -> dict[str, Any]:
    effective_params = dict(configured_params)

    # Config keys are normalized to lowercase during parsing, but a few
    # estimator arguments are case-sensitive in sklearn/xgboost.
    if "c" in effective_params and "C" not in effective_params:
        effective_params["C"] = effective_params.pop("c")

    if classifier_type in {
        "logistic_regression",
        "svm",
        "random_forest",
        "decision_tree",
        "xgboost",
    }:
        effective_params.setdefault("random_state", random_state)

    return effective_params


def _filter_supported_params(
    constructor: Any,
    params: dict[str, Any],
) -> dict[str, Any]:
    signature = inspect.signature(constructor)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return dict(params)

    supported_names = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }
    return {
        key: value
        for key, value in params.items()
        if key in supported_names
    }


def build_classifier(
    run_config: dict[str, Any], random_state: int
) -> tuple[Any, dict[str, Any]]:
    classifier_type = str(run_config.get("type", "")).strip().lower()
    configured_params = _clean_model_params(run_config)
    effective_params = _normalize_classifier_params(
        classifier_type,
        configured_params,
        random_state,
    )

    if classifier_type == "logistic_regression":
        effective_params = _filter_supported_params(LogisticRegression, effective_params)
        classifier = LogisticRegression(**effective_params)
        return classifier, effective_params

    if classifier_type == "svm":
        effective_params = _filter_supported_params(SVC, effective_params)
        classifier = SVC(**effective_params)
        return classifier, effective_params

    if classifier_type == "random_forest":
        effective_params = _filter_supported_params(
            RandomForestClassifier,
            effective_params,
        )
        classifier = RandomForestClassifier(**effective_params)
        return classifier, effective_params

    if classifier_type == "decision_tree":
        effective_params = _filter_supported_params(
            DecisionTreeClassifier,
            effective_params,
        )
        classifier = DecisionTreeClassifier(**effective_params)
        return classifier, effective_params

    if classifier_type == "xgboost":
        try:
            xgboost_module = importlib.import_module("xgboost")
            xgb_classifier = getattr(xgboost_module, "XGBClassifier")
        except Exception as exc:
            raise ImportError(
                "xgboost is required for classifier type 'xgboost'. "
                "Install it with: pip install xgboost"
            ) from exc

        effective_params = _filter_supported_params(xgb_classifier, effective_params)
        classifier = xgb_classifier(**effective_params)
        return classifier, effective_params

    raise ValueError(f"Unsupported classifier type: {classifier_type}")
