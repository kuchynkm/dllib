from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dllib.model import NeuralNetwork
from dllib.tensors import Tensor


class ModelType(Enum):
    CLASSIFICATION = 1
    REGRESSION = 2


class SklearnClassifierWrapper:
    """Wrapper unifying dllib models with sklear estimators."""

    def __init__(self, model: NeuralNetwork) -> None:
        self.model = model

    def predict_proba(self, X: Tensor) -> Tensor:
        y_score = self.model.predict(X)
        y_score = np.reshape(y_score, newshape=(-1,))
        return np.stack([1 - y_score, y_score], axis=1)

    def predict(self, X: Tensor) -> Tensor:
        y_score = self.predict_proba(X)[:, 1]
        return (y_score > 0.5).astype(int)

    def fit(self, X: Tensor, y: Tensor, **kwargs: Any) -> SklearnClassifierWrapper:
        y = np.reshape(y, newshape=(-1, 1))
        self.model.fit(X, y, **kwargs)
        return self


class SklearnRegressionWrapper:
    """Wrapper unifying dllib models with sklear estimators."""

    def __init__(self, model: NeuralNetwork) -> None:
        self.model = model

    def predict(self, X: Tensor) -> Tensor:
        y_pred = self.model.predict(X)
        return np.reshape(y_pred, newshape=(-1,))

    def fit(self, X: Tensor, y: Tensor, **kwargs: Any) -> SklearnRegressionWrapper:
        y = np.reshape(y, newshape=(-1, 1))
        self.model.fit(X, y, **kwargs)
        return self


def evaluate_classification(
    y_true: Tensor, y_pred: Tensor, y_score: Tensor
) -> dict[str, float]:
    return {
        "ROC-AUC": roc_auc_score(y_true, y_score),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def evaluate_regression(y_true: Tensor, y_pred: Tensor) -> dict[str, float]:
    return {
        "R2": r2_score(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


@dataclass
class DataCatalogue:
    model_type: ModelType
    train: dict[str, Tensor]
    test: dict[str, Tensor]

    def evaluate(self) -> dict[str, dict[str, float]]:
        results: dict[str, dict[str, float]] = {}
        evaluation_fn: Any
        train_args = {"y_true": self.train["y"], "y_pred": self.train["y_pred"]}
        test_args = {"y_true": self.test["y"], "y_pred": self.test["y_pred"]}

        if self.model_type == ModelType.REGRESSION:
            evaluation_fn = evaluate_regression

        if self.model_type == ModelType.CLASSIFICATION:
            evaluation_fn = evaluate_classification
            train_args.update(y_score=self.train["y_score"])
            test_args.update(y_score=self.test["y_score"])

        results.update(train=evaluation_fn(**train_args))
        results.update(test=evaluation_fn(**test_args))

        return results


def fit_and_evaluate(
    estimator: Any, dataset: DataCatalogue, **fit_kwargs: Optional[Any]
) -> dict[str, dict[str, float]]:
    # construct pipeline
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("estimator", estimator),
        ]
    )

    # fit pipeline
    pipe.fit(dataset.train["X"], dataset.train["y"], **fit_kwargs)

    # generate train & test predictions
    dataset.train["y_pred"] = pipe.predict(dataset.train["X"])
    dataset.test["y_pred"] = pipe.predict(dataset.test["X"])

    if dataset.model_type == ModelType.CLASSIFICATION:
        dataset.train["y_score"] = pipe.predict_proba(dataset.train["X"])[:, 1]
        dataset.test["y_score"] = pipe.predict_proba(dataset.test["X"])[:, 1]

    # evaluate test predictions
    return dataset.evaluate()
