"""
Regression loss functions.
"""
import numpy as np

from dllib.metrics.loss import Loss
from dllib.tensors import Tensor


class MeanSquaredError(Loss):
    """Mean squared error."""

    def __init__(self) -> None:
        super().__init__(name="mean_square_error")

    def call(self, y_true: Tensor, y_pred: Tensor) -> float:
        return (1 / 2) * np.mean((y_pred - y_true) ** 2)

    def grad(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return (y_pred - y_true) / y_true.shape[0]
