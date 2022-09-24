"""
Regression loss functions.
"""
import numpy as np

from dllib.metrics.loss import Loss
from dllib.tensors.tensor import Tensor


class MeanSquaredError(Loss):
    def __init__(self) -> None:
        super().__init__(name="mean_square_error")

    def call(self, y_true: Tensor, y_pred: Tensor) -> float:
        return (1 / 2) * np.mean((y_pred - y_true) ** 2)

    def grad(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return (1 / y_true.shape[0]) * (y_pred - y_true)


a = np.array([0, 1, 3, 5, 10])
b = np.array([0, 2, 1, 5, 9])
mse = MeanSquaredError()
print("loss:", mse(a, b))
print("grad:", mse.grad(a, b))
