"""
Classification loss functions.
"""
import numpy as np

from dllib.metrics.loss import Loss
from dllib.tensors import Tensor


class BinaryCrossEntropy(Loss):
    def __init__(self) -> None:
        super().__init__(name="binary_cross_entropy")

    def call(self, y_true: Tensor, y_pred: Tensor) -> float:
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def grad(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return (y_true - y_pred) / (y_pred * (1 - y_pred))
