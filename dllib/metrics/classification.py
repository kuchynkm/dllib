"""
Classification loss functions.
"""
import numpy as np

from dllib.metrics.loss import Loss
from dllib.tensors import Tensor
from dllib.utils.math import EPSILON


class BinaryCrossEntropy(Loss):
    """Binary cross-entropy loss."""

    def __init__(self) -> None:
        super().__init__(name="binary_cross_entropy")

    def call(self, y_true: Tensor, y_pred: Tensor) -> float:
        first_term = y_true * np.log(y_pred + EPSILON)
        second_term = (1 - y_true) * np.log(1 - y_pred + EPSILON)
        return -np.mean(first_term + second_term)

    def grad(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return (y_pred - y_true) / len(y_true)
